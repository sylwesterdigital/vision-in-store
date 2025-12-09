import logging
import math
import sys
import time
from collections import deque
from typing import Deque, Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)

import subprocess
import os

# -----------------------------------------------------------------------------
# Logging setup: structured, verbose logging to stdout
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
)
log = logging.getLogger("vision-in-store")

app = Flask(__name__)
history: Deque[dict] = deque(maxlen=200)

# -----------------------------------------------------------------------------
# Model/device init with explicit logging (MPS disabled – CPU only on M2)
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    device = "cuda"
    log.info("Initialising detection model on device=%s (CUDA)", device)
else:
    device = "cpu"
    log.info(
        "Initialising detection model on device=cpu "
        "(MPS explicitly disabled; avoiding Metal command-buffer errors on Apple Silicon)"
    )

t0 = time.perf_counter()
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device)
model.eval()
t1 = time.perf_counter()
log.info("Model loaded in %.3fs", t1 - t0)

# COCO categories: index 0 == __background__, 1 == person, etc.
COCO_CATEGORIES: List[str] = weights.meta.get("categories", [])

MAX_TRACK_HISTORY = 40
MAX_TRACK_DISTANCE = 80.0
MAX_TRACK_AGE = 60

next_track_id = 1
tracks: Dict[int, dict] = {}
frame_index = 0


def empty_summary() -> dict:
    """Returns a neutral summary when no frame or detections are available."""
    return {
        "people_count": 0,
        "basket_count": 0,
        "age_estimates": [],
        "sex_estimates": [],
        "product_regions": [],
        "detections": [],
        "tracks": [],
        "session_id": None,
    }


def _update_tracks(
    detections: List[dict],
    current_frame: int,
    enable_tracking: bool,
) -> Tuple[List[dict], List[dict]]:
    """Maintains simple multi-object tracks and returns trajectories."""
    global next_track_id, tracks

    if not enable_tracking:
        if tracks:
            log.debug("Tracking disabled; clearing %d existing tracks", len(tracks))
        tracks.clear()
        for det in detections:
            det["track_id"] = None
        return detections, []

    # Mark all tracks unseen for this frame
    for track in tracks.values():
        track["seen"] = False

    for det in detections:
        x = float(det["x"])
        y = float(det["y"])
        w = float(det["w"])
        h = float(det["h"])
        cx = x + w / 2.0
        cy = y + h / 2.0

        best_id: Optional[int] = None
        best_dist = MAX_TRACK_DISTANCE

        for track_id, track in tracks.items():
            tx, ty = track["last_center"]
            dist = math.hypot(cx - tx, cy - ty)
            if dist < best_dist:
                best_dist = dist
                best_id = track_id

        if best_id is None:
            track_id = next_track_id
            next_track_id += 1
            tracks[track_id] = {
                "id": track_id,
                "last_center": (cx, cy),
                "last_size": (w, h),
                "points": [(cx, cy)],
                "last_frame": current_frame,
                "seen": True,
            }
            log.debug("Created new track id=%d at frame=%d", track_id, current_frame)
        else:
            track = tracks[best_id]
            track_id = best_id
            track["last_center"] = (cx, cy)
            track["last_size"] = (w, h)
            track["last_frame"] = current_frame
            track["seen"] = True
            pts = track["points"]
            pts.append((cx, cy))
            if len(pts) > MAX_TRACK_HISTORY:
                del pts[0]

        det["track_id"] = track_id

    # Drop stale tracks
    old_ids = [
        tid for tid, tr in tracks.items()
        if current_frame - tr["last_frame"] > MAX_TRACK_AGE
    ]
    for tid in old_ids:
        log.debug("Dropping stale track id=%d", tid)
        del tracks[tid]

    # Build trajectories payload with approximate radius per track
    trajectories: List[dict] = []
    for track in tracks.values():
        pts = [{"x": float(px), "y": float(py)} for (px, py) in track["points"]]
        w, h = track.get("last_size", (50.0, 50.0))
        area = max(w * h, 1.0)
        radius = math.sqrt(area) / 2.0
        trajectories.append(
            {
                "id": track["id"],
                "points": pts,
                "radius": float(radius),
            }
        )

    return detections, trajectories


def analyze_np_frame(
    frame_bgr: np.ndarray,
    score_threshold: float = 0.6,
    enable_tracking: bool = True,
    current_frame: int = 0,
    class_filters: Optional[List[str]] = None,
) -> dict:
    """
    Runs a detector on a BGR frame and returns summary with boxes and tracks.

    class_filters: list of substrings (e.g. ["person", "phone"]) – if provided,
    only detections whose COCO label name contains any of these substrings
    (case-insensitive) are kept. If None or empty, all COCO classes pass.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        log.warning("Empty frame passed into analyze_np_frame")
        return empty_summary()

    h, w = frame_bgr.shape[:2]
    log.debug(
        "Running analysis on frame: shape=(%d,%d), score_threshold=%.2f, tracking=%s, frame_index=%d, filters=%s",
        h,
        w,
        score_threshold,
        enable_tracking,
        current_frame,
        class_filters,
    )

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model([img_tensor])[0]
    t1 = time.perf_counter()
    log.debug("Model forward pass took %.3fs", t1 - t0)

    boxes = out["boxes"].cpu().numpy()
    labels = out["labels"].cpu().numpy()
    scores = out["scores"].cpu().numpy()

    filters_lc = None
    if class_filters:
        filters_lc = [s.lower() for s in class_filters if s.strip()]

    detections: List[dict] = []

    for box, label_id, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue

        idx = int(label_id)
        if 0 <= idx < len(COCO_CATEGORIES):
            label_name = COCO_CATEGORIES[idx]
        else:
            label_name = f"class_{idx}"

        # Apply optional text filters (substring match in label_name)
        if filters_lc:
            name_lc = label_name.lower()
            if not any(f in name_lc for f in filters_lc):
                continue

        x1, y1, x2, y2 = box
        w_box = x2 - x1
        h_box = y2 - y1
        detections.append(
            {
                "x": float(x1),
                "y": float(y1),
                "w": float(w_box),
                "h": float(h_box),
                "label": label_name,
                "score": float(score),
            }
        )

    people_count = sum(1 for d in detections if d.get("label") == "person")

    log.info(
        "Frame %d: %d detections (%d people) (threshold=%.2f, filters=%s)",
        current_frame,
        len(detections),
        people_count,
        score_threshold,
        filters_lc,
    )

    detections, trajectories = _update_tracks(
        detections, current_frame, enable_tracking
    )

    summary = {
        "people_count": people_count,
        "basket_count": 0,
        "age_estimates": [],
        "sex_estimates": [],
        "product_regions": [],
        "detections": detections,
        "tracks": trajectories,
        "session_id": None,  # filled in /analyze_frame
    }
    return summary


@app.route("/")
def index() -> str:
    log.info("Serving index.html to %s", request.remote_addr)
    return render_template("index.html")


@app.route("/analyze_frame", methods=["POST"])
def analyze_frame():
    """Accepts a single image, runs analysis, and returns a JSON summary."""
    global frame_index

    try:
        file = request.files.get("image")
        if file is None:
            log.warning("/analyze_frame called without 'image' file")
            return jsonify({"error": "missing_image"}), 400

        raw = file.read()
        log.debug(
            "/analyze_frame: received image from %s, bytes=%d",
            request.remote_addr,
            len(raw),
        )

        nparr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            log.error("OpenCV failed to decode uploaded image")
            return jsonify({"error": "decode_failed"}), 400

        score_threshold = request.form.get("score_threshold", type=float) or 0.6
        score_threshold = max(0.05, min(score_threshold, 0.95))
        enable_tracking = request.form.get("enable_tracking", "1") == "1"

        classes_field = request.form.get("classes", "") or ""
        if classes_field.strip():
            class_filters = [
                s.strip().lower()
                for s in classes_field.split(",")
                if s.strip()
            ]
        else:
            class_filters = None

        session_id = request.form.get("session_id") or "default"

        frame_index += 1
        t0 = time.perf_counter()
        summary = analyze_np_frame(
            frame,
            score_threshold=score_threshold,
            enable_tracking=enable_tracking,
            current_frame=frame_index,
            class_filters=class_filters,
        )
        t1 = time.perf_counter()
        summary["session_id"] = session_id

        log.info(
            "/analyze_frame finished: frame=%d, people=%d, detections=%d, elapsed=%.3fs, session_id=%s",
            frame_index,
            summary.get("people_count", 0),
            len(summary.get("detections", [])),
            t1 - t0,
            session_id,
        )

        history.append(
            {"ts": time.time(), "session_id": session_id, "summary": summary}
        )
        return jsonify({"summary": summary})
    except Exception:
        log.exception("Unhandled error in /analyze_frame")
        return jsonify({"error": "internal_error"}), 500


@app.route("/history", methods=["GET"])
def get_history():
    """Returns recent analysis summaries with timestamps."""
    log.debug("/history requested by %s, items=%d", request.remote_addr, len(history))
    return jsonify(list(history))


@app.route("/client_event", methods=["POST"])
def client_event():
    """Receives client-side debug telemetry for button clicks, camera events, etc."""
    data = request.get_json(silent=True) or {}
    event = data.get("event", "unknown")
    level = data.get("level", "INFO").upper()
    msg = data.get("message", "")
    extra = {k: v for k, v in data.items() if k not in ("event", "level", "message")}

    line = "client_event=%s message=%s extra=%s" % (event, msg, extra)
    if level == "DEBUG":
        log.debug(line)
    elif level == "WARNING":
        log.warning(line)
    elif level == "ERROR":
        log.error(line)
    else:
        log.info(line)

    return jsonify({"ok": True})


@app.route("/health", methods=["GET"])
def health():
    """Simple health probe to confirm server responsiveness."""
    return jsonify({"status": "ok", "device": device, "history_len": len(history)})


def generate_temp_ssl_cert():
    ssl_dir = "/tmp/ssl"
    os.makedirs(ssl_dir, exist_ok=True)
    cert_file = os.path.join(ssl_dir, "server.crt")
    key_file = os.path.join(ssl_dir, "server.key")

    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        subprocess.run(
            [
                "openssl",
                "req",
                "-new",
                "-newkey",
                "rsa:2048",
                "-days",
                "1",
                "-nodes",
                "-x509",
                "-keyout",
                key_file,
                "-out",
                cert_file,
                "-subj",
                "/C=US/ST=Denial/L=Springfield/O=Dis/CN=localhost",
            ],
            check=True,
        )
    return cert_file, key_file


# ── RUN: disable the auto-reloader (critical), keep debugger on ────────────────
if __name__ == "__main__":
    cert, key = generate_temp_ssl_cert()
    log.info("Starting Flask app on 0.0.0.0:5000 with HTTPS (self-signed cert)")
    app.run(
        host="0.0.0.0",
        port=5000,
        ssl_context=(cert, key),
        debug=True,          # debugger on
        use_reloader=False,  # prevents fork/restart crashes
        threaded=True,       # fine for these routes
    )
