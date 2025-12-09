# vision-in-store



https://github.com/user-attachments/assets/a55442e2-d3c4-4f2d-9214-39ee44d1b39b



Early proof of concept for a smart retail camera pipeline using Python, OpenCV, TorchVision and Flask.

The goal is to instrument a retail space from one or more cameras: detect people and other COCO objects, track their motion, and expose a clean JSON + web UI surface that can later grow into age/sex estimation and product-level analytics (gaze, pick-up events, shelf occupancy, etc.).

---

## Features

### Inputs

- **Live camera** via `getUserMedia` in the browser.
- **Video file** upload, with playback controls and adjustable playback speed.
- **Single frame / image** upload for one-off analysis and debugging.

### Detection & tracking

- **Faster R-CNN ResNet50-FPN v2 (TorchVision, COCO weights)**.
- Runs on:
  - **CPU** by default (Apple Silicon M-series friendly; MPS disabled to avoid Metal errors).
  - **CUDA** if a CUDA GPU is available.
- Detects **any COCO category** (person, cell phone, bottle, backpack, etc.).
- **Dynamic object filters**:
  - Type strings like `person, cell phone, bottle` into the UI.
  - Add (`+`) / remove (`×`) filters; only matching classes are kept.
  - With no filters, all COCO classes are allowed.
- **Simple multi-object tracking** with ID assignment:
  - Each detection gets a `track_id`.
  - Tracks persist across frames while objects stay spatially close.
  - Per-track history of positions and approximate size.

### Visualisation (front-end)

- Single-page **Flask UI** with three modes:
  - **Camera** – live webcam stream.
  - **Video** – local MP4/WEBM/etc. file.
  - **Frame** – individual image.
- Overlays drawn on a dedicated canvas above the media element:
  - **Bounding boxes** with labels and track IDs.
  - **Bezier-smoothed trajectories**:
    - Each tracked object leaves a smooth path rather than jagged poly-lines.
    - Visibility can be toggled to show only lines, only boxes, or both.
  - **Heatmap “dwell” circles** (optional):
    - Circles centred on track paths.
    - Size scales with object size.
    - Colour fades from **green → red** as dwell time increases
      (short presence = cooler; long presence = hotter), mimicking a thermal map.
- **Controls panel**:
  - Score threshold slider (filter low-confidence detections).
  - Toggle **boxes**, **trajectories**, **heatmap** independently.
  - **Analysis FPS** slider – control how often frames are sent to the model.
  - **Video speed** slider – slow down or speed up playback while analysing.
  - Object filter input + chips for quick **add/remove** of target classes.

### Sessions, history & API

- Each analysis run gets a **session ID**:
  - Generated when you press “Start analysis”.
  - Included in the JSON summary for downstream analytics.
- In-memory **history**:
  - Recent per-frame summaries with timestamp and `session_id`.
  - Exposed via `/history` as JSON.
- Core JSON endpoint:
  - `POST /analyze_frame` with an image and form fields:
    - `score_threshold`
    - `enable_tracking`
    - `classes` (comma-separated filters)
    - `session_id`
  - Returns bounding boxes, labels, scores, tracks (with trajectories & radii),
    and counts suitable for downstream analytics.

### Security / transport

- Flask app runs with a **self-signed HTTPS certificate**:
  - Generated automatically on startup with `openssl`.
  - Served on `https://localhost:5000` and also over your LAN IP.
- This satisfies browser requirements for **camera access**:
  - Camera mode will work on HTTPS or on `localhost` in an otherwise
    insecure context.
  - If the page is opened over plain HTTP on a raw LAN IP, the UI shows
    a clear warning and explains why the camera is blocked.

### Debuggability

- Compact **“Recent frames”** feed with people/track counts.
- On-screen **debug log** of client-side events:
  - Camera errors, analysis timings, settings changes, etc.
  - Also POSTs to `/client_event` for structured server-side logging.
- `/health` endpoint:
  - Returns status, active device (CPU/CUDA), and history length.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

This will:

* Load the Faster R-CNN model (COCO weights) on CPU or CUDA.
* Generate a short-lived self-signed certificate (if not already present).
* Start Flask on `https://0.0.0.0:5000`.

Then:

1. Open your browser at **[https://localhost:5000](https://localhost:5000)**
   (or `https://<your-LAN-IP>:5000` if you want to test from another device).
2. Accept the browser’s warning for the self-signed certificate.
3. Use the **Camera / Video / Frame** mode toggle and the control panel
   to experiment with detection, tracking, trajectories and heatmap overlays.

---

## Roadmap / possible extensions

* Age and sex estimation per detected person.
* Product-level analytics:

  * Gaze estimation and dwell time per shelf region.
  * Pick-up / put-back event detection.
  * Shelf occupancy and out-of-stock alerts.
* Persistence to a real datastore (PostgreSQL / DuckDB) instead of in-memory history.
* Multi-camera support and aggregated heatmaps across zones.






