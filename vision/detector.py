import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)


class StoreVision:
    """Runs store-focused visual analysis on frames: detects people and basket-like objects."""

    def __init__(self, score_threshold: float = 0.7, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.categories = weights.meta.get("categories", [])
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        self.preprocess = weights.transforms()
        self.score_threshold = score_threshold

        # Classes in COCO that approximate baskets/carts/bags for a simple POC.
        self._basket_like_labels = {"handbag", "backpack", "suitcase"}

    def _filter_detections(self, outputs: dict) -> list[dict]:
        """Converts raw model outputs into a filtered list of detected objects."""
        scores = outputs["scores"].detach().cpu().numpy()
        labels = outputs["labels"].detach().cpu().numpy()
        boxes = outputs["boxes"].detach().cpu().numpy()

        results: list[dict] = []
        for score, label_idx, box in zip(scores, labels, boxes, strict=False):
            if score < self.score_threshold:
                continue

            label_idx_int = int(label_idx)
            label_name = (
                self.categories[label_idx_int]
                if 0 <= label_idx_int < len(self.categories)
                else str(label_idx_int)
            )

            x1, y1, x2, y2 = box.astype(int).tolist()
            results.append(
                {
                    "label": label_name,
                    "score": float(score),
                    "box": (x1, y1, x2, y2),
                }
            )
        return results

    def _annotate_frame(self, frame_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draws bounding boxes and labels directly onto the frame."""
        annotated = frame_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]
            score = det["score"]

            color = (0, 255, 0) if label == "person" else (255, 0, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            text = f"{label} {score:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - th - baseline),
                (x1 + tw, y1),
                color,
                thickness=-1,
            )
            cv2.putText(
                annotated,
                text,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def _summarize_scene(self, detections: list[dict]) -> dict:
        """Builds a compact, model-agnostic summary for downstream analytics."""
        person_detections = [d for d in detections if d["label"] == "person"]
        basket_detections = [
            d for d in detections if d["label"] in self._basket_like_labels
        ]

        summary = {
            "people_count": len(person_detections),
            "basket_count": len(basket_detections),
            # Placeholders for later age/sex/product recognition models.
            "age_estimates": [],
            "sex_estimates": [],
            "product_regions": [],
        }

        return summary

    def analyze(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        """Runs detection on a BGR frame and returns an annotated frame and a summary."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = self.preprocess(rgb).to(self.device)

        with torch.no_grad():
            outputs = self.model([img_tensor])[0]

        detections = self._filter_detections(outputs)
        annotated = self._annotate_frame(frame_bgr, detections)
        summary = self._summarize_scene(detections)

        return annotated, summary
