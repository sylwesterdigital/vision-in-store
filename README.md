# vision-in-store



https://github.com/user-attachments/assets/a55442e2-d3c4-4f2d-9214-39ee44d1b39b



Early proof of concept for a smart retail camera pipeline using Python, OpenCV, TorchVision and Flask.

## Features

- Live camera or video file input.
- Person and basket-like object detection via Faster R-CNN (COCO).
- MJPEG stream exposed to a simple Flask dashboard.
- JSON endpoint with basic per-frame summary for analytics.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
