# vision-in-store

Early proof-of-concept smart retail camera pipeline using Python, OpenCV, TorchVision and Flask.

The app takes a live camera, video file or single frame, runs a Faster R-CNN detector (COCO), and gives an interactive web UI with object filters, trajectory visualisation, and heatmap-style “dwell” overlays. It’s designed as a playground for people / basket / product analytics (gaze, pick-ups, shelf occupancy, etc.).
