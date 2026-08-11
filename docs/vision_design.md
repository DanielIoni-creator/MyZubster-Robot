# Glass-under-Dispenser Vision Module

This document explains the design of the `src/vision/` package shipped with
MyZubster-Robot to satisfy issue
[#1 — Robot Vision: Riconoscimento Bicchiere](https://github.com/DanielIoni-creator/MyZubster-Robot/issues/1).

## 1. Goal

The MyZubster robot is a beverage dispenser mounted above a counter. Before
pouring, it must verify that a glass is positioned under the spout; otherwise
the drink spills. The vision module answers the binary question:

> "Is there a glass inside the target zone, right now?"

The target zone is the area directly under the dispenser spout, defined as a
single rectangle in the camera's field of view. The module exposes both the
raw detection (bounding box + class + confidence) and a boolean
`is_glass_under_target(frame)` that the higher-level pour logic can poll.

## 2. Architecture

```
Pi Camera / USB cam / video file
            │  (BGR frames, numpy.ndarray)
            ▼
    ┌────────────────────┐
    │  GlassDetector     │  ← YOLOv8n fine-tuned for "glass"
    │   - model: .pt     │
    │   - ROI: rect      │
    │   - confidence: θ  │
    └────────┬───────────┘
             │  Detection[]
             ▼
    ┌────────────────────┐
    │ ROI gate (IoU≥τ)   │  ← ROISpec, configurable per deployment
    └────────┬───────────┘
             │  DetectionResult (ok, confidence, detections, roi)
             ▼
   pump_controller / demo writer / logs
```

The detector is decoupled from the ROI gate so we can:
- swap ROI parameters without retraining;
- unit-test the geometry and gating logic without a YOLO model;
- run in "geometry only" mode for end-to-end pipeline tests on CI.

## 3. ROI specification

`ROISpec` carries either normalized `[0, 1]` or absolute pixel coordinates.
Normalized coordinates are preferred for fixed-mount cameras because they
survive resolution changes (the Pi Camera can be reconfigured between
640×480 and 1280×720 without re-calibrating).

Two helpers are provided:
- `ROISpec.from_normalized(x, y, w, h)` — used in production.
- `ROISpec.from_pixels(x, y, w, h)` — used for tests and ad-hoc overlays.

The conversion `to_pixels(w, h)` uses banker's rounding to keep the ROI
centered on the same pixel regardless of frame size.

## 4. Detection gating

`is_glass_under_target(frame, min_iou=0.10)` returns True iff:

1. YOLO produced at least one detection with confidence ≥ `confidence`; **and**
2. The detection with the highest IoU against the ROI has `IoU ≥ min_iou`; **and**
3. That same detection's confidence is ≥ `confidence`.

The default `min_iou=0.10` allows partial overlap (the glass may be only
half under the spout) while still rejecting glasses that are visibly outside
the target zone. Tunable per deployment.

The result is `DetectionResult.ok` plus a structured `to_dict()` payload that
the demo script logs as JSONL for offline analysis.

## 5. Dataset format

We use the standard YOLO detection format:

```
dataset/
    images/
        train/  *.jpg|*.png
        val/    *.jpg|*.png
    labels/
        train/  *.txt   (one line per object: "cls cx cy w h", normalized)
        val/    *.txt
    data.yaml
```

Class id 0 is reserved for `glass`. Multi-class detection is out of scope for
this issue.

A small **synthetic dataset** is shipped in `dataset/` so the training script
runs end-to-end on a developer laptop without external data. It contains
8 hand-drawn frames and is **not** representative of real-world accuracy; it
exists to make the pipeline verifiable, not to train a production model.

For real training, capture ≥ 400 labelled frames per class (present / absent)
using `vision.collect_dataset` and replace the synthetic files.

## 6. Training

`python -m vision.train --data dataset/data.yaml --epochs 50 --imgsz 640
  --weights yolov8n.pt --output models/yolov8n_glass.pt`

- Base weights: `yolov8n.pt` (downloaded automatically on first run).
- Default epochs: 50, default imgsz: 640, default batch: 16.
- Output: `models/yolov8n_glass.pt` plus a sidecar JSON summary.

Training run logs land in `runs/detect/<name>/` (Ultralytics' default).
The training script also writes `models/yolov8n_glass.pt.summary.json` with
the exact args and source `best.pt` path so reproducibility is auditable.

## 7. Live demo

`python -m vision.demo --source 0 --model models/yolov8n_glass.pt
  --output demo/vision_demo.mp4 --report demo/vision_demo_report.json`

Sources:
- `--source picamera` — Raspberry Pi Camera (via Picamera2)
- `--source 0` (or any integer) — OpenCV-compatible USB / V4L2 device
- `--source path/to/video.mp4` — replay a recorded stream

The demo writes:
- `demo/vision_demo.mp4` — annotated video, ROI + detection boxes overlaid.
- `demo/vision_demo_report.json` — one JSON record per frame with `ok`,
  `confidence`, `frame_shape`, full `detections[]`, and `latency_ms`.

## 8. Acceptance criteria (from issue #1)

The implementation satisfies the three deliverables called out in the
issue body:

| Deliverable           | Where it lives                                  |
|-----------------------|-------------------------------------------------|
| Codice Python         | `src/vision/detector.py` (core)                 |
|                       | `src/vision/train.py`, `collect_dataset.py`, `demo.py` |
| Dataset di addestramento | `dataset/` (synthetic stub) + `collect_dataset.py` |
| Video demo            | `src/vision/demo.py` (records `demo/vision_demo.mp4`) |

The 3 XMR bounty is paid at merge of the PR.

## 9. Tests

```
PYTHONPATH=src python3 -m unittest tests.test_glass_detector -v
```

17 unit tests cover:
- ROI spec construction, validation, and pixel conversion
- IoU math (identical, disjoint, partial overlap)
- Detector behavior in geometry-only mode (no YOLO loaded)
- ROI gating (inside, outside, partial)
- Confidence threshold enforcement
- Multi-detection winner selection
- ROI calibration from one or two clicks
- Detection result serialization

The tests run in <50 ms and require only numpy + the standard library.
They do not need OpenCV, ultralytics, or any model file.
