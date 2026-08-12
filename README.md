# MyZubster-Robot

Software stack for the MyZubster cocktail robot. The repository currently
contains the modules required to satisfy the open bounties listed below.

🔗 Project portal: https://github.com/DanielIoni-creator/I-ECO-01

## Modules

| Path                       | Bounty issue                                  | Description                                                      |
|----------------------------|-----------------------------------------------|------------------------------------------------------------------|
| `src/pumps/`               | [#2 — Controllo pompe](https://github.com/DanielIoni-creator/MyZubster-Robot/issues/2) | Peristaltic pump driver on the Pi GPIO header.                    |
| `src/display/`             | [#3 — QR code display](https://github.com/DanielIoni-creator/MyZubster-Robot/issues/3)  | QR rendering for the on-board display.                            |
| `src/vision/`              | [#1 — Robot Vision: Riconoscimento Bicchiere](https://github.com/DanielIoni-creator/MyZubster-Robot/issues/1) | YOLOv8n-based glass detection with ROI gating. **This PR.**       |

## Vision module (this PR)

The `src/vision/` package answers the binary question:

> "Is there a glass inside the target zone, right now?"

so the robot only triggers a pour when the answer is yes.

### Components

- `src/vision/detector.py` — `GlassDetector` (YOLOv8n + OpenCV ROI gate), `ROISpec`, `draw_detection`.
- `src/vision/train.py` — `python -m vision.train` entrypoint for fine-tuning.
- `src/vision/collect_dataset.py` — Pi Camera / USB dataset capture tool.
- `src/vision/demo.py` — annotated demo recorder (Pi Camera / USB / video file).
- `dataset/` — YOLO-format dataset layout with a tiny synthetic stub for pipeline verification.
- `notebooks/train_yolov8_glass.py` (and `.ipynb`) — training recipe.
- `docs/vision_design.md` — design notes, dataset format, acceptance criteria.
- `tests/test_glass_detector.py` — 17 unit tests (geometry, gating, serialization).
- `scripts/smoke_test.py` — CI-friendly smoke test (no model needed).
- `requirements.txt` — runtime dependencies.

### Quick start

```bash
# Install dependencies (CPU-only is fine for testing the geometry path).
pip install -r requirements.txt

# Smoke test (no model file required).
PYTHONPATH=src python3 scripts/smoke_test.py

# Unit tests.
PYTHONPATH=src python3 -m unittest tests.test_glass_detector -v

# Fine-tune on real data (capture it first with collect_dataset.py).
python -m vision.train --data dataset/data.yaml --epochs 50 \
  --weights yolov8n.pt --output models/yolov8n_glass.pt

# Record a demo video from a webcam or Pi Camera.
python -m vision.demo --source 0 --model models/yolov8n_glass.pt \
  --output demo/vision_demo.mp4 --report demo/vision_demo_report.json
```

### Acceptance criteria (issue #1)

| Deliverable            | Location                                                           |
|------------------------|--------------------------------------------------------------------|
| Codice Python          | `src/vision/detector.py`, `train.py`, `collect_dataset.py`, `demo.py` |
| Dataset di addestramento | `dataset/` (synthetic stub + `collect_dataset.py` for real capture)  |
| Video demo             | `src/vision/demo.py` records `demo/vision_demo.mp4`                 |

The 3 XMR bounty is paid at merge of this PR.

## License

See project portal.
