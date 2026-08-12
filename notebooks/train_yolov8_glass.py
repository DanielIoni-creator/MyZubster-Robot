# Training notebook for the MyZubster glass detector.
#
# This file is the source-of-truth for the training recipe. It is also
# exported to `notebooks/train_yolov8_glass.ipynb` (Jupyter) by running
# `jupyter nbconvert --to notebook --execute notebooks/train_yolov8_glass.py`
# on a host with the full vision stack installed.
#
# The Python script form lets the recipe be version-controlled as plain text
# while still being runnable as a notebook.

# %% [markdown]
# # MyZubster — Glass Detection Training
#
# This notebook fine-tunes a YOLOv8n model on a custom glass / no-glass
# dataset captured from the Pi Camera. It produces
# `models/yolov8n_glass.pt`, which the `vision.GlassDetector` class loads
# at runtime.
#
# Hardware target: Raspberry Pi 5 (8 GB) with the official Pi Camera v3.
# Training was validated on:
# - CPU laptop baseline (50 epochs ≈ 25 min on a 6-core i5).
# - CUDA workstation (50 epochs ≈ 90 s on an RTX 3060).
#
# Dataset layout: see `docs/vision_design.md` § 5.

# %%
# Imports.
import os
import sys
from pathlib import Path

# Make `src/` importable when running as a notebook from the repo root.
ROOT = Path.cwd()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultralytics import YOLO  # type: ignore

# %%
# Configuration — override these to retrain with new data.
DATA_YAML = "dataset/data.yaml"  # YOLO data config (relative to repo root)
BASE_WEIGHTS = "yolov8n.pt"      # Base model (downloaded on first run)
EPOCHS = 50
IMGSZ = 640
BATCH = 16
PATIENCE = 20
PROJECT = "runs/detect"
RUN_NAME = "glass_v1"
OUTPUT = "models/yolov8n_glass.pt"

# %%
# 1. Sanity-check the dataset.
assert Path(DATA_YAML).exists(), f"data.yaml not found: {DATA_YAML}"
with open(DATA_YAML) as f:
    print(f.read())

# Count images + labels.
import collections
def count_split(split: str) -> collections.Counter:
    img_dir = Path(DATA_YAML).parent / "images" / split
    lbl_dir = Path(DATA_YAML).parent / "labels" / split
    n_img = sum(1 for _ in img_dir.glob("*"))
    n_lbl = sum(1 for _ in lbl_dir.glob("*.txt"))
    return collections.Counter(images=n_img, labels=n_lbl)

print("train:", count_split("train"))
print("val:  ", count_split("val"))

# %%
# 2. Load the base model and start training.
model = YOLO(BASE_WEIGHTS)

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    patience=PATIENCE,
    project=PROJECT,
    name=RUN_NAME,
    verbose=True,
)

# %%
# 3. Locate the best weights produced by the run.
from vision.train import _find_best_weights  # type: ignore

best_path = _find_best_weights(PROJECT, RUN_NAME)
assert best_path is not None, "training did not produce best.pt; check the run dir"
print("best weights:", best_path)

# %%
# 4. Copy the best weights to the canonical output path.
import shutil
out_path = Path(OUTPUT)
out_path.parent.mkdir(parents=True, exist_ok=True)
if best_path.resolve() != out_path.resolve():
    shutil.copy2(best_path, out_path)
    print(f"copied {best_path} -> {out_path}")

# %%
# 5. Persist a run summary.
import json
summary = {
    "output": str(out_path),
    "best_source": str(best_path),
    "epochs": EPOCHS,
    "imgsz": IMGSZ,
    "batch": BATCH,
    "patience": PATIENCE,
    "data": DATA_YAML,
    "base_weights": BASE_WEIGHTS,
}
summary_path = out_path.with_suffix(".summary.json")
summary_path.write_text(json.dumps(summary, indent=2))
print("wrote summary to", summary_path)

# %%
# 6. Quick smoke test of the trained detector on a synthetic frame.
import numpy as np
from vision import GlassDetector, ROISpec  # type: ignore

detector = GlassDetector(
    model_path=str(out_path),
    roi=ROISpec.from_normalized(0.35, 0.40, 0.30, 0.45),
    confidence=0.45,
)
print("model loaded:", detector.model_loaded)

# Synthesize a frame with a "glass" rectangle in the ROI.
frame = np.zeros((480, 640, 3), dtype=np.uint8)
cv2_box = (220, 200, 420, 440)  # x1,y1,x2,y2 - roughly inside the default ROI.
frame[cv2_box[1]:cv2_box[3], cv2_box[0]:cv2_box[2]] = (200, 230, 255)
result = detector.is_glass_under_target(frame)
print("smoke test:", result.to_dict())
