"""YOLOv8n fine-tuning entrypoint for the glass-under-dispenser detector.

The training pipeline is intentionally lightweight so it can run on a
developer laptop (CPU) for a quick baseline, and on a single GPU on the
Raspberry Pi 5 (or any CUDA host) for the real run.

Dataset layout (YOLO format)::

    dataset/
        images/
            train/   *.jpg
            val/     *.jpg
        labels/
            train/   *.txt    # one line per glass: "cls cx cy w h"
            val/     *.txt
        data.yaml

Only one class is used: ``glass`` (id 0).

Usage::

    python -m vision.train --data dataset/data.yaml --epochs 50 --imgsz 640 \
        --weights yolov8n.pt --output models/yolov8n_glass.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

LOG = logging.getLogger("vision.train")


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8n fine-tuning entrypoint for glass-under-dispenser detection")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="Base YOLOv8n weights (downloaded automatically if missing).",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="", help="cuda:0, cpu, or empty for auto")
    parser.add_argument(
        "--output",
        default="models/yolov8n_glass.pt",
        help="Destination weights path (created if missing).",
    )
    parser.add_argument(
        "--project",
        default="runs/detect",
        help="Ultralytics project directory for intermediate runs.",
    )
    parser.add_argument("--name", default="glass_v1")
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args(argv)


def ensure_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "ultralytics is required for training. Install it with:\n"
            "  pip install ultralytics\n"
            f"Original error: {exc}"
        )
    return YOLO


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)

    if not os.path.exists(args.data):
        LOG.error("data.yaml not found at %s", args.data)
        return 2

    YOLO = ensure_ultralytics()
    LOG.info("loading base weights from %s", args.weights)
    model = YOLO(args.weights)

    out_dir = Path(args.output).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info(
        "starting training: data=%s epochs=%d imgsz=%d batch=%d device=%s",
        args.data,
        args.epochs,
        args.imgsz,
        args.batch,
        args.device or "auto",
    )
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=args.project,
        name=args.name,
        patience=args.patience,
        verbose=True,
    )

    best_path = _find_best_weights(args.project, args.name)
    if best_path is None:
        LOG.error("training did not produce a best.pt; check the run directory")
        return 3
    LOG.info("best weights at %s", best_path)
    if str(best_path) != str(Path(args.output).resolve()):
        # Copy best.pt to the requested output path.
        import shutil

        shutil.copy2(best_path, args.output)
        LOG.info("copied best weights to %s", args.output)

    summary = {
        "output": args.output,
        "best_source": str(best_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device or "auto",
        "data": args.data,
    }
    summary_path = Path(args.output).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    LOG.info("wrote summary to %s", summary_path)
    return 0


def _find_best_weights(project: str, name: str) -> Optional[Path]:
    base = Path(project) / name / "weights" / "best.pt"
    if base.exists():
        return base
    # Fallback: scan the project for any best.pt.
    project_dir = Path(project)
    if not project_dir.exists():
        return None
    candidates = sorted(project_dir.glob(f"**/{name}/weights/best.pt"))
    return candidates[-1] if candidates else None


if __name__ == "__main__":
    sys.exit(main())