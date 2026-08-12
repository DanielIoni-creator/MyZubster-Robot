"""Live / recorded demo of the glass-under-dispenser detector.

Reads frames from a Pi Camera, USB device, or video file, runs
:class:`vision.GlassDetector`, overlays the result with
:func:`vision.draw_detection`, and writes the annotated stream to a video
file (defaults to ``demo/vision_demo.mp4``). Optionally shows a live preview
window when ``--preview`` is passed.

Usage examples::

    # Run on a recorded video and save the demo
    python -m vision.demo --source path/to/video.mp4 --model models/yolov8n_glass.pt

    # Live preview from the Pi Camera
    python -m vision.demo --source picamera --preview --max-frames 600
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

LOG = logging.getLogger("vision.demo")

# Local import is deferred so --help works without ultralytics/cv2 present.
from .detector import GlassDetector, ROISpec, draw_detection  # noqa: E402

try:  # OpenCV is needed for IO + drawing.
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the glass detector on a stream and save a demo video."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="`picamera`, an integer device index, or a path to a video file.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to the trained YOLO weights (.pt). Optional for geometry demo.",
    )
    parser.add_argument(
        "--roi",
        default="0.35,0.40,0.30,0.45",
        help="ROI as 'x,y,w,h' in normalized [0,1] coordinates.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.45,
        help="Confidence threshold for the detector.",
    )
    parser.add_argument(
        "--output",
        default="demo/vision_demo.mp4",
        help="Output video path. Created if missing.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=600,
        help="Stop after this many frames. 0 means no limit (record until source ends).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a live preview window (requires a display).",
    )
    parser.add_argument(
        "--report",
        default="demo/vision_demo_report.json",
        help="Per-frame JSON report path. Set to '' to disable.",
    )
    return parser.parse_args(argv)


def parse_roi(s: str) -> ROISpec:
    parts = [float(p) for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"ROI must be 'x,y,w,h', got {s!r}")
    return ROISpec.from_normalized(*parts)


def open_source(source: str):
    if source == "picamera":
        try:
            from picamera2 import Picamera2  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Picamera2 not available; pass an integer device index or video file."
            ) from exc
        cam = Picamera2()
        cfg = cam.create_video_configuration(main={"size": (1280, 720)})
        cam.configure(cfg)
        cam.start()

        class _Wrap:
            def read(self):
                arr = cam.capture_array()
                if arr is None:
                    return False, None
                if cv2 is not None:
                    return True, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                return True, arr

            def release(self):
                cam.stop()

        return _Wrap()
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for non-picamera sources.")
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"could not open source: {source}")
    return cap


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)
    if cv2 is None:
        LOG.error("OpenCV is required")
        return 2
    roi = parse_roi(args.roi)
    detector = GlassDetector(
        model_path=args.model,
        roi=roi,
        confidence=args.confidence,
    )
    LOG.info("detector ready: model_loaded=%s roi=%s", detector.model_loaded, roi)

    cap = open_source(args.source)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Probe for fps + frame size
    fps = 30.0
    width = 1280
    height = 720
    ok, frame = cap.read()
    if not ok or frame is None:
        LOG.error("could not read first frame from %s", args.source)
        return 3
    height, width = frame.shape[:2]
    LOG.info("first frame: %dx%d", width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        LOG.error("could not open writer at %s", out_path)
        return 4

    report_path = Path(args.report) if args.report else None
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_file = report_path.open("w", encoding="utf-8")
    else:
        report_file = None

    frames_written = 0
    started = time.time()
    try:
        while True:
            if args.max_frames and frames_written >= args.max_frames:
                LOG.info("reached max-frames=%d, stopping", args.max_frames)
                break
            if frames_written == 0:
                # We already read the first frame.
                pass
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    LOG.info("source exhausted")
                    break
            t0 = time.time()
            result = detector.is_glass_under_target(frame)
            annotated = draw_detection(frame, result)
            writer.write(annotated)
            frames_written += 1
            if report_file is not None:
                row = result.to_dict()
                row["frame_index"] = frames_written
                row["latency_ms"] = round((time.time() - t0) * 1000, 1)
                report_file.write(json.dumps(row) + "\n")
                report_file.flush()
            if args.preview:
                cv2.imshow("MyZubster vision demo", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    LOG.info("preview quit")
                    break
            if frames_written % 30 == 0:
                LOG.info(
                    "frame=%d ok=%s best_conf=%.2f latency=%.1fms",
                    frames_written,
                    result.ok,
                    result.confidence,
                    (time.time() - t0) * 1000,
                )
    finally:
        writer.release()
        try:
            cap.release()
        except Exception:
            pass
        if report_file is not None:
            report_file.close()
        if args.preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
    elapsed = time.time() - started
    LOG.info(
        "wrote %d frames to %s in %.1fs (%.1f fps)",
        frames_written,
        out_path,
        elapsed,
        frames_written / max(elapsed, 1e-6),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())