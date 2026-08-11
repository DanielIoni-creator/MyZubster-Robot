"""Dataset collection helper for the glass-under-dispenser detector.

Captures frames from a Raspberry Pi Camera (Picamera2) or any OpenCV-compatible
video source and writes them to ``dataset/images/raw/`` with a sibling
``labels.json`` describing each capture. A simple CLI is provided to tag a
frame as ``glass_present`` / ``glass_absent`` so we can bootstrap the
classifier before YOLO fine-tuning is run.

This script is intentionally tolerant of environments where Picamera2 is not
installed: it falls back to ``cv2.VideoCapture(index)`` and accepts any video
file path, which makes it easy to record with a phone first and replay the
file on the Pi.

Usage examples::

    # Capture from the default Pi Camera
    python -m vision.collect_dataset --source picamera --output dataset

    # Capture from a USB webcam (device 0)
    python -m vision.collect_dataset --source 0 --output dataset

    # Replay a recorded video
    python -m vision.collect_dataset --source path/to/video.mp4 --output dataset
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

LOG = logging.getLogger("vision.collect")


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture frames from a Pi Camera or USB device for the glass detection dataset."
    )
    parser.add_argument(
        "--source",
        default="picamera",
        help="`picamera`, an integer device index, or a path to a video file.",
    )
    parser.add_argument(
        "--output",
        default="dataset",
        help="Dataset root. Subdirs `images/raw` and `manifests` are created.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds between auto-captures when no key is pressed.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Capture width (ignored for video file sources).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Capture height (ignored for video file sources).",
    )
    return parser.parse_args(argv)


def open_source(source: str, width: int, height: int):
    """Return a context-manager-like object exposing ``read() -> (ok, frame)``.

    Tries Picamera2 first when ``source == "picamera"``, otherwise falls back
    to OpenCV ``VideoCapture``.
    """
    if source == "picamera":
        try:
            from picamera2 import Picamera2  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Picamera2 not available; install with `sudo apt install python3-picamera2` "
                "or pass a different --source (e.g. an integer device index)."
            ) from exc

        cam = Picamera2()
        cfg = cam.create_video_configuration(main={"size": (width, height)})
        cam.configure(cfg)
        cam.start()

        class _Wrap:
            def read(self):
                arr = cam.capture_array()
                if arr is None:
                    return False, None
                # Picamera2 returns RGB; convert to BGR for OpenCV parity.
                try:
                    import cv2  # type: ignore

                    return True, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                except Exception:
                    return True, arr

            def release(self):
                cam.stop()

        return _Wrap()

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "OpenCV (cv2) is required to use a non-picamera source."
        ) from exc
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)
    out = Path(args.output).resolve()
    img_dir = out / "images" / "raw"
    manifest_dir = out / "manifests"
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cv2  # type: ignore
    except Exception as exc:
        LOG.error("OpenCV is required: %s", exc)
        return 2

    LOG.info("opening source: %s", args.source)
    cap = open_source(args.source, args.width, args.height)
    LOG.info("press SPACE to save as 'present', X for 'absent', Q to quit")

    manifest_path = manifest_dir / "captures.jsonl"
    manifest_file = manifest_path.open("a", encoding="utf-8")
    last_save = 0.0
    count_present = 0
    count_absent = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                LOG.info("source closed; stopping capture loop")
                break
            cv2.imshow("MyZubster dataset capture", frame)
            key = cv2.waitKey(int(args.interval * 1000)) & 0xFF
            now = time.time()
            if key in (ord("q"), 27):
                LOG.info("quit requested")
                break
            label = None
            if key == ord(" "):
                label = "present"
            elif key in (ord("x"), ord("X")):
                label = "absent"
            elif now - last_save >= args.interval * 4:  # passive auto-save
                label = "auto"
            if label is None:
                continue
            ts = time.strftime("%Y%m%d_%H%M%S")
            idx = count_present + count_absent
            stem = f"frame_{ts}_{idx:04d}_{label}"
            img_path = img_dir / f"{stem}.jpg"
            cv2.imwrite(str(img_path), frame)
            record = {
                "image": str(img_path.relative_to(out)),
                "label": label,
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "source": str(args.source),
            }
            manifest_file.write(json.dumps(record) + "\n")
            manifest_file.flush()
            if label == "present":
                count_present += 1
            elif label == "absent":
                count_absent += 1
            last_save = now
            LOG.info("saved %s (%s)", img_path.name, label)
    finally:
        manifest_file.close()
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    LOG.info("captured present=%d absent=%d", count_present, count_absent)
    LOG.info("manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())