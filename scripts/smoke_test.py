"""Smoke test: import the package and run the geometry-only detector path.

This is the entry point used by the repository's CI / pre-merge checks. It
does NOT require ultralytics, OpenCV, or any model file. It is the fastest
end-to-end verification we can run on a slim container.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# scripts/ and src/ are siblings under the repo root in this layout.
CANDIDATES = [
    os.path.abspath(os.path.join(HERE, "..", "src")),
    os.path.abspath(os.path.join(HERE, "..", "..", "src")),
]
SRC = next((p for p in CANDIDATES if os.path.isdir(os.path.join(p, "vision"))), CANDIDATES[0])
ROOT = os.path.dirname(SRC)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np  # noqa: E402

from vision import GlassDetector, ROISpec  # noqa: E402


def main() -> int:
    # Geometry-only path (no YOLO loaded).
    detector = GlassDetector(
        model_path=None,
        roi=ROISpec.from_normalized(0.30, 0.30, 0.40, 0.40),
        confidence=0.45,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.is_glass_under_target(frame)
    payload = result.to_dict()
    assert payload["ok"] is False, "empty frame should not trigger"
    assert payload["detections"] == [], "no model loaded -> no detections"
    print("smoke ok:", json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())