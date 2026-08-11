"""Unit tests for the vision package.

These tests cover the geometry helpers, the ROI math, and the geometry-only
mode of :class:`vision.GlassDetector` (no YOLO weights needed). They run on
plain numpy + the standard library and do not require OpenCV or ultralytics.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

# Ensure src/ is on sys.path when tests are run directly.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np

import vision  # noqa: F401  ensure package import works
from vision.detector import (
    Detection,
    DetectionResult,
    GlassDetector,
    ROISpec,
    draw_detection,
)


class FakeYOLOBox:
    """Minimal stand-in for an ultralytics Boxes object."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.array(xyxy, dtype=float)
        self.conf = conf
        self.cls = cls


class FakeYOLOResult:
    def __init__(self, boxes, names=None):
        self.boxes = boxes
        self.names = names or {0: "glass"}


class FakeYOLOModel:
    def __init__(self, boxes, names=None):
        self._boxes = boxes
        self._names = names or {0: "glass"}
        self.calls = 0

    def predict(self, source, conf, iou, device, imgsz, verbose):
        self.calls += 1
        return [FakeYOLOResult(self._boxes, self._names)]


class ROISpecTests(unittest.TestCase):
    def test_normalized_basic(self):
        r = ROISpec.from_normalized(0.1, 0.2, 0.3, 0.4)
        self.assertTrue(r.normalized)
        self.assertEqual((0.1, 0.2, 0.3, 0.4), (r.x, r.y, r.w, r.h))

    def test_pixel_to_pixel_passthrough(self):
        r = ROISpec.from_pixels(10, 20, 30, 40)
        out = r.to_pixels(640, 480)
        self.assertEqual((10, 20, 30, 40), (out.x, out.y, out.w, out.h))
        self.assertFalse(out.normalized)

    def test_normalized_to_pixels(self):
        r = ROISpec.from_normalized(0.0, 0.5, 1.0, 0.5)
        out = r.to_pixels(640, 480)
        self.assertEqual((0, 240, 640, 240), (out.x, out.y, out.w, out.h))

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            ROISpec.from_normalized(0.0, 0.0, 0.0, 0.1)
        with self.assertRaises(ValueError):
            ROISpec.from_normalized(-0.1, 0.0, 0.2, 0.2)
        with self.assertRaises(ValueError):
            ROISpec.from_normalized(0.9, 0.5, 0.5, 0.2)


class BboxIoUTests(unittest.TestCase):
    def test_identical(self):
        a = (0.0, 0.0, 10.0, 10.0)
        self.assertAlmostEqual(1.0, _bbox_iou(a, a), places=6)

    def test_disjoint(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (20.0, 20.0, 30.0, 30.0)
        self.assertAlmostEqual(0.0, _bbox_iou(a, b), places=6)

    def test_partial(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 5.0, 15.0, 15.0)
        # Intersection = 5x5 = 25. Union = 100 + 100 - 25 = 175. IoU = 25/175.
        self.assertAlmostEqual(25.0 / 175.0, _bbox_iou(a, b), places=6)


def _bbox_iou(a, b):
    from vision.detector import _bbox_iou as impl
    return impl(a, b)


class GlassDetectorGeometryTests(unittest.TestCase):
    def _frame(self, h=480, w=640):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_no_model_returns_no_detections(self):
        d = GlassDetector(model_path=None, roi=ROISpec.from_normalized(0.1, 0.1, 0.8, 0.8))
        result = d.is_glass_under_target(self._frame())
        self.assertFalse(result.ok)
        self.assertEqual([], list(result.detections))
        self.assertEqual(0.0, result.confidence)
        self.assertEqual((480, 640), result.frame_shape)

    def test_detection_inside_roi(self):
        d = GlassDetector(model_path=None, roi=ROISpec.from_normalized(0.3, 0.3, 0.4, 0.4))
        d._model = FakeYOLOModel(
            [FakeYOLOBox(xyxy=[260, 260, 420, 420], conf=np.array([0.92]), cls=np.array([0]))]
        )
        result = d.is_glass_under_target(self._frame(480, 640))
        self.assertTrue(result.ok)
        self.assertAlmostEqual(0.92, result.confidence, places=4)
        self.assertEqual(1, len(result.detections))
        self.assertEqual("glass", result.detections[0].class_name)

    def test_detection_outside_roi(self):
        d = GlassDetector(model_path=None, roi=ROISpec.from_normalized(0.4, 0.4, 0.2, 0.2))
        d._model = FakeYOLOModel(
            [FakeYOLOBox(xyxy=[0, 0, 50, 50], conf=np.array([0.88]), cls=np.array([0]))]
        )
        result = d.is_glass_under_target(self._frame(480, 640))
        self.assertFalse(result.ok)
        # Still records the detection.
        self.assertEqual(1, len(result.detections))

    def test_multiple_detections_best_confidence_wins(self):
        d = GlassDetector(model_path=None, roi=ROISpec.from_normalized(0.3, 0.3, 0.4, 0.4))
        d._model = FakeYOLOModel(
            [
                # Outside ROI but high confidence - should NOT win for ROI gating.
                FakeYOLOBox(xyxy=[0, 0, 100, 100], conf=np.array([0.99]), cls=np.array([0])),
                # Inside ROI with mid confidence - this drives ok=True.
                FakeYOLOBox(xyxy=[240, 180, 440, 380], conf=np.array([0.81]), cls=np.array([0])),
            ]
        )
        result = d.is_glass_under_target(self._frame(480, 640))
        # ROI gating accepted the in-ROI detection.
        self.assertTrue(result.ok)
        # The reported confidence is from the highest-IoU (in-ROI) detection.
        self.assertAlmostEqual(0.81, result.confidence, places=4)
        # All detections are still reported.
        self.assertEqual(2, len(result.detections))
        # best_detection() returns the overall highest-confidence detection.
        best = result.best_detection()
        self.assertIsNotNone(best)
        self.assertAlmostEqual(0.99, best.confidence, places=4)

    def test_confidence_threshold_respected(self):
        d = GlassDetector(
            model_path=None,
            roi=ROISpec.from_normalized(0.3, 0.3, 0.4, 0.4),
            confidence=0.8,
        )
        d._model = FakeYOLOModel(
            [FakeYOLOBox(xyxy=[300, 300, 400, 400], conf=np.array([0.55]), cls=np.array([0]))]
        )
        result = d.is_glass_under_target(self._frame(480, 640))
        self.assertFalse(result.ok, "low-confidence detection must not pass")
        # Detection is still captured.
        self.assertEqual(1, len(result.detections))

    def test_invalid_confidence_rejected(self):
        with self.assertRaises(ValueError):
            GlassDetector(model_path=None, confidence=0.0)
        with self.assertRaises(ValueError):
            GlassDetector(model_path=None, confidence=1.5)

    def test_invalid_frame_rejected(self):
        d = GlassDetector(model_path=None)
        with self.assertRaises(ValueError):
            d.is_glass_under_target(None)
        with self.assertRaises(ValueError):
            d.is_glass_under_target(np.zeros((480, 480), dtype=np.uint8))  # 2D

    def test_calibration_one_click(self):
        d = GlassDetector(model_path=None)
        roi = d.calibrate_roi_from_click((480, 640), [(320, 240)], padding=0.05)
        self.assertTrue(roi.normalized)
        self.assertAlmostEqual(0.45, roi.x, places=4)
        self.assertAlmostEqual(0.45, roi.y, places=4)
        self.assertAlmostEqual(0.10, roi.w, places=4)
        self.assertAlmostEqual(0.10, roi.h, places=4)

    def test_calibration_two_clicks(self):
        d = GlassDetector(model_path=None)
        roi = d.calibrate_roi_from_click((480, 640), [(200, 200), (440, 400)], padding=0.0)
        # x range = 200..440 of 640, so 0.3125..0.6875
        self.assertAlmostEqual(0.3125, roi.x, places=4)
        self.assertAlmostEqual(0.4167, roi.y, places=3)
        self.assertAlmostEqual(0.375, roi.w, places=4)


class DetectionSerializationTests(unittest.TestCase):
    def test_to_dict_round_trip(self):
        det = Detection(bbox=(10, 20, 30, 40), confidence=0.5, class_id=0, class_name="glass")
        roi = ROISpec.from_pixels(0, 0, 100, 100)
        result = DetectionResult(
            ok=True, confidence=0.5, detections=(det,), roi=roi, frame_shape=(100, 100)
        )
        d = result.to_dict()
        self.assertTrue(d["ok"])
        self.assertEqual(1, len(d["detections"]))
        self.assertEqual([10, 20, 30, 40], d["detections"][0]["bbox"])
        self.assertEqual("glass", d["detections"][0]["class_name"])


if __name__ == "__main__":
    unittest.main()