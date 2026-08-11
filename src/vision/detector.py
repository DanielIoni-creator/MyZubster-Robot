"""Glass-under-dispenser detection for the MyZubster robot.

Combines a fine-tuned YOLOv8n detector with an OpenCV region-of-interest
(ROI) check. The detector finds glass instances in a frame; the ROI check
confirms that at least one detected glass falls inside the user-defined
"target zone" (the area under the dispenser spout). The robot should only
trigger a pour when ``is_glass_under_target()`` returns True.

Design goals:
- Pure-stdlib + numpy + OpenCV at runtime. ``ultralytics`` is loaded lazily so
  the module can be imported on systems that only need the geometry helpers.
- Hardware agnostic: works with any source that yields BGR ``numpy.ndarray``
  frames (PiCamera2, OpenCV ``VideoCapture``, recorded video, or test images).
- Deterministic ROI math: ROI rectangles can be expressed in normalized
  ``[0, 1]`` coordinates (preferred for fixed-mount cameras) or absolute
  pixel coordinates.

The module exposes:

- :class:`ROISpec` — ROI rectangle (normalized or absolute).
- :class:`Detection` — single detection result.
- :class:`GlassDetector` — load model + run inference + ROI check.
- :func:`draw_detection` — OpenCV visualization helper for the demo.

Example::

    import cv2
    from vision import GlassDetector, ROISpec

    detector = GlassDetector(
        model_path="models/yolov8n_glass.pt",
        roi=ROISpec.normalized(0.35, 0.40, 0.65, 0.85),
        confidence=0.45,
    )
    frame = cv2.imread("frame.jpg")
    result = detector.is_glass_under_target(frame)
    if result.ok:
        print("Glass under spout, confidence", result.confidence)
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

LOG = logging.getLogger(__name__)

try:  # OpenCV is only required at runtime; keep the module importable for tests.
    import cv2  # type: ignore

    _HAVE_CV2 = True
except Exception:  # pragma: no cover - exercised only on slim envs
    cv2 = None  # type: ignore
    _HAVE_CV2 = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ROISpec:
    """Region of interest expressed in either normalized [0,1] or pixel units.

    Attributes:
        x: left edge.
        y: top edge.
        w: width.
        h: height.
        normalized: True when the values are fractions of frame width/height.
    """

    x: float
    y: float
    w: float
    h: float
    normalized: bool = True

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"ROI width/height must be > 0 (got {self.w}x{self.h})")
        if self.normalized:
            if not (0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0):
                raise ValueError(f"normalized ROI origin out of range: ({self.x},{self.y})")
            if not (0.0 <= self.x + self.w <= 1.0 and 0.0 <= self.y + self.h <= 1.0):
                raise ValueError("normalized ROI rectangle exceeds [0,1] bounds")

    @classmethod
    def from_normalized(cls, x: float, y: float, w: float, h: float) -> "ROISpec":
        """Build a normalized ROI in [0,1] frame-relative coordinates."""
        return cls(x=x, y=y, w=w, h=h, normalized=True)

    @classmethod
    def from_pixels(cls, x: int, y: int, w: int, h: int) -> "ROISpec":
        """Build a ROI in absolute pixel coordinates."""
        return cls(x=float(x), y=float(y), w=float(w), h=float(h), normalized=False)

    def to_pixels(self, frame_w: int, frame_h: int) -> "ROISpec":
        """Return an equivalent pixel-space ROI for the given frame size."""
        if not self.normalized:
            return self
        return ROISpec.from_pixels(
            int(round(self.x * frame_w)),
            int(round(self.y * frame_h)),
            int(round(self.w * frame_w)),
            int(round(self.h * frame_h)),
        )


@dataclass(frozen=True)
class Detection:
    """Single detected object.

    Attributes:
        bbox: (x1, y1, x2, y2) pixel-space bounding box.
        confidence: detector confidence in [0,1].
        class_id: YOLO class id (0 for our fine-tuned ``glass`` class).
        class_name: human-readable class name.
    """

    bbox: tuple
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> tuple:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        x1, _, x2, _ = self.bbox
        return x2 - x1

    @property
    def height(self) -> float:
        _, y1, _, y2 = self.bbox
        return y2 - y1


@dataclass(frozen=True)
class DetectionResult:
    """Outcome of :meth:`GlassDetector.is_glass_under_target`."""

    ok: bool
    confidence: float
    detections: tuple
    roi: ROISpec
    frame_shape: tuple

    def best_detection(self) -> Optional[Detection]:
        """Return the highest-confidence detection, or ``None``."""
        if not self.detections:
            return None
        return max(self.detections, key=lambda d: d.confidence)

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict (used by tests + demo logger)."""
        return {
            "ok": self.ok,
            "confidence": self.confidence,
            "frame_shape": list(self.frame_shape),
            "roi": dataclasses.asdict(self.roi),
            "detections": [
                {
                    "bbox": list(d.bbox),
                    "confidence": d.confidence,
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                }
                for d in self.detections
            ],
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class GlassDetector:
    """YOLOv8n-based glass detector with ROI gating."""

    DEFAULT_CLASS_NAME = "glass"

    def __init__(
        self,
        model_path: Optional[str] = None,
        roi: Optional[ROISpec] = None,
        confidence: float = 0.45,
        iou: float = 0.45,
        device: str = "",
        class_names: Optional[Sequence[str]] = None,
        imgsz: int = 640,
    ) -> None:
        """Load a YOLO model and configure gating parameters.

        Args:
            model_path: Path to a ``.pt`` weights file. When ``None``, the
                detector is constructed in *geometry-only* mode: ``detect``
                returns no boxes, but ROI math and geometry tests still work.
            roi: Target zone rectangle. Defaults to a centered rectangle
                covering 30% of frame width and 45% of frame height (the
                area directly under the spout for a fixed-mount camera).
            confidence: Confidence threshold forwarded to YOLO.
            iou: IoU threshold for non-max suppression.
            device: Inference device forwarded to YOLO (e.g. ``"cpu"``,
                ``"cuda:0"``, or empty string for auto-detect).
            class_names: Class id → human-readable name. Index 0 must be the
                glass class. Defaults to ``("glass",)``.
            imgsz: Inference image size passed to YOLO.
        """
        if not 0.0 < confidence <= 1.0:
            raise ValueError(f"confidence must be in (0,1], got {confidence}")
        if not 0.0 < iou <= 1.0:
            raise ValueError(f"iou must be in (0,1], got {iou}")
        self.model_path = model_path
        self.roi = roi or ROISpec.from_normalized(0.35, 0.40, 0.30, 0.45)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.device = device
        self.imgsz = int(imgsz)
        self.class_names: tuple = tuple(class_names) if class_names else (self.DEFAULT_CLASS_NAME,)
        self._model = None
        if model_path is not None:
            self._load_model(model_path)

    # ------------------------------------------------------------------ model

    def _load_model(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO weights not found: {model_path}")
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - exercised only on slim envs
            raise RuntimeError(
                "ultralytics is required to load YOLO weights; "
                "install with `pip install ultralytics`"
            ) from exc
        LOG.info("loading YOLO model from %s", model_path)
        self._model = YOLO(model_path)

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------ inference

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR frame and return detections."""
        if frame is None:
            raise ValueError("frame is None")
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise ValueError(f"frame must be HxWx3 or HxWx4, got shape {frame.shape}")
        if self._model is None:
            LOG.debug("model not loaded; detect() returns empty list")
            return []
        model = self._model
        bgr = frame[:, :, :3] if frame.shape[2] == 4 else frame
        # ultralytics accepts both file paths and numpy arrays.
        results = model.predict(
            source=bgr,
            conf=self.confidence,
            iou=self.iou,
            device=self.device or None,
            imgsz=self.imgsz,
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections
        result = results[0]
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections
        for box in boxes:
            xyxy = box.xyxy
            if hasattr(xyxy, "cpu"):
                xyxy = xyxy.cpu().numpy()
            else:
                xyxy = np.asarray(xyxy)
            xyxy = xyxy.reshape(-1)
            if xyxy.size < 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in xyxy[:4])
            conf_attr = getattr(box, "conf", None)
            cls_attr = getattr(box, "cls", None)
            if conf_attr is None or cls_attr is None:
                continue
            if hasattr(conf_attr, "cpu"):
                conf_val = float(conf_attr.cpu().numpy().reshape(-1)[0])
            elif hasattr(conf_attr, "item"):
                conf_val = float(conf_attr.item())
            elif hasattr(conf_attr, "__float__"):
                conf_val = float(conf_attr)
            else:
                conf_val = float(np.asarray(conf_attr).reshape(-1)[0])
            if hasattr(cls_attr, "cpu"):
                cls_id_val = int(cls_attr.cpu().numpy().reshape(-1)[0])
            elif hasattr(cls_attr, "item"):
                cls_id_val = int(cls_attr.item())
            elif hasattr(cls_attr, "__int__"):
                cls_id_val = int(cls_attr)
            else:
                cls_id_val = int(np.asarray(cls_attr).reshape(-1)[0])
            if 0 <= cls_id_val < len(self.class_names):
                cls_name = self.class_names[cls_id_val]
            else:
                cls_name = names.get(cls_id_val, str(cls_id_val))
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf_val,
                    class_id=cls_id_val,
                    class_name=cls_name,
                )
            )
        return detections

    # ---------------------------------------------------------------- gating

    def is_glass_under_target(
        self,
        frame: np.ndarray,
        min_iou: float = 0.10,
    ) -> DetectionResult:
        """Decide whether the robot may dispense.

        A detection counts as "under target" when its bounding box overlaps
        the ROI by at least ``min_iou`` (default 10%).

        Args:
            frame: BGR image as a ``numpy.ndarray``.
            min_iou: Minimum IoU between detection box and ROI required to
                accept the detection.

        Returns:
            A :class:`DetectionResult` carrying all detections, the ROI, and
            a boolean ``ok`` field.
        """
        if frame is None:
            raise ValueError("frame is None")
        h, w = frame.shape[:2]
        roi_px = self.roi.to_pixels(w, h)
        detections = self.detect(frame)
        best_iou = 0.0
        best_conf = 0.0
        for det in detections:
            iou = _bbox_iou(det.bbox, _rect_to_bbox(roi_px))
            if iou > best_iou:
                best_iou = iou
                best_conf = det.confidence
        ok = best_iou >= min_iou and best_conf >= self.confidence
        return DetectionResult(
            ok=ok,
            confidence=best_conf,
            detections=tuple(detections),
            roi=roi_px,
            frame_shape=(h, w),
        )

    # --------------------------------------------------------- calibration

    def calibrate_roi_from_click(
        self,
        frame_shape: tuple,
        click_points: Iterable[tuple],
        padding: float = 0.10,
    ) -> ROISpec:
        """Build a normalized ROI from one or two click points on a frame.

        Single click → ROI is a square of side ``2 * padding`` centered on the
        click (in normalized coordinates). Two clicks → ROI spans the two
        points with ``padding`` margin on every side.

        Args:
            frame_shape: ``(h, w)`` of the calibration frame.
            click_points: Iterable of ``(x, y)`` in absolute pixel coordinates.
            padding: Normalized padding around the ROI (default 0.10).

        Returns:
            A new :class:`ROISpec` (normalized).
        """
        h, w = frame_shape[:2]
        pts = list(click_points)
        if not pts:
            raise ValueError("at least one click point is required")
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        # Pad in normalized space.
        nx0 = max(0.0, (x0 / w) - padding)
        ny0 = max(0.0, (y0 / h) - padding)
        nx1 = min(1.0, (x1 / w) + padding)
        ny1 = min(1.0, (y1 / h) + padding)
        return ROISpec.from_normalized(nx0, ny0, max(1e-3, nx1 - nx0), max(1e-3, ny1 - ny0))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _rect_to_bbox(roi: ROISpec) -> tuple:
    x, y, w, h = roi.x, roi.y, roi.w, roi.h
    return (x, y, x + w, y + h)


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def draw_detection(
    frame: np.ndarray,
    result: DetectionResult,
    color_success: tuple = (0, 200, 0),
    color_box: tuple = (255, 200, 0),
    color_roi: tuple = (200, 200, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Overlay ROI + detection boxes on a copy of ``frame`` and return it.

    Requires OpenCV. Returns ``frame`` unchanged when cv2 is missing.
    """
    if not _HAVE_CV2 or frame is None:
        return frame
    cv2_mod = cv2  # captured for type checkers
    assert cv2_mod is not None
    canvas = frame.copy()
    # ROI
    rx, ry, rw, rh = (
        int(result.roi.x),
        int(result.roi.y),
        int(result.roi.w),
        int(result.roi.h),
    )
    cv2_mod.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), color_roi, thickness)
    cv2_mod.putText(
        canvas,
        "ROI",
        (rx + 4, max(15, ry - 6)),
        cv2_mod.FONT_HERSHEY_SIMPLEX,
        0.5,
        color_roi,
        1,
        cv2_mod.LINE_AA,
    )
    # Detections
    for det in result.detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        color = color_success if result.ok else color_box
        cv2_mod.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2_mod.putText(
            canvas,
            label,
            (x1, max(15, y1 - 6)),
            cv2_mod.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2_mod.LINE_AA,
        )
    status = "GLASS UNDER TARGET" if result.ok else "WAITING FOR GLASS"
    cv2_mod.putText(
        canvas,
        status,
        (10, canvas.shape[0] - 10),
        cv2_mod.FONT_HERSHEY_SIMPLEX,
        0.6,
        color_success if result.ok else (0, 0, 255),
        2,
        cv2_mod.LINE_AA,
    )
    return canvas


__all__ = [
    "ROISpec",
    "Detection",
    "DetectionResult",
    "GlassDetector",
    "draw_detection",
]