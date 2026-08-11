"""Computer vision package for the MyZubster robot.

Provides glass-under-dispenser detection used by the robot to decide when a
drink can actually be dispensed. Built around a fine-tuned YOLOv8n detector
plus an OpenCV region-of-interest (ROI) check that confirms the glass is
positioned under the spout.

Public surface:
- :class:`GlassDetector`: YOLO + OpenCV ROI check.
- :class:`ROISpec`: rectangle specification (normalized or absolute pixels).
- :func:`draw_detection`: visualization helper used by the demo script.
"""

from .detector import GlassDetector, Detection, ROISpec, draw_detection

__all__ = ["GlassDetector", "Detection", "ROISpec", "draw_detection"]
__version__ = "0.1.0"