"""
Detection Stage

Runs YOLO person detection on synchronized drone frames.

Components:
- models: Data structures (BoundingBox, Detection, DetectionSet)
- yolo_detector: Single-frame YOLO wrapper
- batch_detector: Process all frames in a SynchronizedFrameSet

Usage:
    from detection import BatchDetector, Detection, DetectionSet

    batch_detector = BatchDetector()

    all_detections = batch_detector.process(sync_set)

    for drone_id, detection_set in all_detections.items():
        for detection in detection_set.detections:
            logger.info("Person at %s, conf=%.2f", detection.bbox.center, detection.confidence)
"""

# Data models
from algorithm.detection.models import (
    BoundingBox,
    Detection,
    DetectionSet,
)

# Detectors
from algorithm.detection.yolo_detector import YOLODetector
from algorithm.detection.batch_detector import BatchDetector


__all__ = [
    # Models
    "BoundingBox",
    "Detection",
    "DetectionSet",
    # Detectors
    "YOLODetector",
    "BatchDetector",
]
