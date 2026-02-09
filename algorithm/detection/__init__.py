"""
Detection Stage

Runs YOLO person detection on synchronized drone frames.

Components:
- models: Data structures (BoundingBox, Detection, DetectionSet)
- yolo_detector: Single-frame YOLO wrapper
- batch_detector: Process all frames in a SynchronizedFrameSet

Usage:
    from detection import BatchDetector, Detection, DetectionSet
    from ingestion import SynchronizedFrameSet

    # Create detector
    detector = BatchDetector(weights_path="weights/yolov8n.pt")

    # Process synchronized frames
    sync_set = synchronizer_queue.get()
    all_detections = detector.process(sync_set)

    # Access detections
    for drone_id, detection_set in all_detections.items():
        for detection in detection_set.detections:
            print(f"Person at {detection.center}, conf={detection.confidence}")
"""

# Data models
from detection.models import (
    BoundingBox,
    Detection,
    DetectionSet,
    compute_iou_matrix,
)

# Detectors
from detection.yolo_detector import YOLODetector
from detection.batch_detector import BatchDetector


__all__ = [
    # Models
    "BoundingBox",
    "Detection",
    "DetectionSet",
    "compute_iou_matrix",
    # Detectors
    "YOLODetector",
    "BatchDetector",
]
