"""
Detection Stage Data Models

Defines data structures for object detection outputs.

Key Types:
- BoundingBox: 2D bounding box in image coordinates
- Detection: Single detected object with class, confidence, bbox
- DetectionSet: All detections for one drone frame
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        assert self.x2 >= self.x1, f"Invalid bbox: x2={self.x2} < x1={self.x1}"
        assert self.y2 >= self.y1, f"Invalid bbox: y2={self.y2} < y1={self.y1}"

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        cx = (self.x1 + self.x2) / 2.0
        cy = (self.y1 + self.y2) / 2.0
        return (cx, cy)

    @property
    def area(self) -> float:
        return self.width * self.height

    @classmethod
    def from_xyxy(cls, coords: np.ndarray) -> "BoundingBox":
        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])

    def clip(self, width: int, height: int) -> "BoundingBox":
        """
        Clip box to image boundaries.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            New BoundingBox clipped to [0, width) x [0, height)
        """
        x1_clipped = max(0, min(self.x1, width))
        y1_clipped = max(0, min(self.y1, height))
        x2_clipped = max(0, min(self.x2, width))
        y2_clipped = max(0, min(self.y2, height))

        return BoundingBox(x1=x1_clipped, y1=y1_clipped, x2=x2_clipped, y2=y2_clipped)


@dataclass
class Detection:
    """
    Single detected object in an image.

    This represents one detection from YOLO:

    Attributes:
        bbox: 2D bounding box in image coordinates
        class_id: Object class (0 for person)
        confidence: Detection confidence in [0, 1]
        drone_id: Which drone this detection came from
        frame_num: Frame sequence number
        local_id: Detection ID within this frame (assigned by detector)
    """

    bbox: BoundingBox
    class_id: int
    confidence: float
    drone_id: int
    frame_num: int
    local_id: Optional[int] = None
    features: Optional[np.ndarray] = (
        None  # WCH feature vector, set by feature extraction stage
    )

    def __post_init__(self):
        assert (
            0.0 <= self.confidence <= 1.0
        ), f"Confidence must be in [0, 1], got {self.confidence}"
        assert self.class_id >= 0, f"class_id must be >= 0, got {self.class_id}"
        assert self.drone_id >= 1, f"drone_id must be >= 1, got {self.drone_id}"

    def __repr__(self) -> str:
        """String representation for debugging."""
        cx, cy = self.bbox.center
        return (
            f"Detection(drone={self.drone_id}, frame={self.frame_num}, "
            f"class={self.class_id}, conf={self.confidence:.2f}, "
            f"center=({cx:.0f}, {cy:.0f}), size={self.bbox.width:.0f}x{self.bbox.height:.0f})"
        )


@dataclass
class DetectionSet:
    """
    All detections for one drone frame.

    This is the output of running YOLO on a single DroneFrame.
    Contains all detected persons in that frame.

    Attributes:
        drone_id: Which drone these detections came from
        frame_num: Frame sequence number
        detections: List of Detection objects
        inference_time: How long YOLO took (seconds)
    """

    drone_id: int
    frame_num: int
    detections: list[Detection]
    inference_time: float = 0.0

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    @property
    def is_empty(self) -> bool:
        return len(self.detections) == 0

    def get_boxes(self) -> list[BoundingBox]:
        return [det.bbox for det in self.detections]

    def get_confidences(self) -> np.ndarray:
        return np.array([det.confidence for det in self.detections], dtype=np.float32)

    def get_centers(self) -> np.ndarray:
        if self.is_empty:
            return np.zeros((0, 2), dtype=np.float32)

        return np.array([det.bbox.center for det in self.detections], dtype=np.float32)

    def filter_by_confidence(self, min_confidence: float) -> "DetectionSet":
        filtered = [det for det in self.detections if det.confidence >= min_confidence]

        return DetectionSet(
            drone_id=self.drone_id,
            frame_num=self.frame_num,
            detections=filtered,
            inference_time=self.inference_time,
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"DetectionSet(drone={self.drone_id}, frame={self.frame_num}, "
            f"detections={self.num_detections}, time={self.inference_time:.3f}s)"
        )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Testing Detection Models")
    logger.info("=" * 60)

    logger.info("\nTest BoundingBox:")
    bbox1 = BoundingBox(x1=100, y1=100, x2=200, y2=300)
    logger.info(
        "bbox1: center=%s, size=%.0fx%.0f, area=%.0f",
        bbox1.center,
        bbox1.width,
        bbox1.height,
        bbox1.area,
    )

    bbox2 = BoundingBox(x1=150, y1=150, x2=250, y2=250)
    logger.info("bbox2: center=%s", bbox2.center)

    logger.info("\nTest Detection:")
    det = Detection(
        bbox=bbox1, class_id=0, confidence=0.95, drone_id=1, frame_num=42, local_id=0
    )
    logger.info(det)

    logger.info("\nTest DetectionSet:")
    det_set = DetectionSet(
        drone_id=1, frame_num=42, detections=[det], inference_time=0.015
    )
    logger.info(det_set)
    logger.info("Centers: %s", det_set.get_centers())

    logger.info("\n" + "=" * 60)
    logger.info("All tests passed!")
