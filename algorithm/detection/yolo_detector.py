import logging
import time

import numpy as np
from ultralytics import YOLO

from detection.models import BoundingBox, Detection, DetectionSet
from config import settings


logger = logging.getLogger(__name__)


class YOLODetector:

    def __init__(self):
        # Load from config
        self.weights_path = settings.weights_path
        # Confidence threshold
        self.conf_threshold = settings.detection.conf_threshold
        # IoU threshold for NMS 
        self.iou_threshold = settings.detection.iou_threshold
        # Device: 'cuda' or 'cpu'
        self.device = settings.detection.device
        # Class ID for 'person'
        self.person_class_id = settings.detection.person_class_id
        # image size for YOLO
        self.imgsz = settings.detection.imgsz

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")

        logger.info("Loading YOLO model from %s", self.weights_path)
        self.model = YOLO(str(self.weights_path))

        self.model.to(self.device)

        logger.info("YOLO detector ready (device=%s, conf=%.2f, iou=%.2f, imgsz=%d)",
                   self.device, self.conf_threshold, self.iou_threshold, self.imgsz)

        self.total_inferences = 0
        self.total_detections = 0
        self.total_time = 0.0

    def detect(self, frame: np.ndarray, drone_id: int, frame_num: int) -> DetectionSet:

        start_time = time.time()

        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[self.person_class_id],
            verbose=False,
            imgsz=self.imgsz,
            device=self.device
        )

        inference_time = time.time() - start_time

        detections = self._parse_results(results[0], drone_id, frame_num)

        self.total_inferences += 1
        self.total_detections += len(detections)
        self.total_time += inference_time

        logger.debug("Detected %d persons in drone %d frame %d (%.3fs)",
                    len(detections), drone_id, frame_num, inference_time)

        return DetectionSet(
            drone_id=drone_id,
            frame_num=frame_num,
            detections=detections,
            inference_time=inference_time
        )

    def _parse_results(self, results, drone_id: int, frame_num: int) -> list[Detection]:

        detections = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        boxes_xyxy = results.boxes.xyxy.cpu().numpy()  # (N, 4)
        confidences = results.boxes.conf.cpu().numpy()  # (N,)
        class_ids = results.boxes.cls.cpu().numpy()     # (N,)

        for local_id, (box, conf, cls) in enumerate(zip(boxes_xyxy, confidences, class_ids)):
            bbox = BoundingBox(
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3])
            )

            detection = Detection(
                bbox=bbox,
                class_id=int(cls),
                confidence=float(conf),
                drone_id=drone_id,
                frame_num=frame_num,
                local_id=local_id
            )

            detections.append(detection)

        return detections

    def get_stats(self) -> dict:
        avg_time = self.total_time / max(1, self.total_inferences)
        avg_detections = self.total_detections / max(1, self.total_inferences)

        return {
            "total_inferences": self.total_inferences,
            "total_detections": self.total_detections,
            "total_time": self.total_time,
            "avg_time_per_frame": avg_time,
            "avg_detections_per_frame": avg_detections,
            "fps": 1.0 / avg_time if avg_time > 0 else 0.0
        }

    def reset_stats(self) -> None:
        self.total_inferences = 0
        self.total_detections = 0
        self.total_time = 0.0

if __name__ == "__main__":

    import cv2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    )

    logger.info("Testing YOLO Detector")
    logger.info("=" * 60)

    # Create detector (using config)
    detector = YOLODetector()

    logger.info("Using config:")
    logger.info("  Weights: %s", detector.weights_path)
    logger.info("  Device: %s", detector.device)
    logger.info("  Confidence: %.2f", detector.conf_threshold)

    # Create a test image with random noise
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    logger.info("\nRunning detection on random test image...")
    detection_set = detector.detect(test_frame, drone_id=1, frame_num=0)

    logger.info("\nResults:")
    logger.info(detection_set)

    if detection_set.num_detections > 0:
        logger.info("\nDetections:")
        for det in detection_set.detections:
            logger.info("  %s", det)

    # Show stats
    stats = detector.get_stats()
    logger.info("\nDetector Statistics:")
    for key, value in stats.items():
        logger.info("  %s: %.4f", key, value)

    logger.info("\n" + "=" * 60)
    logger.info("Test complete!")
