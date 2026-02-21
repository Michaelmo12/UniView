import logging
import time

import numpy as np
from ultralytics import YOLO

from algorithm.detection.models import BoundingBox, Detection, DetectionSet
from algorithm.config import settings


logger = logging.getLogger(__name__)


class YOLODetector:

    def __init__(self):
        # Load from config
        self.weights_path = settings.weights_path
        self.conf_threshold = settings.detection.conf_threshold
        self.iou_threshold = settings.detection.iou_threshold
        self.device = settings.detection.device
        self.person_class_id = settings.detection.person_class_id
        self.imgsz = settings.detection.imgsz

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")

        logger.info("Loading YOLO model from %s", self.weights_path)
        self.model = YOLO(str(self.weights_path))

        self.model.to(self.device)

        logger.info(
            "YOLO detector ready (device=%s, conf=%.2f, iou=%.2f, imgsz=%d)",
            self.device,
            self.conf_threshold,
            self.iou_threshold,
            self.imgsz,
        )

    def detect(self, frame: np.ndarray, drone_id: int, frame_num: int) -> DetectionSet:

        start_time = time.time()

        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[self.person_class_id],
            # to not see any ultralytics logging output, set verbose=False. We will log our own info. and not flood the console.
            verbose=False,
            imgsz=self.imgsz,
            device=self.device,
        )

        inference_time = time.time() - start_time

        detections = self._parse_results(results[0], drone_id, frame_num)

        logger.debug(
            "Detected %d persons in drone %d frame %d (%.3fs)",
            len(detections),
            drone_id,
            frame_num,
            inference_time,
        )

        #Return a DetectionSet containing all detections for this frame
        return DetectionSet(
            drone_id=drone_id,
            frame_num=frame_num,
            detections=detections,
            inference_time=inference_time,
        )

    def _parse_results(self, results, drone_id: int, frame_num: int) -> list[Detection]:
        """
        Takes raw YOLO results (ultralytics format)
        Converts each box into a Detection object
        Returns a plain list of detections
        """
        detections = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        boxes_xyxy = results.boxes.xyxy.cpu().numpy()  # (N, 4)
        confidences = results.boxes.conf.cpu().numpy()  # (N,)
        class_ids = results.boxes.cls.cpu().numpy()  # (N,)

        for local_id, (box, conf, cls) in enumerate(
            zip(boxes_xyxy, confidences, class_ids)
        ):
            bbox = BoundingBox(
                x1=float(box[0]), y1=float(box[1]), x2=float(box[2]), y2=float(box[3])
            )

            detection = Detection(
                bbox=bbox,
                class_id=int(cls),
                confidence=float(conf),
                drone_id=drone_id,
                frame_num=frame_num,
                local_id=local_id,
            )

            detections.append(detection)

        return detections


if __name__ == "__main__":

    import cv2
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    logger.info("Testing YOLO Detector")
    logger.info("=" * 60)

    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        # Default to test image in the same directory as this script
        image_path = Path(__file__).parent / "people.jpg"

    # Check if image exists
    if not image_path.exists():
        logger.error("Image not found: %s", image_path)
        logger.info("\nUsage:")
        logger.info("  python -m detection.yolo_detector <image_path>")
        logger.info("\nOr edit line 140 to set default image path")
        sys.exit(1)

    # Create detector
    detector = YOLODetector()

    logger.info("Config:")
    logger.info("  Weights: %s", detector.weights_path)
    logger.info("  Device: %s", detector.device)
    logger.info("  Confidence: %.2f", detector.conf_threshold)
    logger.info("")

    # Load image
    logger.info("Loading image: %s", image_path)
    test_frame = cv2.imread(str(image_path))

    if test_frame is None:
        logger.error("Failed to load image!")
        sys.exit(1)

    logger.info("Image size: %dx%d", test_frame.shape[1], test_frame.shape[0])
    logger.info("")

    # Run detection
    logger.info("Running detection...")
    detection_set = detector.detect(test_frame, drone_id=1, frame_num=0)

    # Show results
    logger.info("\nResults:")
    logger.info("  Detections: %d", detection_set.num_detections)
    logger.info("  Inference time: %.3fs", detection_set.inference_time)
    logger.info("")

    if detection_set.num_detections > 0:
        logger.info("Detected persons:")
        for i, det in enumerate(detection_set.detections):
            cx, cy = det.bbox.center
            logger.info(
                "  [%d] conf=%.2f, center=(%.0f, %.0f), size=%.0fx%.0f",
                i,
                det.confidence,
                cx,
                cy,
                det.bbox.width,
                det.bbox.height,
            )

        # Draw bounding boxes on image
        output_frame = test_frame.copy()
        for det in detection_set.detections:
            x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
            x2, y2 = int(det.bbox.x2), int(det.bbox.y2)

            # Draw box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 10)

            # Draw confidence
            label = f"{det.confidence:.2f}"
            cv2.putText(
                output_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Save output image
        output_path = (
            image_path.parent / f"{image_path.stem}_detections{image_path.suffix}"
        )
        cv2.imwrite(str(output_path), output_frame)
        logger.info("\nSaved output to: %s", output_path)

    else:
        logger.info("No persons detected")

    logger.info("\n" + "=" * 60)
    logger.info("Test complete!")
