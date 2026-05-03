import logging
import time

import numpy as np
from ultralytics import YOLO

from src.detection.models import BoundingBox, Detection, DetectionSet
from src.config import settings


logger = logging.getLogger(__name__)


class YOLODetector:

    def __init__(self):
        # Load from config
        self.weights_path = settings.weights_path
        self.conf_threshold = settings.detection.conf_threshold
        # used for NMS
        self.iou_threshold = settings.detection.iou_threshold
        self.device = settings.detection.device
        self.person_class_id = settings.detection.person_class_id
        # img size before running
        self.imgsz = settings.detection.imgsz

        # Weights check + model load
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")

        logger.info("Loading YOLO model from %s", self.weights_path)
        self.model = YOLO(str(self.weights_path), task="detect")

        logger.info(
            "YOLO detector ready (device=%s, conf=%.2f, iou=%.2f, imgsz=%d)",
            self.device,
            self.conf_threshold,
            self.iou_threshold,
            self.imgsz,
        )

        # Warmup: run one dummy inference so OpenVINO JIT-compiles the model now.
        # Without this the first real frame takes ~5s instead of ~100ms.
        logger.info("Warming up YOLO model (OpenVINO JIT)...")
        _dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.model.predict(
            _dummy,
            conf=self.conf_threshold,
            # if two boxes share more than 70% of their area, they're duplicates — kill the weaker one.
            iou=self.iou_threshold,
            classes=[self.person_class_id],
            verbose=False,
            imgsz=self.imgsz,
            device=self.device,
        )
        logger.info("YOLO warmup complete.")

    def detect(self, frame: np.ndarray, drone_id: int, frame_num: int) -> DetectionSet:

        start_time = time.time()

        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[self.person_class_id],
            # to not see any ultralytics logging output, and not flood the console.
            verbose=False,
            imgsz=self.imgsz,
            device=self.device,
        )

        inference_time = time.time() - start_time

        detections = self._parse_results(results[0], drone_id, frame_num)

        logger.info(
            "YOLO inference drone %d frame %d: %.3fs, %d person(s) detected",
            drone_id,
            frame_num,
            inference_time,
            len(detections),
        )

        # Return a DetectionSet containing all detections for this frame
        return DetectionSet(
            drone_id=drone_id,
            frame_num=frame_num,
            detections=detections,
            inference_time=inference_time,
        )

    def warmup_batch(self) -> None:
        """Warm up the model for batch inference using 4 dummy frames.

        Call this before using detect_batch() so OpenVINO selects
        CUMULATIVE_THROUGHPUT mode (requires batch > 1 on first call).
        """
        logger.info("Warming up YOLO model for BATCH inference (4 dummy frames)...")
        _dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.model.predict(
            [_dummy] * 4,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[self.person_class_id],
            verbose=False,
            imgsz=self.imgsz,
            device=self.device,
        )
        logger.info("Batch warmup complete.")

    def detect_batch(
        self,
        frames: list[np.ndarray],
        drone_ids: list[int],
        frame_num: int,
    ) -> dict[int, "DetectionSet"]:
        """Run YOLO on all frames in one batched predict call.

        Args:
            frames:    List of BGR arrays, one per drone (same order as drone_ids).
            drone_ids: Drone IDs corresponding to each frame.
            frame_num: Shared frame number for this synchronized set.

        Returns:
            Dict mapping drone_id -> DetectionSet.
        """
        start_time = time.time()

        # EXAMPLE: batch_results=[r3, r4, r6, r7]
        batch_results = self.model.predict(
            frames,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[self.person_class_id],
            verbose=False,
            imgsz=self.imgsz,
            device=self.device,
        )

        # LOGGING
        # total time for all drones combined
        total_time = time.time() - start_time
        # avg
        per_drone_time = total_time / len(drone_ids)

        # output dict: drone_id -> DetectionSet
        output: dict[int, DetectionSet] = {}

        # pair each drone_id with its YOLO result (same order as input frames list)
        for drone_id, result in zip(drone_ids, batch_results):

            # convert raw YOLO result into list of Detection objects
            detections = self._parse_results(result, drone_id, frame_num)

            # wrap detections in a DetectionSet and store under drone_id
            output[drone_id] = DetectionSet(
                drone_id=drone_id,
                frame_num=frame_num,
                detections=detections,
                inference_time=per_drone_time,
            )

        # LOGGING
        logger.info(
            "YOLO batch inference frame %d: %.3fs total, %.3fs/drone avg (%d drones)",
            frame_num,
            total_time,
            per_drone_time,
            len(drone_ids),
        )

        return output

    def _parse_results(self, results, drone_id: int, frame_num: int) -> list[Detection]:
        """
        Takes raw YOLO results (ultralytics format)
        Converts each box into a Detection object
        Returns a plain list of detections
        """
        # empty list to collect Detection objects
        detections = []

        # if YOLO found nothing, return empty list early
        if results.boxes is None or len(results.boxes) == 0:
            return detections

        # all bounding boxes as numpy array, shape (N, 4) — each row is [x1, y1, x2, y2]
        boxes_xyxy = results.boxes.xyxy.cpu().numpy()  # (N, 4)
        # confidence score per detection, shape (N,)
        confidences = results.boxes.conf.cpu().numpy()  # (N,)
        # class ID per detection (always 0 = person), shape (N,)
        class_ids = results.boxes.cls.cpu().numpy()  # (N,)

        # iterate all three arrays together; enumerate gives local_id (0, 1, 2...) for each detection
        for local_id, (box, conf, cls) in enumerate(
            zip(boxes_xyxy, confidences, class_ids)
        ):
            # convert raw [x1,y1,x2,y2] row into a BoundingBox; float() converts numpy scalar to Python float
            bbox = BoundingBox(
                x1=float(box[0]), y1=float(box[1]), x2=float(box[2]), y2=float(box[3])
            )

            # build Detection object; local_id is the index within this frame used by fusion stage
            detection = Detection(
                bbox=bbox,
                class_id=int(cls),
                confidence=float(conf),
                drone_id=drone_id,
                frame_num=frame_num,
                local_id=local_id,
            )

            detections.append(detection)

        # return all detections for this frame
        return detections
