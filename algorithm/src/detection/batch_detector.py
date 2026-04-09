"""
Batch Detector

Processes all frames in a SynchronizedFrameSet using YOLO.

This module:
- Takes a SynchronizedFrameSet (multiple drone frames)
- Runs YOLO detection on each frame
- Returns map of {drone_id: DetectionSet}
"""

import logging
from typing import Dict

from src.detection.models import DetectionSet
from src.detection.yolo_detector import YOLODetector
from src.ingestion.models import SynchronizedFrameSet


logger = logging.getLogger(__name__)


class BatchDetector:
    def __init__(self, batch_mode: bool = False):
        self.detector = YOLODetector()
        self.batch_mode = batch_mode

        if batch_mode:
            self.detector.warmup_batch()

        logger.info(
            "Batch detector initialized (device=%s, mode=%s)",
            self.detector.device,
            "BATCH" if batch_mode else "sequential",
        )

    def process(self, sync_set: SynchronizedFrameSet) -> Dict[int, DetectionSet]:
        """
        Process all frames in a synchronized set.

        Args:
            sync_set: SynchronizedFrameSet containing frames from multiple drones

        Returns:
            Dictionary mapping {drone_id: DetectionSet}
        """
        results = {}

        frame_num = sync_set.frame_num
        num_drones = sync_set.num_drones_present

        logger.debug(
            "Processing synchronized set %d (%d drones)", frame_num, num_drones
        )

        for drone_id in sorted(sync_set.frames.keys()):
            drone_frame = sync_set.frames[drone_id]

            detection_set = self.detector.detect(
                frame=drone_frame.frame, drone_id=drone_id, frame_num=frame_num
            )

            results[drone_id] = detection_set

        total_detections = sum(ds.num_detections for ds in results.values())
        logger.info(
            "Frame %d: %d total detections across %d drones",
            frame_num,
            total_detections,
            num_drones,
        )

        return results

    def process_batch(self, sync_set: SynchronizedFrameSet) -> Dict[int, DetectionSet]:
        """Process all frames in a synchronized set using one batched predict call.

        Requires BatchDetector(batch_mode=True) for optimal OpenVINO performance.

        Args:
            sync_set: SynchronizedFrameSet containing frames from multiple drones.

        Returns:
            Dictionary mapping {drone_id: DetectionSet}, same shape as process().
        """
        drone_ids = sorted(sync_set.frames.keys())
        frames    = [sync_set.frames[d].frame for d in drone_ids]

        results = self.detector.detect_batch(frames, drone_ids, sync_set.frame_num)

        total_detections = sum(ds.num_detections for ds in results.values())
        logger.info(
            "Frame %d (batch): %d total detections across %d drones",
            sync_set.frame_num, total_detections, len(drone_ids),
        )

        return results


