"""
Batch Detector

Processes all frames in a SynchronizedFrameSet using YOLO.

This module:
- Takes a SynchronizedFrameSet (multiple drone frames)
- Runs YOLO detection on each frame
- Returns map of {drone_id: DetectionSet}
"""

import logging

from src.detection.models import DetectionSet
from src.detection.yolo_detector import YOLODetector
from src.ingestion.models import SynchronizedFrameSet


logger = logging.getLogger(__name__)


class BatchDetector:
    def __init__(self, batch_mode: bool = False):
        # create the underlying YOLO model (loads weights + single-frame warmup)
        self.detector = YOLODetector()
        self.batch_mode = batch_mode

        # extra warmup so OpenVINO switches to throughput mode for batch inference
        if batch_mode:
            self.detector.warmup_batch()

        logger.info(
            "Batch detector initialized (device=%s, mode=%s)",
            self.detector.device,
            "BATCH" if batch_mode else "sequential",
        )

    def process_batch(self, sync_set: SynchronizedFrameSet) -> dict[int, DetectionSet]:
        """Process all frames in a synchronized set using one batched predict call.

        Requires BatchDetector(batch_mode=True) for optimal OpenVINO performance.

        Args:
            sync_set: SynchronizedFrameSet containing frames from multiple drones.

        Returns:
            Dictionary mapping {drone_id: DetectionSet}.
        """
        # dict.keys() order is non-deterministic — sort so drone_ids and frames stay in sync
        drone_ids = sorted(sync_set.frames.keys())
        # build plain list of numpy frames in the same order as drone_ids
        frames = []
        for d in drone_ids:
            frames.append(sync_set.frames[d].frame)

        # one YOLO call for all drones — returns {drone_id: DetectionSet}
        results = self.detector.detect_batch(frames, drone_ids, sync_set.frame_num)

        # sum detections across all drones for the log line
        total_detections = 0
        for ds in results.values():
            total_detections += ds.num_detections
        logger.info(
            "Frame %d (batch): %d total detections across %d drones",
            sync_set.frame_num,
            total_detections,
            len(drone_ids),
        )

        return results
