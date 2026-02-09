"""
Batch Detector

Processes all frames in a SynchronizedFrameSet using YOLO.

This module:
- Takes a SynchronizedFrameSet (multiple drone frames)
- Runs YOLO detection on each frame
- Returns map of {drone_id: DetectionSet}

Usage:
    from detection.batch_detector import BatchDetector
    from ingestion.models import SynchronizedFrameSet

    batch_detector = BatchDetector(weights_path="weights/yolov8n.pt")

    # Process synchronized frame set
    all_detections = batch_detector.process(sync_set)

    # Access detections per drone
    for drone_id, detection_set in all_detections.items():
        logger.info("Drone %d: %d detections", drone_id, detection_set.num_detections)
"""

import logging
from typing import Dict

# Use relative imports when running as script
try:
    from detection.models import DetectionSet
    from detection.yolo_detector import YOLODetector
    from ingestion.models import SynchronizedFrameSet
except ImportError:
    from models import DetectionSet
    from yolo_detector import YOLODetector
    import sys
    sys.path.insert(0, '..')
    from ingestion.models import SynchronizedFrameSet


logger = logging.getLogger(__name__)


# =============================================================================
# BATCH DETECTOR
# =============================================================================

class BatchDetector:
    """
    Batch processor for detecting persons across multiple drone frames.

    Uses a single YOLODetector instance to process all frames in a
    SynchronizedFrameSet sequentially (or could be parallelized later).
    """

    def __init__(self):
        """
        Initialize batch detector.

        All settings loaded from config.settings.detection.
        """
        # Create single YOLO detector instance (shared across all drones)
        self.detector = YOLODetector()

        logger.info("Batch detector initialized (device=%s)", self.detector.device)

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

        logger.debug("Processing synchronized set %d (%d drones)",
                    frame_num, num_drones)

        # Process each drone frame
        for drone_id in sorted(sync_set.frames.keys()):
            drone_frame = sync_set.frames[drone_id]

            # Run YOLO detection
            detection_set = self.detector.detect(
                frame=drone_frame.frame,
                drone_id=drone_id,
                frame_num=frame_num
            )

            results[drone_id] = detection_set

        # Log summary
        total_detections = sum(ds.num_detections for ds in results.values())
        logger.info("Frame %d: %d total detections across %d drones",
                   frame_num, total_detections, num_drones)

        return results

    def get_stats(self) -> dict:
        """
        Get detector statistics.

        Returns:
            Dictionary with performance metrics from underlying YOLODetector
        """
        return self.detector.get_stats()

    def reset_stats(self) -> None:
        """Reset detector statistics."""
        self.detector.reset_stats()


# =============================================================================
# DEBUG / TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Test batch detector with real streamer data.

    Requirements:
    1. Start mock_drone_streamer:
       cd ../mock_drone_streamer
       python server.py

    2. Have YOLO weights at weights/yolov8n.pt

    3. Run this test:
       cd algorithm
       python detection/batch_detector.py
    """
    import queue
    import time
    from pathlib import Path

    from ingestion.tcp_receiver import TCPReceiver
    from ingestion.synchronizer import FrameSynchronizer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    )

    logger.info("Testing Batch Detector with REAL Data")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Requirements:")
    logger.info("  1. mock_drone_streamer running (python server.py)")
    logger.info("  2. YOLO weights at weights/best.pt")
    logger.info("")
    logger.info("=" * 60)

    # Queues
    receiver_queue = queue.Queue()
    sync_queue = queue.Queue()

    # Create receivers for 3 drones (using config)
    receivers = {}
    for drone_id in [1, 2, 3]:
        receiver = TCPReceiver(
            drone_id=drone_id,
            output_queue=receiver_queue
        )
        receivers[drone_id] = receiver

    # Create synchronizer (using config)
    sync = FrameSynchronizer(
        input_queue=receiver_queue,
        output_queue=sync_queue
    )

    # Create batch detector (using config)
    batch_detector = BatchDetector()

    logger.info("")
    logger.info("Using config:")
    logger.info("  Weights: %s", batch_detector.detector.weights_path)
    logger.info("  Device: %s", batch_detector.detector.device)
    logger.info("  Confidence: %.2f", batch_detector.detector.conf_threshold)
    logger.info("")

    try:
        # Start pipeline
        logger.info("Starting pipeline...")
        sync.start()

        for drone_id, receiver in receivers.items():
            receiver.start()

        time.sleep(1.0)  # Wait for connections

        logger.info("")
        logger.info("Processing frames with detection...")
        logger.info("-" * 60)

        # Process 5 synchronized sets
        for i in range(5):
            try:
                # Get synchronized set
                sync_set = sync_queue.get(timeout=5.0)

                logger.info("\nSet %d: frame_num=%d, drones=%s",
                           i + 1, sync_set.frame_num, sync_set.drone_ids)

                # Run batch detection
                all_detections = batch_detector.process(sync_set)

                # Show results per drone
                for drone_id, detection_set in all_detections.items():
                    logger.info("  Drone %d: %d detections (%.3fs inference)",
                               drone_id,
                               detection_set.num_detections,
                               detection_set.inference_time)

                    # Show first few detections
                    for det in detection_set.detections[:3]:
                        cx, cy = det.center
                        logger.info("    - Person @ (%.0f, %.0f), conf=%.2f, size=%.0fx%.0f",
                                   cx, cy, det.confidence, det.width, det.height)

            except queue.Empty:
                logger.warning("Timeout waiting for synchronized set")
                break

        logger.info("")
        logger.info("-" * 60)
        logger.info("")
        logger.info("Detection Statistics:")
        stats = batch_detector.get_stats()
        for key, value in stats.items():
            logger.info("  %s: %.4f", key, value)

        logger.info("")
        logger.info("=" * 60)
        logger.info("Test complete!")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")

    finally:
        # Cleanup
        logger.info("\nCleaning up...")
        for receiver in receivers.values():
            receiver.stop()
        sync.stop()
        logger.info("Done")
