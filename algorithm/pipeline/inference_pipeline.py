"""
Inference Pipeline Loop

Async background task that runs the full 5-stage pipeline each frame:
  1. Ingestion (TCPReceiver + FrameSynchronizer running in background threads)
  2. Detection  (BatchDetector)
  3. Features   (WCHExtractor)
  4. Fusion     (CrossCameraMatcher)
  5. Reconstruction (SceneReconstructor)
  6. Tracking   (PersonTracker)

Each processed frame is serialized by format_output() and broadcast
over WebSocket via the ConnectionManager singleton.
"""

import asyncio
import logging
import queue

from algorithm.detection.batch_detector import BatchDetector
from algorithm.features.wch_extractor import WCHExtractor
from algorithm.fusion.cross_camera_matcher import CrossCameraMatcher
from algorithm.reconstruction.scene_reconstructor import SceneReconstructor
from algorithm.tracking.tracker import PersonTracker
from algorithm.ingestion.tcp_receiver import TCPReceiver
from algorithm.ingestion.synchronizer import FrameSynchronizer
from algorithm.api.websocket import ConnectionManager
from algorithm.pipeline.output_formatter import format_output
from algorithm.config.settings import settings

logger = logging.getLogger(__name__)


async def run_pipeline_loop(manager: ConnectionManager) -> None:
    """
    Run the full pipeline as an asyncio background task.

    Initializes all stages once, then loops:
      - Poll sync_queue for new SynchronizedFrameSet
      - Run all 5 stages sequentially
      - Serialize and broadcast result via manager.broadcast_text()

    Handles asyncio.CancelledError by stopping ingestion threads cleanly.

    Args:
        manager: ConnectionManager singleton for broadcasting results
    """
    logger.info("Pipeline loop starting — initializing stages")

    # --- Stage constructors (config-only pattern) ---
    batch_detector = BatchDetector()
    wch_extractor = WCHExtractor(settings.features)
    cross_camera_matcher = CrossCameraMatcher(settings.fusion)
    scene_reconstructor = SceneReconstructor(settings.reconstruction)
    person_tracker = PersonTracker()

    # --- Ingestion: queues and threads ---
    receiver_queue: queue.Queue = queue.Queue()
    sync_queue: queue.Queue = queue.Queue()

    receivers: dict[int, TCPReceiver] = {}
    for drone_id in range(1, settings.ingestion.num_drones + 1):
        receivers[drone_id] = TCPReceiver(
            drone_id=drone_id, output_queue=receiver_queue
        )

    synchronizer = FrameSynchronizer(
        input_queue=receiver_queue, output_queue=sync_queue
    )

    # Start ingestion threads
    synchronizer.start()
    for receiver in receivers.values():
        receiver.start()

    logger.info(
        "Pipeline ready — %d TCP receivers started, synchronizer running",
        len(receivers),
    )

    try:
        while True:
            # Non-blocking poll — yield to event loop if nothing available
            try:
                sync_set = sync_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue

            frame_num = sync_set.frame_num
            logger.debug("Pipeline: processing frame %d", frame_num)

            # Stage 1: Detection
            detection_sets = batch_detector.process(sync_set)

            # Stage 2: Feature extraction (one call per drone frame)
            features_dict: dict = {}
            for drone_id, drone_frame in sync_set.frames.items():
                det_set = detection_sets.get(drone_id)
                if det_set is not None:
                    frame_features = wch_extractor.extract_frame(
                        drone_frame, det_set
                    )
                    features_dict[drone_id] = frame_features.features

            # Stage 3: Cross-camera fusion
            projection_matrices = {
                drone_id: drone_frame.calibration.projection_matrix
                for drone_id, drone_frame in sync_set.frames.items()
            }
            fusion_result = cross_camera_matcher.match_frame(
                detection_sets=detection_sets,
                projection_matrices=projection_matrices,
                features_dict=features_dict,
            )

            # Stage 4: 3D reconstruction
            reconstruction_result = scene_reconstructor.reconstruct(
                fusion_result=fusion_result,
                detection_sets=detection_sets,
                sync_set=sync_set,
            )

            # Stage 5: Temporal tracking
            tracking_result = person_tracker.update(reconstruction_result)

            # Serialize and broadcast
            json_str = format_output(
                tracking_result=tracking_result,
                sync_set=sync_set,
                detection_sets=detection_sets,
            )
            await manager.broadcast_text(json_str)

            logger.info(
                "Frame %d: confirmed=%d, clients=%d, msg_len=%d bytes",
                frame_num,
                len(tracking_result.tracked_persons),
                manager.num_connections,
                len(json_str),
            )

    except asyncio.CancelledError:
        logger.info("Pipeline loop cancelled — stopping ingestion threads")
        for receiver in receivers.values():
            receiver.stop()
        synchronizer.stop()
        logger.info("Pipeline loop stopped cleanly")
        raise
