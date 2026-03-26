"""
Inference Pipeline Loop

Wires all 5 algorithm stages into a single async loop:
  SynchronizedFrameSet -> Detection -> Features -> Fusion -> Reconstruction -> Tracking -> Output

Runs as a background asyncio task started in main.py lifespan.
Posts one StreamPayload per drone per frame to the gateway via HTTP POST.

Threading model:
- TCP receivers + frame synchronizer run in background threads (queue-based)
- This coroutine polls the sync queue with asyncio.sleep to yield control between frames
- CPU-bound stages (YOLO, WCH, etc.) run synchronously — at 2 FPS there is sufficient slack
"""

import asyncio
import logging
import queue
import time

from src.api.gateway_client import post_payload
from src.config.settings import settings
from src.detection.batch_detector import BatchDetector
from src.features.projection_matrix import ProjectionMatrixCalculator
from src.features.wch_extractor import WCHExtractor
from src.fusion.cross_camera_matcher import CrossCameraMatcher
from src.ingestion.synchronizer import FrameSynchronizer
from src.ingestion.tcp_receiver import TCPReceiver
from src.pipeline.output_formatter import build_payloads
from src.reconstruction.scene_reconstructor import SceneReconstructor
from src.tracking.tracker import PersonTracker

logger = logging.getLogger(__name__)


async def run_pipeline_loop() -> None:
    """
    Main pipeline loop. Runs until cancelled (asyncio.CancelledError).

    Initializes all pipeline components, starts TCP receivers + synchronizer
    in background threads, then processes synchronized frame sets one by one.
    POSTs one StreamPayload per drone to the gateway after each frame.
    """
    logger.info("Initializing pipeline components...")

    # --- Initialize all stages ---
    batch_detector = BatchDetector()
    wch_extractor = WCHExtractor(settings.features)
    projection_calc = ProjectionMatrixCalculator()
    cross_camera_matcher = CrossCameraMatcher(settings.fusion)
    scene_reconstructor = SceneReconstructor(settings.reconstruction)
    person_tracker = PersonTracker()

    # --- Initialize ingestion (threading-based) ---
    receiver_queue: queue.Queue = queue.Queue()
    sync_queue: queue.Queue = queue.Queue()

    receivers: dict[int, TCPReceiver] = {}
    for drone_id in range(1, settings.ingestion.num_drones + 1):
        receivers[drone_id] = TCPReceiver(drone_id=drone_id, output_queue=receiver_queue)

    synchronizer = FrameSynchronizer(
        input_queue=receiver_queue,
        output_queue=sync_queue,
    )

    # Start ingestion threads
    synchronizer.start()
    for receiver in receivers.values():
        receiver.start()

    logger.info("Pipeline initialized. Waiting for frames...")

    try:
        while True:
            # Poll sync queue without blocking the event loop
            try:
                sync_set = sync_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)  # yield control, retry in 50ms
                continue

            frame_num = sync_set.frame_num
            pipeline_start = time.monotonic()

            logger.debug(
                "Processing frame %d (%d drones)", frame_num, sync_set.num_drones_present
            )

            try:
                # Stage 1: Detection
                detection_sets = batch_detector.process(sync_set)

                # Stage 2: Feature extraction + projection matrices
                calibrations = sync_set.get_all_calibrations()
                projection_matrices = projection_calc.compute_batch(calibrations)

                features_dict: dict = {}
                for drone_id, drone_frame in sync_set.frames.items():
                    det_set = detection_sets.get(drone_id)
                    if det_set and not det_set.is_empty:
                        frame_features = wch_extractor.extract_frame(
                            frame=drone_frame,
                            detectionSet=det_set,
                        )
                        features_dict[drone_id] = frame_features.features
                    else:
                        features_dict[drone_id] = []

                # Stage 3: Cross-camera fusion
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

                # Output: POST one StreamPayload per drone to gateway
                for payload in build_payloads(
                    result=tracking_result,
                    detection_sets=detection_sets,
                    sync_set=sync_set,
                    pipeline_start_time=pipeline_start,
                ):
                    await post_payload(payload)

                logger.debug(
                    "Frame %d: %d confirmed tracks, posted %d payloads",
                    frame_num,
                    len(tracking_result.tracked_persons),
                    len(sync_set.frames),
                )

            except Exception as e:
                logger.error(
                    "Pipeline error on frame %d: %s", frame_num, e, exc_info=True
                )
                # Continue to next frame — don't crash the loop on single-frame errors

            # Yield to event loop between frames
            await asyncio.sleep(0)

    except asyncio.CancelledError:
        logger.info("Pipeline loop cancelled. Shutting down ingestion...")
        for receiver in receivers.values():
            receiver.stop()
        synchronizer.stop()
        logger.info("Pipeline shutdown complete.")
        raise
