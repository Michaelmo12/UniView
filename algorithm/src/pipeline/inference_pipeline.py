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
from src.ingestion.enet_receiver import create_enet_receivers, start_all_enet_receivers, stop_all_enet_receivers
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

    try:
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

        drone_ids = settings.ingestion.drone_ids
        receivers = create_enet_receivers(drone_ids, output_queue=receiver_queue)

        synchronizer = FrameSynchronizer(
            input_queue=receiver_queue,
            output_queue=sync_queue,
        )

        # Start ingestion threads
        synchronizer.start()
        start_all_enet_receivers(receivers)
    except Exception as exc:
        logger.error("Pipeline initialization failed: %s", exc, exc_info=True)
        raise

    logger.info("Pipeline initialized. Waiting for frames...")

    try:
        while True:
            # Poll sync queue without blocking the event loop
            try:
                sync_set = sync_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)  # yield control, retry in 50ms
                continue

            t_dequeued = time.monotonic()
            frame_num = sync_set.frame_num
            pipeline_start = t_dequeued

            if not hasattr(run_pipeline_loop, '_last_frame_t'):
                run_pipeline_loop._last_frame_t = t_dequeued
            gap_ms = (t_dequeued - run_pipeline_loop._last_frame_t) * 1000
            run_pipeline_loop._last_frame_t = t_dequeued

            logger.info(
                "Frame %d dequeued (%d/%d drones) — gap since last frame: %.0fms",
                frame_num,
                sync_set.num_drones_present,
                sync_set.num_drones_expected,
                gap_ms,
            )

            try:
                t0 = time.monotonic()

                # Stage 1: Detection
                detection_sets = batch_detector.process(sync_set)
                t1 = time.monotonic()

                # Stage 2: Feature extraction + projection matrices
                calibrations = sync_set.get_all_calibrations()
                projection_matrices = projection_calc.compute_batch(calibrations)

                features_dict: dict = {}
                for drone_id, drone_frame in sync_set.frames.items():
                    det_set = detection_sets.get(drone_id)
                    if det_set and not det_set.is_empty:
                        _tw0 = time.monotonic()
                        frame_features = wch_extractor.extract_frame(
                            frame=drone_frame,
                            detectionSet=det_set,
                        )
                        logger.info(
                            "Frame %d drone %d WCH: %.1fms (%d dets)",
                            frame_num, drone_id,
                            (time.monotonic() - _tw0) * 1000,
                            len(det_set.detections),
                        )
                        features_dict[drone_id] = frame_features.features
                    else:
                        features_dict[drone_id] = []
                t2 = time.monotonic()

                # Stage 3: Cross-camera fusion
                fusion_result = cross_camera_matcher.match_frame(
                    detection_sets=detection_sets,
                    projection_matrices=projection_matrices,
                    features_dict=features_dict,
                )
                t3 = time.monotonic()

                # Stage 4: 3D reconstruction
                reconstruction_result = scene_reconstructor.reconstruct(
                    fusion_result=fusion_result,
                    detection_sets=detection_sets,
                    sync_set=sync_set,
                )
                t4 = time.monotonic()

                # Stage 5: Temporal tracking
                tracking_result = person_tracker.update(reconstruction_result)
                t5 = time.monotonic()

                stage_timings_ms = {
                    "detection":      round((t1 - t0) * 1000, 2),
                    "features":       round((t2 - t1) * 1000, 2),
                    "fusion":         round((t3 - t2) * 1000, 2),
                    "reconstruction": round((t4 - t3) * 1000, 2),
                    "tracking":       round((t5 - t4) * 1000, 2),
                    "total":          round((t5 - t0) * 1000, 2),
                }

                logger.info(
                    "Frame %d timings (ms): det=%.1f feat=%.1f fus=%.1f rec=%.1f trk=%.1f | total=%.1f",
                    frame_num,
                    stage_timings_ms["detection"],
                    stage_timings_ms["features"],
                    stage_timings_ms["fusion"],
                    stage_timings_ms["reconstruction"],
                    stage_timings_ms["tracking"],
                    stage_timings_ms["total"],
                )

                # Output: POST one StreamPayload per drone to gateway
                for payload in build_payloads(
                    result=tracking_result,
                    detection_sets=detection_sets,
                    sync_set=sync_set,
                    pipeline_start_time=pipeline_start,
                    stage_timings_ms=stage_timings_ms,
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
        stop_all_enet_receivers(receivers)
        synchronizer.stop()
        logger.info("Pipeline shutdown complete.")
        raise
