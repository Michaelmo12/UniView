"""
Inference Pipeline Loop

Wires all 5 algorithm stages into a single async loop:
  SynchronizedFrameSet -> Detection -> Features -> Fusion -> Reconstruction -> Tracking -> Output

Runs as a background asyncio task started in main.py lifespan.
Posts one StreamPayload per drone per frame to the gateway via HTTP POST.

Threading model:
- ENet receiver runs in a separate OS process (ReceiverProcess) 
- SynchronizedFrameSets cross the process boundary via multiprocessing.Queue (pickle).
- CPU-bound stages (YOLO, WCH, etc.) run synchronously — at 2 FPS there is sufficient slack.
"""

# runs the pipeline as an async loop without blocking on I/O
import asyncio

# logging
import logging

# for creating the cross-process queue that carries SynchronizedFrameSets from the receiver child process
import multiprocessing

# needed only to catch queue.Empty when polling sync_queue without blocking
import queue

# for measuring how long each pipeline stage takes per frame
import time

# posts one StreamPayload per drone to the gateway via HTTP POST
from src.api.gateway_client import post_payload

from src.config.settings import settings
from src.detection.batch_detector import BatchDetector
from src.features.projection_matrix import ProjectionMatrixCalculator
from src.features.wch_extractor import WCHExtractor
from src.fusion.cross_camera_matcher import CrossCameraMatcher
from src.ingestion.receiver_process import create_receiver_process
from src.pipeline.output_formatter import build_payloads
from src.reconstruction.scene_reconstructor import SceneReconstructor
from src.tracking.tracker import PersonTracker

logger = logging.getLogger(__name__)


# async def — this is a coroutine, must be awaited by the event loop; allows using await inside
async def run_pipeline_loop() -> None:
    """
    Main pipeline loop. Runs until cancelled (asyncio.CancelledError).

    Initializes all pipeline components, starts TCP receivers + synchronizer
    in background threads, then processes synchronized frame sets one by one.
    POSTs one StreamPayload per drone to the gateway after each frame.
    """
    logger.info("Initializing pipeline components...")

    try:
        # create all stage objects once at startup — reused for every frame
        # batch_mode=True triggers OpenVINO throughput warmup (batch size > 1 on first call)
        batch_detector = BatchDetector(batch_mode=True)

        # WCH extractor reads feature settings (bin counts, body split ratio)
        wch_extractor = WCHExtractor(settings.features)

        # stateless — computes P = K @ [R|t] on demand
        projection_calc = ProjectionMatrixCalculator()

        # reads fusion settings (epipolar distance threshold, appearance weight)
        cross_camera_matcher = CrossCameraMatcher(settings.fusion)

        # reads reconstruction settings (DBSCAN eps, min_samples)
        scene_reconstructor = SceneReconstructor(settings.reconstruction)

        # stateful — holds Kalman tracks across frames
        person_tracker = PersonTracker()

        # cross-process queue — SynchronizedFrameSets travel from child process to here via pickle
        # maxsize=2 — if pipeline falls behind, 3rd frame is dropped instead of filling memory
        sync_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=2)

        # drone IDs to listen to (e.g. [3, 4, 6, 7]) from config
        drone_ids = settings.ingestion.drone_ids
        # factory: creates ReceiverProcess configured from settings
        receiver_process = create_receiver_process(
            drone_ids,
            output_queue=sync_queue,
        )
        # spawns the child OS process — ENet receiver starts running independently from here
        receiver_process.start()
    except Exception as exc:
        # log full stack trace and re-raise — caller knows init failed
        logger.error("Pipeline initialization failed: %s", exc, exc_info=True)
        raise

    logger.info("Pipeline initialized. Waiting for frames...")

    try:
        # runs forever — one iteration = one synchronized frame set
        while True:
            try:
                # non-blocking pull — raises queue.Empty immediately if nothing is there
                sync_set = sync_queue.get_nowait()
            except queue.Empty:
                # nothing arrived — yield control to event loop for 50ms then retry
                await asyncio.sleep(0.05)
                continue

            # record when this frame was dequeued — used to measure end-to-end latency
            t_dequeued = time.monotonic()
            frame_num = sync_set.frame_num
            pipeline_start = t_dequeued

            # compute gap since last frame — stored as attribute on the function object (avoids global)
            if not hasattr(run_pipeline_loop, "_last_frame_t"):
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
                # t0–t5 bracket each stage so we can log per-stage timing
                t0 = time.monotonic()

                # Stage 1: Detection — sends all drone frames to YOLO in one batch
                # returns {drone_id: DetectionSet}
                detection_sets = batch_detector.process_batch(sync_set)
                t1 = time.monotonic()

                # Stage 2a: get calibration per drone then compute projection matrix P per drone
                calibrations = sync_set.get_all_calibrations()
                # returns {drone_id: P} where P = K @ [R|t], shape (3,4)
                projection_matrices = projection_calc.compute_batch(calibrations)

                # Stage 2b: extract WCH appearance vectors for each detection on each drone
                features_dict: dict = {}
                for drone_id, drone_frame in sync_set.frames.items():
                    det_set = detection_sets.get(drone_id)
                    if det_set and not det_set.is_empty:
                        _tw0 = time.monotonic()
                        # returns FrameFeatures containing list of 96-dim WCH vectors
                        frame_features = wch_extractor.extract_frame(
                            frame=drone_frame,
                            detectionSet=det_set,
                        )
                        logger.info(
                            "Frame %d drone %d WCH: %.1fms (%d dets)",
                            frame_num,
                            drone_id,
                            (time.monotonic() - _tw0) * 1000,
                            len(det_set.detections),
                        )
                        # store list of feature vectors indexed by drone_id
                        features_dict[drone_id] = frame_features.features
                    else:
                        # no detections on this drone — empty list
                        features_dict[drone_id] = []
                t2 = time.monotonic()

                # Stage 3: cross-camera fusion — matches same person across cameras
                # uses epipolar geometry (F matrix) + WCH appearance similarity
                fusion_result = cross_camera_matcher.match_frame(
                    detection_sets=detection_sets,
                    projection_matrices=projection_matrices,
                    features_dict=features_dict,
                )
                t3 = time.monotonic()

                # Stage 4: 3D reconstruction — triangulates matched detections into world positions
                # also handles single-view persons (seen by only one camera)
                reconstruction_result = scene_reconstructor.reconstruct(
                    fusion_result=fusion_result,
                    detection_sets=detection_sets,
                    sync_set=sync_set,
                )
                t4 = time.monotonic()

                # Stage 5: temporal tracking — Kalman filter assigns stable global_id across frames
                tracking_result = person_tracker.update(reconstruction_result)
                t5 = time.monotonic()

                # build per-stage timing dict in milliseconds for logging and payload
                stage_timings_ms = {
                    "detection": round((t1 - t0) * 1000, 2),
                    "features": round((t2 - t1) * 1000, 2),
                    "fusion": round((t3 - t2) * 1000, 2),
                    "reconstruction": round((t4 - t3) * 1000, 2),
                    "tracking": round((t5 - t4) * 1000, 2),
                    "total": round((t5 - t0) * 1000, 2),
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

                # output — build one StreamPayload per drone and POST each to the gateway
                # await — HTTP is I/O so we yield to event loop while waiting for response
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
                # single-frame error — log and continue to next frame instead of crashing the loop
                logger.error(
                    "Pipeline error on frame %d: %s", frame_num, e, exc_info=True
                )

            # yield to event loop between frames so HTTP server can handle requests
            await asyncio.sleep(0)

    except asyncio.CancelledError:
        # server is shutting down — stop the child process cleanly before exiting
        logger.info("Pipeline loop cancelled. Shutting down ingestion...")
        # wait up to 15s for child to exit cleanly, then force-kill
        receiver_process.stop(timeout=15.0)
        logger.info("Pipeline shutdown complete.")
        # re-raise so asyncio knows the cancellation was handled
        raise
