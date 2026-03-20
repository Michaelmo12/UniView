"""
Fusion Integration Test - Real Streaming Pipeline  [SUPERSEDED]

This test covers Fusion only (no Reconstruction).
The full pipeline test (Fusion + Reconstruction) is at:
    algorithm/test_pipeline_integration.py

Usage:
    python -m algorithm.test_pipeline_integration
"""

import logging
import queue
import sys
import time

from src.ingestion.tcp_receiver import create_receivers
from src.ingestion.synchronizer import FrameSynchronizer
from src.detection.batch_detector import BatchDetector
from src.features.wch_extractor import WCHExtractor
from src.fusion.cross_camera_matcher import CrossCameraMatcher
from src.config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

NUM_FRAMES_TO_TEST = 5       # How many synchronized frame sets to process
STREAMER_WARMUP_SEC = 3.0    # Seconds to wait for streamer to start sending
FRAME_TIMEOUT_SEC  = 15.0    # Max wait for one synchronized frame set

# ─── Pipeline Setup ──────────────────────────────────────────────────────────

def build_pipeline():
    """Instantiate all pipeline stages."""
    logger.info("Building pipeline stages...")

    # Two queues: receivers → raw_queue → synchronizer → sync_queue
    raw_queue  = queue.Queue(maxsize=100)
    sync_queue = queue.Queue(maxsize=10)

    drone_ids    = list(range(1, settings.ingestion.num_drones + 1))
    receivers    = create_receivers(drone_ids, output_queue=raw_queue)
    synchronizer = FrameSynchronizer(input_queue=raw_queue, output_queue=sync_queue)
    detector     = BatchDetector()
    extractor    = WCHExtractor(settings.features)
    matcher      = CrossCameraMatcher(settings.fusion)

    logger.info("All stages initialized")
    return sync_queue, receivers, synchronizer, detector, extractor, matcher


def start_streaming(receivers, synchronizer):
    """Start TCP receivers and synchronizer threads."""
    logger.info("Starting %d TCP receivers...", len(receivers))
    for r in receivers.values():
        r.start()

    logger.info("Starting frame synchronizer...")
    synchronizer.start()


def stop_streaming(receivers, synchronizer):
    """Gracefully shut down all background threads."""
    logger.info("Stopping pipeline...")
    synchronizer.stop()
    for r in receivers.values():
        r.stop()


# ─── Per-Frame Processing ────────────────────────────────────────────────────

def process_frame(sync_set, detector, extractor, matcher):
    """
    Run one synchronized frame set through the full pipeline.

    Returns:
        dict with keys: frame_num, detection_counts, features_counts,
                        fusion_result, projection_matrices, features_dict
    """
    frame_num = sync_set.frame_num
    logger.info("─" * 60)
    logger.info("Processing frame set %d (%d cameras present)",
                frame_num, sync_set.num_drones_present)

    # ── Stage 1: Detection ────────────────────────────────────────────────────
    detection_sets = detector.process(sync_set)

    detection_counts = {
        drone_id: ds.num_detections
        for drone_id, ds in detection_sets.items()
    }
    total_detections = sum(detection_counts.values())
    logger.info("Detection: %s  (total=%d)", detection_counts, total_detections)

    # ── Stage 2: Feature Extraction ───────────────────────────────────────────
    features_dict = {}       # {drone_id: [PersonFeatures]}
    projection_matrices = {} # {drone_id: P (3x4)}

    for drone_id, drone_frame in sync_set.frames.items():
        det_set = detection_sets.get(drone_id)
        if det_set is None or det_set.is_empty:
            continue

        frame_features = extractor.extract_frame(drone_frame, det_set)
        features_dict[drone_id] = frame_features.features

        # Projection matrix comes from calibration embedded in the stream packet
        projection_matrices[drone_id] = drone_frame.calibration.projection_matrix

    features_counts = {
        drone_id: len(feats)
        for drone_id, feats in features_dict.items()
    }
    logger.info("Features:  %s", features_counts)

    # ── Stage 3: Fusion ───────────────────────────────────────────────────────
    fusion_result = matcher.match_frame(
        detection_sets=detection_sets,
        projection_matrices=projection_matrices,
        features_dict=features_dict,
    )

    logger.info(
        "Fusion:    %d match groups, %d pairwise matches",
        fusion_result.num_groups,
        fusion_result.total_matches,
    )

    return dict(
        frame_num=frame_num,
        detection_counts=detection_counts,
        features_counts=features_counts,
        fusion_result=fusion_result,
        projection_matrices=projection_matrices,
        features_dict=features_dict,
    )


# ─── Assertions ──────────────────────────────────────────────────────────────

def validate_frame_result(result):
    """
    Run correctness assertions on one frame's fusion output.
    Raises AssertionError on failure.
    """
    fusion_result = result["fusion_result"]
    frame_num     = result["frame_num"]

    # 1. No detection appears in more than one group
    all_dets = []
    for group in fusion_result.match_groups:
        all_dets.extend(group.detections)

    unique_dets = set(all_dets)
    assert len(all_dets) == len(unique_dets), (
        f"Frame {frame_num}: detection appears in multiple groups! "
        f"total={len(all_dets)}, unique={len(unique_dets)}"
    )

    # 2. Every group has detections from at least 2 different cameras
    for i, group in enumerate(fusion_result.match_groups):
        drone_ids = group.get_drone_ids()
        assert len(drone_ids) >= 2, (
            f"Frame {frame_num}: group {i} only has 1 camera ({group.detections})"
        )

    # 3. Appearance scores are in valid range [0, 1]
    for i, group in enumerate(fusion_result.match_groups):
        score = group.mean_appearance_score
        assert 0.0 <= score <= 1.0, (
            f"Frame {frame_num}: group {i} has invalid score {score:.4f}"
        )

    # 4. total_matches is consistent (>= num_groups since each group needs >=1 match)
    if fusion_result.num_groups > 0:
        assert fusion_result.total_matches >= fusion_result.num_groups, (
            f"Frame {frame_num}: total_matches ({fusion_result.total_matches}) "
            f"< num_groups ({fusion_result.num_groups})"
        )


def print_frame_summary(result):
    """Print human-readable breakdown of one frame's fusion output."""
    fusion_result = result["fusion_result"]
    frame_num     = result["frame_num"]

    logger.info("  Frame %d Summary:", frame_num)
    logger.info("    Detections per camera : %s", result["detection_counts"])
    logger.info("    Features per camera   : %s", result["features_counts"])
    logger.info("    Match groups found    : %d", fusion_result.num_groups)
    logger.info("    Total pairwise matches: %d", fusion_result.total_matches)

    for i, group in enumerate(fusion_result.match_groups):
        logger.info(
            "    Group %d: cameras=%s  detections=%s  score=%.3f",
            i + 1,
            sorted(group.get_drone_ids()),
            group.detections,
            group.mean_appearance_score,
        )


# ─── Main Test Runner ────────────────────────────────────────────────────────

def run():
    logger.info("=" * 60)
    logger.info("FUSION INTEGRATION TEST")
    logger.info("Frames to test : %d", NUM_FRAMES_TO_TEST)
    logger.info("Frame timeout  : %.1fs", FRAME_TIMEOUT_SEC)
    logger.info("=" * 60)

    sync_queue, receivers, synchronizer, detector, extractor, matcher = build_pipeline()

    start_streaming(receivers, synchronizer)

    logger.info("Waiting %.1fs for streamer warmup...", STREAMER_WARMUP_SEC)
    time.sleep(STREAMER_WARMUP_SEC)

    results = []
    frames_processed = 0
    frames_with_matches = 0

    try:
        while frames_processed < NUM_FRAMES_TO_TEST:
            try:
                sync_set = sync_queue.get(timeout=FRAME_TIMEOUT_SEC)
            except queue.Empty:
                logger.error(
                    "No synchronized frame received within %.1fs. "
                    "Is mock_drone_streamer running?",
                    FRAME_TIMEOUT_SEC,
                )
                sys.exit(1)

            result = process_frame(sync_set, detector, extractor, matcher)
            validate_frame_result(result)
            print_frame_summary(result)
            results.append(result)

            frames_processed += 1
            if result["fusion_result"].num_groups > 0:
                frames_with_matches += 1

    finally:
        stop_streaming(receivers, synchronizer)

    # ── Final Report ──────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL REPORT")
    logger.info("=" * 60)

    total_groups   = sum(r["fusion_result"].num_groups   for r in results)
    total_matches  = sum(r["fusion_result"].total_matches for r in results)
    total_dets     = sum(sum(r["detection_counts"].values()) for r in results)

    logger.info("Frames processed       : %d / %d", frames_processed, NUM_FRAMES_TO_TEST)
    logger.info("Frames with matches    : %d / %d", frames_with_matches, frames_processed)
    logger.info("Total detections       : %d", total_dets)
    logger.info("Total match groups     : %d", total_groups)
    logger.info("Total pairwise matches : %d", total_matches)

    if total_dets > 0:
        logger.info(
            "Match rate             : %.1f%%",
            100.0 * total_matches / total_dets,
        )

    logger.info("")
    logger.info("All assertions passed!")
    logger.info("Fusion pipeline is working correctly with real streaming data.")


if __name__ == "__main__":
    run()
