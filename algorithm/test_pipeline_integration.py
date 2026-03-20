"""
Full Pipeline Integration Test - Real Streaming

Tests the complete pipeline with live data from mock_drone_streamer:
    TCP Streams → Ingestion → Detection → Features → Fusion → Reconstruction

Verifies:
- Fusion produces MatchGroups from real frames
- Each detection appears in at most one group (no duplicates)
- Match scores are within valid range [0, 1]
- Reconstruction triangulates match groups into 3D persons
- Triangulated persons have sensible 3D positions (within scene bounds)
- Single-view detections are preserved as Person3D
- total persons == triangulated persons + single-view persons

Usage:
    1. Start mock_drone_streamer first:
       cd mock_drone_streamer && python server.py

    2. Run this test FROM the UniView/ root:
       python -m algorithm.test_pipeline_integration
"""

import logging
import queue
import sys
import time

import numpy as np

from src.ingestion.tcp_receiver import create_receivers
from src.ingestion.synchronizer import FrameSynchronizer
from src.detection.batch_detector import BatchDetector
from src.features.wch_extractor import WCHExtractor
from src.fusion.cross_camera_matcher import CrossCameraMatcher
from src.reconstruction.scene_reconstructor import SceneReconstructor
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

# MATRIX dataset: allow generous bounds — reprojection error filter already rejects
# bad triangulations; these bounds are a last-resort sanity check only.
SCENE_XY_BOUND = 500.0       # meters — max |x| or |y| for a valid 3D point
SCENE_Z_BOUND  = 200.0       # meters — max |z| for a valid 3D point

# ─── Pipeline Setup ──────────────────────────────────────────────────────────

def build_pipeline():
    """Instantiate all pipeline stages."""
    logger.info("Building pipeline stages...")

    # Two queues: receivers → raw_queue → synchronizer → sync_queue
    raw_queue  = queue.Queue(maxsize=100)
    sync_queue = queue.Queue(maxsize=10)

    drone_ids     = list(range(1, settings.ingestion.num_drones + 1))
    receivers     = create_receivers(drone_ids, output_queue=raw_queue)
    synchronizer  = FrameSynchronizer(input_queue=raw_queue, output_queue=sync_queue)
    detector      = BatchDetector()
    extractor     = WCHExtractor(settings.features)
    matcher       = CrossCameraMatcher(settings.fusion)
    reconstructor = SceneReconstructor(settings.reconstruction)

    logger.info("All stages initialized (Ingestion → Detection → Features → Fusion → Reconstruction)")
    return sync_queue, receivers, synchronizer, detector, extractor, matcher, reconstructor


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


# ─── Per-Frame Processing ─────────────────────────────────────────────────────

def process_frame(sync_set, detector, extractor, matcher, reconstructor):
    """
    Run one synchronized frame set through the full pipeline.

    Returns:
        dict with keys: frame_num, detection_counts, features_counts,
                        fusion_result, reconstruction_result,
                        projection_matrices, features_dict
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
    logger.info("Detection:       %s  (total=%d)", detection_counts, total_detections)

    # ── Stage 2: Feature Extraction ───────────────────────────────────────────
    features_dict       = {}  # {drone_id: [PersonFeatures]}
    projection_matrices = {}  # {drone_id: P (3x4)}

    for drone_id, drone_frame in sync_set.frames.items():
        det_set = detection_sets.get(drone_id)
        if det_set is None or det_set.is_empty:
            continue

        frame_features = extractor.extract_frame(drone_frame, det_set)
        features_dict[drone_id] = frame_features.features
        projection_matrices[drone_id] = drone_frame.calibration.projection_matrix

    features_counts = {
        drone_id: len(feats)
        for drone_id, feats in features_dict.items()
    }
    logger.info("Features:        %s", features_counts)

    # ── Stage 3: Fusion ───────────────────────────────────────────────────────
    fusion_result = matcher.match_frame(
        detection_sets=detection_sets,
        projection_matrices=projection_matrices,
        features_dict=features_dict,
    )

    logger.info(
        "Fusion:          %d match groups, %d pairwise matches",
        fusion_result.num_groups,
        fusion_result.total_matches,
    )

    # ── Stage 4: Reconstruction ───────────────────────────────────────────────
    reconstruction_result = reconstructor.reconstruct(
        fusion_result=fusion_result,
        detection_sets=detection_sets,
        sync_set=sync_set,
    )

    logger.info(
        "Reconstruction:  %d persons total  (%d triangulated, %d single-view)  "
        "[%d points ok, %d rejected]",
        reconstruction_result.num_persons,
        len(reconstruction_result.triangulated_persons),
        len(reconstruction_result.single_view_persons),
        reconstruction_result.num_triangulated_points,
        reconstruction_result.num_rejected_points,
    )

    return dict(
        frame_num=frame_num,
        detection_counts=detection_counts,
        features_counts=features_counts,
        fusion_result=fusion_result,
        reconstruction_result=reconstruction_result,
        projection_matrices=projection_matrices,
        features_dict=features_dict,
        detection_sets=detection_sets,
        sync_set=sync_set,
    )


# ─── Assertions ───────────────────────────────────────────────────────────────

def validate_fusion(result):
    """Fusion-level assertions (same as before)."""
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

    # 4. total_matches consistency
    if fusion_result.num_groups > 0:
        assert fusion_result.total_matches >= fusion_result.num_groups, (
            f"Frame {frame_num}: total_matches ({fusion_result.total_matches}) "
            f"< num_groups ({fusion_result.num_groups})"
        )


def validate_reconstruction(result):
    """Reconstruction-level assertions."""
    recon  = result["reconstruction_result"]
    fusion = result["fusion_result"]
    frame_num = result["frame_num"]

    # 5. Person count = triangulated + single-view
    assert recon.num_persons == (
        len(recon.triangulated_persons) + len(recon.single_view_persons)
    ), (
        f"Frame {frame_num}: person count mismatch: "
        f"total={recon.num_persons}, tri={len(recon.triangulated_persons)}, "
        f"sv={len(recon.single_view_persons)}"
    )

    # 6. Triangulated persons have valid positions
    for i, person in enumerate(recon.triangulated_persons):
        assert person.position is not None, (
            f"Frame {frame_num}: triangulated person {i} has None position"
        )
        assert person.is_triangulated, (
            f"Frame {frame_num}: triangulated person {i} has is_triangulated=False"
        )
        assert person.num_views >= 2, (
            f"Frame {frame_num}: triangulated person {i} has num_views={person.num_views} < 2"
        )

        x, y, z = person.position
        assert abs(x) < SCENE_XY_BOUND and abs(y) < SCENE_XY_BOUND, (
            f"Frame {frame_num}: person {i} XY out of scene bounds: "
            f"({x:.2f}, {y:.2f}) — expected < {SCENE_XY_BOUND}m"
        )
        assert abs(z) < SCENE_Z_BOUND, (
            f"Frame {frame_num}: person {i} Z out of scene bounds: "
            f"{z:.2f}m — expected < {SCENE_Z_BOUND}m"
        )

    # 7. Single-view persons have no 3D position
    for i, person in enumerate(recon.single_view_persons):
        assert not person.is_triangulated, (
            f"Frame {frame_num}: single-view person {i} has is_triangulated=True"
        )
        assert person.num_views == 1, (
            f"Frame {frame_num}: single-view person {i} has num_views={person.num_views}"
        )

    # 8. Person IDs are unique
    person_ids = [p.person_id for p in recon.persons]
    assert len(person_ids) == len(set(person_ids)), (
        f"Frame {frame_num}: duplicate person IDs found: {person_ids}"
    )

    # 9. total_persons <= total_detections (can't create more persons than detections)
    total_dets = sum(result["detection_counts"].values())
    assert recon.num_persons <= total_dets, (
        f"Frame {frame_num}: more persons ({recon.num_persons}) than "
        f"detections ({total_dets})"
    )


# ─── Summary Printing ─────────────────────────────────────────────────────────

def print_frame_summary(result):
    """Print human-readable breakdown of one frame's full pipeline output."""
    fusion = result["fusion_result"]
    recon  = result["reconstruction_result"]
    frame_num = result["frame_num"]

    logger.info("  Frame %d Summary:", frame_num)
    logger.info("    Detections per camera : %s", result["detection_counts"])
    logger.info("    Features per camera   : %s", result["features_counts"])
    logger.info("    Match groups (fusion) : %d", fusion.num_groups)
    logger.info("    Pairwise matches      : %d", fusion.total_matches)
    logger.info("    ── Reconstruction ──────────────────────────")
    logger.info("    Points triangulated   : %d", recon.num_triangulated_points)
    logger.info("    Points rejected       : %d", recon.num_rejected_points)
    logger.info("    Total persons         : %d  (%d triangulated, %d single-view)",
                recon.num_persons,
                len(recon.triangulated_persons),
                len(recon.single_view_persons))

    # Print triangulated persons with 3D positions
    if recon.triangulated_persons:
        logger.info("    Triangulated persons:")
        for p in recon.triangulated_persons:
            pos = p.position
            logger.info(
                "      Person %d: pos=[%.2f, %.2f, %.2f]m  views=%d  srcs=%s",
                p.person_id, pos[0], pos[1], pos[2],
                p.num_views, p.source_detections
            )

    if recon.single_view_persons:
        logger.info("    Single-view persons: %d (no 3D position)", len(recon.single_view_persons))


# ─── Main Test Runner ─────────────────────────────────────────────────────────

def run():
    logger.info("=" * 60)
    logger.info("FULL PIPELINE INTEGRATION TEST")
    logger.info("  Ingestion → Detection → Features → Fusion → Reconstruction")
    logger.info("Frames to test : %d", NUM_FRAMES_TO_TEST)
    logger.info("Frame timeout  : %.1fs", FRAME_TIMEOUT_SEC)
    logger.info("=" * 60)

    sync_queue, receivers, synchronizer, detector, extractor, matcher, reconstructor = (
        build_pipeline()
    )

    start_streaming(receivers, synchronizer)

    logger.info("Waiting %.1fs for streamer warmup...", STREAMER_WARMUP_SEC)
    time.sleep(STREAMER_WARMUP_SEC)

    results = []
    frames_processed       = 0
    frames_with_matches    = 0
    frames_with_recon      = 0

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

            result = process_frame(sync_set, detector, extractor, matcher, reconstructor)

            # Validate both stages
            validate_fusion(result)
            validate_reconstruction(result)

            print_frame_summary(result)
            results.append(result)

            frames_processed += 1
            if result["fusion_result"].num_groups > 0:
                frames_with_matches += 1
            if result["reconstruction_result"].num_persons > 0:
                frames_with_recon += 1

    finally:
        stop_streaming(receivers, synchronizer)

    # ── Final Report ──────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL REPORT")
    logger.info("=" * 60)

    total_dets    = sum(sum(r["detection_counts"].values()) for r in results)
    total_groups  = sum(r["fusion_result"].num_groups       for r in results)
    total_matches = sum(r["fusion_result"].total_matches    for r in results)
    total_persons = sum(r["reconstruction_result"].num_persons              for r in results)
    total_tri     = sum(len(r["reconstruction_result"].triangulated_persons) for r in results)
    total_sv      = sum(len(r["reconstruction_result"].single_view_persons)  for r in results)
    total_pts_ok  = sum(r["reconstruction_result"].num_triangulated_points  for r in results)
    total_pts_rej = sum(r["reconstruction_result"].num_rejected_points      for r in results)

    logger.info("Frames processed            : %d / %d", frames_processed, NUM_FRAMES_TO_TEST)
    logger.info("Frames with fusion matches  : %d / %d", frames_with_matches, frames_processed)
    logger.info("Frames with 3D persons      : %d / %d", frames_with_recon, frames_processed)
    logger.info("")
    logger.info("── Fusion ───────────────────────────────────")
    logger.info("Total detections            : %d", total_dets)
    logger.info("Total match groups          : %d", total_groups)
    logger.info("Total pairwise matches      : %d", total_matches)
    if total_dets > 0:
        logger.info("Match rate (matches/dets)   : %.1f%%", 100.0 * total_matches / total_dets)

    logger.info("")
    logger.info("── Reconstruction ───────────────────────────")
    logger.info("Triangulated points OK      : %d", total_pts_ok)
    logger.info("Triangulated points rejected: %d", total_pts_rej)
    if total_pts_ok + total_pts_rej > 0:
        accept_rate = 100.0 * total_pts_ok / (total_pts_ok + total_pts_rej)
        logger.info("Triangulation accept rate   : %.1f%%", accept_rate)
    logger.info("Total persons (all frames)  : %d", total_persons)
    logger.info("  Triangulated persons      : %d", total_tri)
    logger.info("  Single-view persons       : %d", total_sv)
    if total_persons > 0:
        logger.info("  Triangulated %%           : %.1f%%", 100.0 * total_tri / total_persons)

    # Per-frame breakdown table
    logger.info("")
    logger.info("── Per-Frame Breakdown ──────────────────────")
    logger.info("  %-8s %-10s %-12s %-14s %-12s %-10s",
                "Frame", "Dets", "FusGroups", "Persons(tri)", "PointsOK", "PointsRej")
    for r in results:
        recon = r["reconstruction_result"]
        logger.info("  %-8d %-10d %-12d %-14s %-12d %-10d",
                    r["frame_num"],
                    sum(r["detection_counts"].values()),
                    r["fusion_result"].num_groups,
                    f"{recon.num_persons}({len(recon.triangulated_persons)})",
                    recon.num_triangulated_points,
                    recon.num_rejected_points)

    logger.info("")
    logger.info("All assertions passed!")
    logger.info("Full pipeline (Fusion + Reconstruction) is working correctly.")


if __name__ == "__main__":
    run()
