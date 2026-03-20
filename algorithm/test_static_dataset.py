"""
Static Dataset Pipeline Test

Runs the full pipeline on fixed frames from the MATRIX_30X30 dataset.
No streamer needed — reads images and calibrations directly from disk.

This enables reproducible benchmarking: same frames every run, so you can
compare the effect of changing config parameters (thresholds, DBSCAN eps, etc.)

Usage:
    python -m algorithm.test_static_dataset

Output:
    Per-frame pipeline results + comparison against ground truth 3D positions.
"""

import logging
import time
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import json
from pathlib import Path

from src.detection.batch_detector import BatchDetector
from src.features.wch_extractor import WCHExtractor
from src.fusion.cross_camera_matcher import CrossCameraMatcher
from src.reconstruction.scene_reconstructor import SceneReconstructor
from src.ingestion.models import CameraCalibration, DroneFrame, SynchronizedFrameSet
from src.config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

DATASET_ROOT = Path("MATRIX_30X30/MATRIX_30x30")

# Fixed frames to test — change these to benchmark different frames
FRAMES_TO_TEST = [0, 1, 2, 3, 4]

NUM_DRONES = 8

# ─── Dataset Loaders ──────────────────────────────────────────────────────────

def load_calibration(drone_id: int, frame_num: int) -> CameraCalibration:
    """Load intrinsic + extrinsic calibration for one drone/frame."""
    intr_path = DATASET_ROOT / "calibrations" / "intrinsic" / f"intr_Drone{drone_id}_{frame_num:04d}.xml"
    extr_path = DATASET_ROOT / "calibrations" / "extrinsic" / f"extr_Drone{drone_id}_{frame_num:04d}.xml"

    # Parse intrinsic
    intr_tree = ET.parse(intr_path)
    intr_root = intr_tree.getroot()

    K_node = intr_root.find("camera_matrix")
    K_data = list(map(float, K_node.find("data").text.split()))
    K = np.array(K_data, dtype=np.float32).reshape(3, 3)

    dist_node = intr_root.find("distortion_coefficients")
    dist_data = list(map(float, dist_node.find("data").text.split()))
    dist = np.array(dist_data, dtype=np.float32).flatten()

    # Parse extrinsic — rvec and tvec are binary-encoded in the XML
    fs = cv2.FileStorage(str(extr_path), cv2.FILE_STORAGE_READ)
    rvec = fs.getNode("rvec").mat()  # (3, 1)
    tvec = fs.getNode("tvec").mat()  # (3, 1)
    fs.release()

    # Convert Rodrigues rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    R = R.astype(np.float32)
    t = tvec.reshape(3, 1).astype(np.float32)

    return CameraCalibration(K=K, R=R, t=t, dist=dist)


def load_frame(drone_id: int, frame_num: int) -> np.ndarray:
    """Load BGR image for one drone/frame."""
    img_path = DATASET_ROOT / "image_subsets" / f"D{drone_id}" / f"{frame_num:04d}.png"
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return img


def load_sync_set(frame_num: int) -> SynchronizedFrameSet:
    """Build a SynchronizedFrameSet from disk for a given frame number."""
    frames = {}
    for drone_id in range(1, NUM_DRONES + 1):
        calibration = load_calibration(drone_id, frame_num)
        image = load_frame(drone_id, frame_num)
        frames[drone_id] = DroneFrame(
            drone_id=drone_id,
            frame_num=frame_num,
            timestamp=float(frame_num),  # use frame number as timestamp
            frame=image,
            calibration=calibration,
        )
    return SynchronizedFrameSet(
        frame_num=frame_num,
        timestamp=float(frame_num),
        frames=frames,
        num_drones_expected=NUM_DRONES,
    )


def load_ground_truth(frame_num: int) -> list[tuple[float, float, float]]:
    """
    Load ground truth 3D positions from the LoS matching files.
    Returns list of (x, y, z) for all persons visible in this frame.
    """
    gt_path = DATASET_ROOT / "matchings" / "Pedestrians" / "LoS" / f"Drone1_3d_{frame_num:04d}.txt"
    if not gt_path.exists():
        return []

    positions = []
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                positions.append((x, y, z))
    return positions


# ─── Pipeline Processing ──────────────────────────────────────────────────────

def process_frame(sync_set, detector, extractor, matcher, reconstructor):
    """Run one frame through the full pipeline. Returns result dict."""
    frame_num = sync_set.frame_num

    # Stage 1: Detection
    detection_sets = detector.process(sync_set)
    detection_counts = {d: ds.num_detections for d, ds in detection_sets.items()}
    total_detections = sum(detection_counts.values())

    # Stage 2: Feature Extraction
    features_dict = {}
    projection_matrices = {}
    for drone_id, drone_frame in sync_set.frames.items():
        det_set = detection_sets.get(drone_id)
        if det_set is None or det_set.is_empty:
            continue
        frame_features = extractor.extract_frame(drone_frame, det_set)
        features_dict[drone_id] = frame_features.features
        projection_matrices[drone_id] = drone_frame.calibration.projection_matrix

    # Stage 3: Fusion
    fusion_result = matcher.match_frame(
        detection_sets=detection_sets,
        projection_matrices=projection_matrices,
        features_dict=features_dict,
    )

    # Stage 4: Reconstruction
    reconstruction_result = reconstructor.reconstruct(
        fusion_result=fusion_result,
        detection_sets=detection_sets,
        sync_set=sync_set,
    )

    logger.info(
        "Frame %d | dets=%d | groups=%d | persons=%d (%d tri, %d sv) | pts ok=%d rej=%d",
        frame_num, total_detections,
        fusion_result.num_groups,
        reconstruction_result.num_persons,
        len(reconstruction_result.triangulated_persons),
        len(reconstruction_result.single_view_persons),
        reconstruction_result.num_triangulated_points,
        reconstruction_result.num_rejected_points,
    )

    return dict(
        frame_num=frame_num,
        detection_counts=detection_counts,
        fusion_result=fusion_result,
        reconstruction_result=reconstruction_result,
    )


# ─── Ground Truth Comparison ──────────────────────────────────────────────────

def compare_to_ground_truth(result, gt_positions):
    """
    For each triangulated person, find the nearest GT person and report distance.
    Returns list of (our_pos, nearest_gt_pos, distance_m).
    """
    recon = result["reconstruction_result"]
    frame_num = result["frame_num"]

    if not gt_positions:
        logger.info("  No GT available for frame %d", frame_num)
        return

    gt_array = np.array(gt_positions)  # (N, 3)

    logger.info("  Ground Truth: %d persons in scene (Z=0 ground plane)", len(gt_positions))
    logger.info("  Our triangulated persons: %d", len(recon.triangulated_persons))

    matched = 0
    outliers = 0

    for person in recon.triangulated_persons:
        pos = person.position
        # Distance to every GT position (XY only — GT is always Z=0)
        diffs = gt_array[:, :2] - pos[:2]
        distances = np.sqrt((diffs ** 2).sum(axis=1))
        nearest_idx = np.argmin(distances)
        nearest_dist = distances[nearest_idx]
        nearest_gt = gt_positions[nearest_idx]

        is_outlier = abs(pos[0]) > 50 or abs(pos[1]) > 50 or abs(pos[2]) > 20
        status = "OUTLIER" if is_outlier else (f"dist={nearest_dist:.1f}m" if nearest_dist < 5 else f"FAR({nearest_dist:.1f}m)")

        if is_outlier:
            outliers += 1
        elif nearest_dist < 5:
            matched += 1

        logger.info(
            "    Person %d: [%.1f, %.1f, %.1f]m  →  nearest GT [%.1f, %.1f]  %s",
            person.person_id, pos[0], pos[1], pos[2],
            nearest_gt[0], nearest_gt[1], status,
        )

    logger.info(
        "  Summary: %d matched (within 5m of GT), %d outliers, %d total triangulated",
        matched, outliers, len(recon.triangulated_persons)
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    logger.info("=" * 60)
    logger.info("STATIC DATASET PIPELINE TEST")
    logger.info("  Frames: %s", FRAMES_TO_TEST)
    logger.info("  appearance_threshold: %.2f", settings.fusion.appearance_threshold)
    logger.info("  epipolar_threshold:   %.1f px", settings.fusion.epipolar_threshold)
    logger.info("  max_reprojection_error: %.1f px", settings.reconstruction.max_reprojection_error)
    logger.info("  dbscan_eps: %.1f m", settings.reconstruction.dbscan_eps)
    logger.info("=" * 60)

    # Build pipeline
    detector     = BatchDetector()
    extractor    = WCHExtractor(settings.features)
    matcher      = CrossCameraMatcher(settings.fusion)
    reconstructor = SceneReconstructor(settings.reconstruction)

    results = []
    total_matched = 0
    total_outliers = 0
    total_triangulated = 0

    for frame_num in FRAMES_TO_TEST:
        logger.info("─" * 60)
        logger.info("Loading frame %d from disk...", frame_num)

        t0 = time.time()
        sync_set = load_sync_set(frame_num)
        load_time = time.time() - t0
        logger.info("  Loaded in %.2fs (8 cameras, calibrations)", load_time)

        result = process_frame(sync_set, detector, extractor, matcher, reconstructor)
        gt_positions = load_ground_truth(frame_num)
        compare_to_ground_truth(result, gt_positions)

        results.append((result, gt_positions))

    # ── Final Report ──────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL REPORT")
    logger.info("Config: appearance=%.2f  epipolar=%.1fpx  reproj=%.1fpx  dbscan_eps=%.1fm",
                settings.fusion.appearance_threshold,
                settings.fusion.epipolar_threshold,
                settings.reconstruction.max_reprojection_error,
                settings.reconstruction.dbscan_eps)
    logger.info("=" * 60)
    logger.info("  %-6s %-8s %-10s %-12s %-10s %-10s",
                "Frame", "Dets", "Groups", "Persons(tri)", "PtsOK", "PtsRej")

    for result, _ in results:
        recon = result["reconstruction_result"]
        logger.info("  %-6d %-8d %-10d %-12s %-10d %-10d",
                    result["frame_num"],
                    sum(result["detection_counts"].values()),
                    result["fusion_result"].num_groups,
                    f"{recon.num_persons}({len(recon.triangulated_persons)})",
                    recon.num_triangulated_points,
                    recon.num_rejected_points)

    total_tri   = sum(len(r["reconstruction_result"].triangulated_persons) for r, _ in results)
    total_sv    = sum(len(r["reconstruction_result"].single_view_persons)  for r, _ in results)
    total_pts   = sum(r["reconstruction_result"].num_triangulated_points   for r, _ in results)
    total_rej   = sum(r["reconstruction_result"].num_rejected_points       for r, _ in results)

    logger.info("")
    logger.info("Totals across %d frames:", len(FRAMES_TO_TEST))
    logger.info("  Triangulated persons : %d", total_tri)
    logger.info("  Single-view persons  : %d", total_sv)
    logger.info("  Points accepted      : %d", total_pts)
    logger.info("  Points rejected      : %d", total_rej)
    if total_pts + total_rej > 0:
        logger.info("  Accept rate          : %.1f%%", 100.0 * total_pts / (total_pts + total_rej))


if __name__ == "__main__":
    run()
