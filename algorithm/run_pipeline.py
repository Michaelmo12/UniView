"""
Complete Pipeline Runner

Runs the full algorithm pipeline from ingestion through fusion (and future stages).

Usage:
    python run_pipeline.py --source <path_to_matrix_data> --num-frames <N>
    python run_pipeline.py --test-synthetic  # Run with synthetic data for testing

Pipeline stages:
1. Ingestion: Read frames from MATRIX dataset or generate synthetic
2. Detection: YOLO person detection
3. Features: WCH extraction
4. Fusion: Cross-camera matching
5. [Future] Reconstruction: 3D triangulation
6. [Future] Tracking: Temporal tracking
7. [Future] Output: WebSocket streaming
"""

import argparse
import logging
import queue
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add algorithm directory to path
algorithm_dir = Path(__file__).parent
if str(algorithm_dir) not in sys.path:
    sys.path.insert(0, str(algorithm_dir))

from config.settings import settings
from detection.batch_detector import BatchDetector
from features.wch_extractor import WCHExtractor
from fusion.cross_camera_matcher import CrossCameraMatcher
from ingestion.models import CameraCalibration, DroneFrame, SynchronizedFrameSet
from ingestion.tcp_receiver import TCPReceiver
from ingestion.synchronizer import FrameSynchronizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    Manages the complete algorithm pipeline execution.
    """

    def __init__(self):
        logger.info("Initializing pipeline stages...")

        # Stage 2: Detection
        self.detector = BatchDetector()
        logger.info(f"  ✓ Detection: YOLOv11 on {self.detector.detector.device}")

        # Stage 3: Features
        self.feature_extractor = WCHExtractor(settings.features)
        logger.info(
            f"  ✓ Features: WCH {settings.features.bins_per_channel} bins/channel"
        )

        # Stage 4: Fusion
        self.fusion_matcher = CrossCameraMatcher(settings.fusion)
        logger.info(
            f"  ✓ Fusion: epipolar<{settings.fusion.epipolar_threshold}px, "
            f"appearance>{settings.fusion.appearance_threshold}"
        )

        logger.info("Pipeline initialized successfully")

    def process_frame_set(self, sync_set: SynchronizedFrameSet):
        """
        Process a single synchronized frame set through all pipeline stages.

        Args:
            sync_set: SynchronizedFrameSet with frames from multiple drones

        Returns:
            dict with results from each stage
        """
        frame_num = sync_set.frame_num
        logger.info("=" * 80)
        logger.info(f"Processing Frame {frame_num}")
        logger.info("=" * 80)

        results = {"frame_num": frame_num, "timestamps": {}}

        # ===== STAGE 1: INGESTION =====
        logger.info(f"Stage 1: Ingestion - {sync_set.num_drones_present} drones")
        for drone_id in sorted(sync_set.frames.keys()):
            frame = sync_set.frames[drone_id]
            logger.info(
                f"  Drone {drone_id}: {frame.frame.shape} @ {frame.timestamp:.3f}s"
            )

        # ===== STAGE 2: DETECTION =====
        start = time.perf_counter()
        detection_sets = self.detector.process(sync_set)
        detection_time = time.perf_counter() - start

        total_detections = sum(len(ds.detections) for ds in detection_sets.values())
        logger.info(
            f"Stage 2: Detection - {total_detections} persons in {detection_time*1000:.1f}ms"
        )
        for drone_id, det_set in sorted(detection_sets.items()):
            logger.info(f"  Drone {drone_id}: {len(det_set.detections)} detections")

        results["detection"] = {
            "total_detections": total_detections,
            "time_ms": detection_time * 1000,
            "detections_by_drone": {
                drone_id: len(ds.detections) for drone_id, ds in detection_sets.items()
            },
        }

        # ===== STAGE 3: FEATURES =====
        start = time.perf_counter()
        features_dict = {}
        for drone_id, det_set in sorted(detection_sets.items()):
            frame = sync_set.frames[drone_id]
            frame_features = self.feature_extractor.extract_frame(frame, det_set)
            features_dict[drone_id] = frame_features.features

        feature_time = time.perf_counter() - start

        features_extracted = sum(
            sum(1 for d in ds.detections if d.features is not None)
            for ds in detection_sets.values()
        )
        logger.info(
            f"Stage 3: Features - {features_extracted}/{total_detections} features in {feature_time*1000:.1f}ms"
        )

        results["features"] = {
            "features_extracted": features_extracted,
            "time_ms": feature_time * 1000,
        }

        # ===== STAGE 4: FUSION =====
        start = time.perf_counter()

        # Extract projection matrices from calibration
        projection_matrices = {
            drone_id: sync_set.frames[drone_id].calibration.projection_matrix
            for drone_id in detection_sets.keys()
        }

        fusion_result = self.fusion_matcher.match_frame(
            detection_sets, projection_matrices, features_dict
        )
        fusion_time = time.perf_counter() - start

        logger.info(
            f"Stage 4: Fusion - {len(fusion_result.match_groups)} groups, "
            f"{fusion_result.total_matches} matches in {fusion_time*1000:.1f}ms"
        )

        for i, group in enumerate(fusion_result.match_groups):
            camera_ids = group.get_drone_ids()
            logger.info(
                f"  Group {i+1}: {group.num_cameras} cameras {camera_ids}, "
                f"{group.num_detections} detections, score={group.mean_appearance_score:.3f}"
            )

        results["fusion"] = {
            "match_groups": len(fusion_result.match_groups),
            "total_matches": fusion_result.total_matches,
            "total_detections": fusion_result.total_detections,
            "time_ms": fusion_time * 1000,
            "groups": [
                {
                    "cameras": g.get_drone_ids(),
                    "detections": g.num_detections,
                    "score": g.mean_appearance_score,
                }
                for g in fusion_result.match_groups
            ],
        }

        # ===== TOTAL PIPELINE TIME =====
        total_time = detection_time + feature_time + fusion_time
        logger.info(f"Total pipeline time: {total_time*1000:.1f}ms")
        logger.info(
            f"  Detection: {detection_time/total_time*100:.1f}% | "
            f"Features: {feature_time/total_time*100:.1f}% | "
            f"Fusion: {fusion_time/total_time*100:.1f}%"
        )

        results["total_time_ms"] = total_time * 1000

        return results


# ============================================================================
# SYNTHETIC DATA GENERATION (for testing without MATRIX dataset)
# ============================================================================


def generate_synthetic_frame_set(frame_num: int, num_cameras: int = 3) -> SynchronizedFrameSet:
    """
    Generate a synthetic multi-camera frame set for testing.

    Creates a scenario with multiple persons visible across different cameras.
    """
    logger.info(f"Generating synthetic frame {frame_num} with {num_cameras} cameras")

    frames = {}

    # Camera positions (spread along X axis)
    camera_positions = [(i * 2.0, 0.0, 0.0) for i in range(num_cameras)]

    # Person scenarios (color, cameras where visible, bbox per camera)
    scenarios = [
        {
            "color": (0, 0, 255),  # Red person
            "name": "Person A",
            "cameras": {
                0: (400, 300, 500, 550),
                1: (500, 320, 600, 570),
            },
        },
        {
            "color": (255, 0, 0),  # Blue person
            "name": "Person B",
            "cameras": {
                1: (900, 280, 1000, 530),
                2: (600, 300, 700, 550),
            },
        },
        {
            "color": (0, 255, 0),  # Green person (single-view)
            "name": "Person C",
            "cameras": {
                0: (700, 350, 800, 600),
            },
        },
    ]

    for cam_idx in range(num_cameras):
        drone_id = cam_idx + 1
        pos = camera_positions[cam_idx]

        # Create camera calibration
        K = np.array(
            [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        R = np.eye(3, dtype=np.float32)
        t = np.array([[pos[0]], [pos[1]], [pos[2]]], dtype=np.float32)
        dist = np.zeros(5, dtype=np.float32)

        calibration = CameraCalibration(K=K, R=R, t=t, dist=dist)

        # Create image
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        img[:] = (50, 50, 50)  # Gray background

        # Add persons visible in this camera
        for scenario in scenarios:
            if cam_idx in scenario["cameras"]:
                bbox = scenario["cameras"][cam_idx]
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), scenario["color"], -1)
                # Add some noise for realism
                noise = np.random.randint(-10, 10, img[y1:y2, x1:x2].shape, dtype=np.int16)
                img[y1:y2, x1:x2] = np.clip(img[y1:y2, x1:x2].astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Create DroneFrame
        frames[drone_id] = DroneFrame(
            drone_id=drone_id,
            frame_num=frame_num,
            timestamp=frame_num / 30.0,  # 30 FPS
            frame=img,
            calibration=calibration,
        )

        logger.info(f"  Camera {drone_id}: {len([s for s in scenarios if cam_idx in s['cameras']])} persons")

    return SynchronizedFrameSet(
        frame_num=frame_num,
        timestamp=frame_num / 30.0,
        frames=frames,
    )


# ============================================================================
# MAIN
# ============================================================================


def run_live_pipeline(num_frames: int, num_drones: int = 8):
    """
    Run pipeline with live TCP streaming from mock_drone_streamer.

    Args:
        num_frames: Number of synchronized frame sets to process
        num_drones: Number of drones to connect to (default 8)
    """
    logger.info("=" * 80)
    logger.info("LIVE TCP STREAMING PIPELINE TEST")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Connecting to {num_drones} drone streams...")
    logger.info(f"Base port: {settings.network.base_port}")
    logger.info(f"Ports: {settings.network.base_port} to {settings.network.base_port + num_drones - 1}")
    logger.info("")

    # Create queues for data flow
    raw_frame_queue = queue.Queue(maxsize=100)  # TCPReceiver -> Synchronizer
    sync_frame_queue = queue.Queue(maxsize=10)  # Synchronizer -> Pipeline

    # Create TCP receivers for each drone
    receivers = []
    for drone_id in range(1, num_drones + 1):
        receiver = TCPReceiver(drone_id, raw_frame_queue)
        receivers.append(receiver)

    # Create synchronizer
    synchronizer = FrameSynchronizer(raw_frame_queue, sync_frame_queue)

    # Start all components
    logger.info("Starting TCP receivers...")
    for receiver in receivers:
        receiver.start()

    logger.info("Starting frame synchronizer...")
    synchronizer.start()

    logger.info("")
    logger.info("Waiting for streams to connect (this may take a few seconds)...")
    time.sleep(3)  # Give time for TCP connections to establish

    # Initialize pipeline
    pipeline = PipelineRunner()
    logger.info("")

    # Process frames
    all_results = []
    frames_processed = 0

    try:
        logger.info(f"Processing {num_frames} synchronized frame sets...")
        logger.info("")

        while frames_processed < num_frames:
            try:
                # Get synchronized frame set (with timeout)
                sync_set = sync_frame_queue.get(timeout=10.0)

                # Process through pipeline
                results = pipeline.process_frame_set(sync_set)
                all_results.append(results)
                frames_processed += 1

                logger.info("")

            except queue.Empty:
                logger.warning("Timeout waiting for synchronized frame set")
                logger.warning("Make sure mock_drone_streamer is running!")
                break

    except KeyboardInterrupt:
        logger.info("")
        logger.info("Interrupted by user")

    finally:
        # Cleanup
        logger.info("")
        logger.info("Stopping receivers and synchronizer...")

        for receiver in receivers:
            receiver.stop()

        synchronizer.stop()

        logger.info("Cleanup complete")
        logger.info("")

    # Summary
    if all_results:
        logger.info("=" * 80)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 80)

        total_detections = sum(r["detection"]["total_detections"] for r in all_results)
        total_features = sum(r["features"]["features_extracted"] for r in all_results)
        total_groups = sum(r["fusion"]["match_groups"] for r in all_results)
        avg_time = np.mean([r["total_time_ms"] for r in all_results])

        logger.info(f"Frames processed: {len(all_results)}")
        logger.info(f"Total detections: {total_detections}")
        logger.info(f"Total features extracted: {total_features}")
        logger.info(f"Total match groups: {total_groups}")
        logger.info(f"Average pipeline time: {avg_time:.1f}ms/frame")
        logger.info("")
        logger.info("✓ Pipeline integration successful!")
    else:
        logger.error("No frames processed - check that mock_drone_streamer is running")
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(description="Run complete algorithm pipeline")
    parser.add_argument(
        "--test-synthetic",
        action="store_true",
        help="Run with synthetic test data (for quick testing without streamer)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=5,
        help="Number of frames to process",
    )
    parser.add_argument(
        "--num-drones",
        type=int,
        default=8,
        help="Number of drones to connect to (default 8)",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("ALGORITHM PIPELINE RUNNER")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Pipeline stages:")
    logger.info("  1. Ingestion → SynchronizedFrameSet")
    logger.info("  2. Detection → YOLO person detection")
    logger.info("  3. Features → WCH extraction")
    logger.info("  4. Fusion → Cross-camera matching")
    logger.info("  5. [Future] Reconstruction → 3D triangulation")
    logger.info("  6. [Future] Tracking → Temporal tracking")
    logger.info("  7. [Future] Output → WebSocket streaming")
    logger.info("")

    if args.test_synthetic:
        # Initialize pipeline for synthetic test
        pipeline = PipelineRunner()
        logger.info("")

        # Process synthetic frames
        all_results = []

        for frame_idx in range(args.num_frames):
            sync_set = generate_synthetic_frame_set(frame_num=frame_idx, num_cameras=3)
            results = pipeline.process_frame_set(sync_set)
            all_results.append(results)
            logger.info("")

        # Summary
        logger.info("=" * 80)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 80)

        total_detections = sum(r["detection"]["total_detections"] for r in all_results)
        total_features = sum(r["features"]["features_extracted"] for r in all_results)
        total_groups = sum(r["fusion"]["match_groups"] for r in all_results)
        avg_time = np.mean([r["total_time_ms"] for r in all_results])

        logger.info(f"Frames processed: {len(all_results)}")
        logger.info(f"Total detections: {total_detections}")
        logger.info(f"Total features extracted: {total_features}")
        logger.info(f"Total match groups: {total_groups}")
        logger.info(f"Average pipeline time: {avg_time:.1f}ms/frame")
        logger.info("")
        logger.info("✓ Pipeline integration successful!")

        return 0
    else:
        # Run with live TCP streaming
        return run_live_pipeline(args.num_frames, args.num_drones)


if __name__ == "__main__":
    exit(main())
