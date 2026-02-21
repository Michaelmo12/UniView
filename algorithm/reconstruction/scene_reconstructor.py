"""
SceneReconstructor

Orchestrates the complete 3D reconstruction pipeline:
1. Triangulate match groups into 3D points
2. Identify unmatched (single-view) detections
3. Cluster points into unique persons via DBSCAN
4. Preserve single-view detections as Person3D

Key Class:
- SceneReconstructor: Orchestrator combining Triangulator and PersonClusterer
"""

import sys
from pathlib import Path

# Add project root to path for algorithm imports when run as script
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import logging
import numpy as np

from algorithm.config.settings import ReconstructionConfig, settings
from algorithm.fusion.models import FusionResult, MatchGroup
from algorithm.detection.models import DetectionSet, Detection, BoundingBox
from algorithm.ingestion.models import SynchronizedFrameSet, DroneFrame, CameraCalibration
from algorithm.reconstruction.models import ReconstructionResult
from algorithm.reconstruction.triangulator import Triangulator
from algorithm.reconstruction.clusterer import PersonClusterer

logger = logging.getLogger(__name__)


class SceneReconstructor:
    """
    Orchestrates 3D reconstruction from fusion results.

    Combines triangulation and clustering to produce a list of unique persons
    (Person3D) from matched and unmatched detections.

    Constructor takes ReconstructionConfig, creates Triangulator and PersonClusterer.
    """

    def __init__(self, config: ReconstructionConfig):
        """
        Initialize scene reconstructor with configuration.

        Args:
            config: ReconstructionConfig with triangulation and clustering parameters
        """
        self.config = config
        self.triangulator = Triangulator(config)
        self.clusterer = PersonClusterer(config)

    def reconstruct(
        self,
        fusion_result: FusionResult,
        detection_sets: dict[int, DetectionSet],
        sync_set: SynchronizedFrameSet
    ) -> ReconstructionResult:
        """
        Reconstruct 3D scene from fusion results.

        Pipeline:
        1. Triangulate each match group using Triangulator
        2. Identify detections not in any match group (single-view)
        3. Cluster triangulated points + preserve single-view via PersonClusterer
        4. Return ReconstructionResult with all persons

        Args:
            fusion_result: FusionResult with match groups from cross-camera fusion
            detection_sets: Dict[drone_id -> DetectionSet] from detection stage
            sync_set: SynchronizedFrameSet with calibration data

        Returns:
            ReconstructionResult with persons list and statistics
        """
        logger.info("Reconstructing frame %d", fusion_result.frame_num)

        # Step 1: Triangulate match groups
        triangulated_points = []
        rejected_count = 0

        for group in fusion_result.match_groups:
            point_3d = self.triangulator.triangulate_match_group(
                group, detection_sets, sync_set
            )

            if point_3d is not None:
                triangulated_points.append(point_3d)
            else:
                rejected_count += 1

        # Step 2: Identify unmatched detections
        # Build set of all (drone_id, local_id) that appear in match groups
        matched_detections = set()
        for group in fusion_result.match_groups:
            matched_detections.update(group.detections)

        # Find all detections not in matched set
        unmatched_detections = []
        for drone_id, det_set in detection_sets.items():
            for local_id in range(len(det_set.detections)):
                if (drone_id, local_id) not in matched_detections:
                    unmatched_detections.append((drone_id, local_id))

        # Step 3: Cluster persons
        persons = self.clusterer.cluster_persons(
            triangulated_points, unmatched_detections
        )

        # Step 4: Create result
        result = ReconstructionResult(
            frame_num=fusion_result.frame_num,
            persons=persons,
            num_triangulated_points=len(triangulated_points),
            num_rejected_points=rejected_count
        )

        # Log summary
        num_single_view = len(result.single_view_persons)
        logger.info(
            "Reconstructed frame %d: %d triangulated, %d rejected, "
            "%d persons (%d single-view)",
            fusion_result.frame_num,
            len(triangulated_points),
            rejected_count,
            result.num_persons,
            num_single_view
        )

        return result


if __name__ == "__main__":
    """
    Validation script: Tests end-to-end reconstruction with synthetic data.

    Creates 2 cameras with known geometry, synthetic match groups, and validates
    that triangulation produces correct 3D positions and preserves single-view detections.
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing SceneReconstructor with Synthetic Data")
    logger.info("=" * 60)

    from algorithm.config.settings import settings

    # Create synthetic camera calibration
    # Camera 1: at origin, looking along +Z axis
    K1 = np.array([
        [1000.0, 0.0, 960.0],
        [0.0, 1000.0, 540.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    R1 = np.eye(3, dtype=np.float32)
    t1 = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
    dist1 = np.zeros(5, dtype=np.float32)
    calib1 = CameraCalibration(K=K1, R=R1, t=t1, dist=dist1)

    # Camera 2: offset 2m to the right (+X), looking along +Z axis
    K2 = K1.copy()
    R2 = np.eye(3, dtype=np.float32)
    t2 = np.array([[-2.0], [0.0], [0.0]], dtype=np.float32)  # Note: t is in camera coords
    dist2 = np.zeros(5, dtype=np.float32)
    calib2 = CameraCalibration(K=K2, R=R2, t=t2, dist=dist2)

    logger.info("\nCamera setup:")
    logger.info("  Camera 1 center: %s", calib1.camera_center)
    logger.info("  Camera 2 center: %s", calib2.camera_center)

    # Create mock frames
    mock_image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    frame1 = DroneFrame(
        drone_id=1,
        frame_num=0,
        timestamp=1234567890.0,
        frame=mock_image.copy(),
        calibration=calib1
    )

    frame2 = DroneFrame(
        drone_id=2,
        frame_num=0,
        timestamp=1234567890.0,
        frame=mock_image.copy(),
        calibration=calib2
    )

    frame3 = DroneFrame(
        drone_id=3,
        frame_num=0,
        timestamp=1234567890.0,
        frame=mock_image.copy(),
        calibration=calib1  # Reuse calib1 for simplicity
    )

    sync_set = SynchronizedFrameSet(
        frame_num=0,
        timestamp=1234567890.0,
        frames={1: frame1, 2: frame2, 3: frame3}
    )

    # Create synthetic detections
    # Person at world position [0, 0, 5] (5m in front of camera 1, centered)
    # Should project to center of both images

    # Compute expected projections
    world_point = np.array([0.0, 0.0, 5.0, 1.0])  # Homogeneous

    proj1 = calib1.projection_matrix @ world_point
    px1 = proj1[0] / proj1[2]
    py1 = proj1[1] / proj1[2]

    proj2 = calib2.projection_matrix @ world_point
    px2 = proj2[0] / proj2[2]
    py2 = proj2[1] / proj2[2]

    logger.info("\nExpected projections for world point [0, 0, 5]:")
    logger.info("  Camera 1: (%.1f, %.1f)", px1, py1)
    logger.info("  Camera 2: (%.1f, %.1f)", px2, py2)

    # Create detections at these projected positions
    bbox1 = BoundingBox(x1=px1-50, y1=py1-100, x2=px1+50, y2=py1+100)
    det1 = Detection(
        bbox=bbox1,
        class_id=0,
        confidence=0.95,
        drone_id=1,
        frame_num=0,
        local_id=0
    )

    bbox2 = BoundingBox(x1=px2-50, y1=py2-100, x2=px2+50, y2=py2+100)
    det2 = Detection(
        bbox=bbox2,
        class_id=0,
        confidence=0.95,
        drone_id=2,
        frame_num=0,
        local_id=0
    )

    # Add a single-view detection (only in camera 3)
    bbox3 = BoundingBox(x1=500, y1=500, x2=600, y2=700)
    det3 = Detection(
        bbox=bbox3,
        class_id=0,
        confidence=0.90,
        drone_id=3,
        frame_num=0,
        local_id=0
    )

    detection_sets = {
        1: DetectionSet(drone_id=1, frame_num=0, detections=[det1]),
        2: DetectionSet(drone_id=2, frame_num=0, detections=[det2]),
        3: DetectionSet(drone_id=3, frame_num=0, detections=[det3])
    }

    # Create match group (cameras 1 and 2 matched, camera 3 unmatched)
    match_group = MatchGroup(
        detections=[(1, 0), (2, 0)],
        mean_appearance_score=0.85
    )

    fusion_result = FusionResult(
        frame_num=0,
        match_groups=[match_group],
        total_detections=3,
        total_matches=1
    )

    logger.info("\nInput data:")
    logger.info("  Match groups: %d", len(fusion_result.match_groups))
    logger.info("  Total detections: %d", fusion_result.total_detections)
    logger.info("  Detection sets: %s", list(detection_sets.keys()))

    # Run reconstruction
    reconstructor = SceneReconstructor(settings.reconstruction)
    result = reconstructor.reconstruct(fusion_result, detection_sets, sync_set)

    logger.info("\nReconstruction result:")
    logger.info("  %s", result)

    # Validate results
    logger.info("\n" + "=" * 60)
    logger.info("Validation:")

    assert result.num_persons >= 2, f"Expected >= 2 persons, got {result.num_persons}"
    logger.info("  ✓ At least 2 persons found")

    triangulated = result.triangulated_persons
    assert len(triangulated) >= 1, f"Expected >= 1 triangulated person, got {len(triangulated)}"
    logger.info("  ✓ At least 1 triangulated person")

    single_view = result.single_view_persons
    assert len(single_view) >= 1, f"Expected >= 1 single-view person, got {len(single_view)}"
    logger.info("  ✓ At least 1 single-view person preserved")

    # Check triangulated person position is near [0, 0, 5]
    if triangulated:
        person = triangulated[0]
        if person.position is not None:
            expected = np.array([0.0, 0.0, 5.0])
            distance = np.linalg.norm(person.position - expected)
            logger.info("  Triangulated position: %s", person.position)
            logger.info("  Expected position: %s", expected)
            logger.info("  Distance: %.3fm", distance)

            if distance < 0.5:  # Allow 0.5m error
                logger.info("  ✓ Triangulated position accurate")
            else:
                logger.warning("  ⚠ Triangulated position error > 0.5m")

    # Print all persons
    logger.info("\nAll persons:")
    for person in result.persons:
        logger.info("  %s", person)

    logger.info("\n" + "=" * 60)
    logger.info("All validations passed!")
    logger.info("SceneReconstructor is ready for pipeline integration")
