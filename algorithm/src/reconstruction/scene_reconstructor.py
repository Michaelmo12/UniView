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

import logging

from src.config.settings import ReconstructionConfig
from src.fusion.models import FusionResult
from src.detection.models import DetectionSet
from src.ingestion.models import SynchronizedFrameSet
from src.reconstruction.models import ReconstructionResult
from src.reconstruction.triangulator import Triangulator
from src.reconstruction.clusterer import PersonClusterer

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

        # Step 1: Triangulate match groups — collect all raw pairwise Point3Ds
        triangulated_points = []
        # how many groups were fully rejected (no valid Point3Ds)
        rejected_count = 0
        # detections from groups where triangulation completely failed — treated as single-view
        fallback_detections: list[tuple[int, int]] = []

        # for each match group — triangulate and collect Point3Ds or mark as fallback
        for group in fusion_result.match_groups:
            raw_points, used_fallback = self.triangulator.triangulate_match_group_robust(
                group, detection_sets, sync_set
            )

            # add all valid Point3Ds from this group
            if raw_points:
                triangulated_points.extend(raw_points)
            # no valid points
            else:
                rejected_count += 1
                for det_id in group.detections:
                    fallback_detections.append(det_id)

        # Step 2: Identify unmatched detections
        # a list of (drone_id, local_id) tuples for every detection that has no valid 3D position
        
        # Build set of all (drone_id, local_id) that appear in match groups
        matched_detections = set()
        for group in fusion_result.match_groups:
            matched_detections.update(group.detections)
            
        # Detections from failed robust groups are also treated as unmatched
        for det_id in fallback_detections:
            matched_detections.discard(det_id)

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

