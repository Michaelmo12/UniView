"""
Cross-Camera Matcher Orchestrator

Main fusion pipeline that processes all camera pairs to identify unique persons.

Process:
1. Compute fundamental matrices for all camera pairs
2. For each pair: geometric filtering (epipolar) -> appearance verification (WCH)
3. Merge pairwise matches into connected components (match groups)
4. Identify unmatched detections

White-box: Graph-based connected components via BFS for transitive closure.
"""

import logging
import time
import itertools
import numpy as np
from collections import defaultdict

from algorithm.fusion.models import CrossCameraMatch, MatchGroup, FusionResult
from algorithm.fusion.fundamental_matrix import compute_fundamental_matrix
from algorithm.fusion.epipolar_filter import compute_epipolar_distance
from algorithm.fusion.appearance_matcher import AppearanceMatcher
from algorithm.detection.models import Detection, DetectionSet
from algorithm.features.models import PersonFeatures
from algorithm.config.settings import FusionConfig

logger = logging.getLogger(__name__)


class CrossCameraMatcher:
    """
    Main orchestrator for cross-camera fusion.

    Processes all camera pairs to:
    - Filter by epipolar geometry
    - Verify by appearance (WCH similarity)
    - Merge into consistent match groups
    - Identify unmatched detections
    """

    def __init__(self, config: FusionConfig):
        """
        Initialize cross-camera matcher.

        Args:
            config: FusionConfig with epipolar and appearance thresholds
        """
        self.config = config
        self.appearance_matcher = AppearanceMatcher(config)

    def match_frame(
        self,
        detection_sets: dict[int, DetectionSet],
        projection_matrices: dict[int, np.ndarray],
        features_dict: dict[int, list[PersonFeatures]],
    ) -> FusionResult:
        """
        Match detections across all cameras for one synchronized frame set.

        Args:
            detection_sets: {drone_id: DetectionSet} for all cameras
            projection_matrices: {drone_id: P_matrix (3, 4)} for all cameras
            features_dict: {drone_id: [PersonFeatures]} extracted features per camera

        Returns:
            FusionResult with match groups and unmatched detections
        """
        start_time = time.perf_counter()

        # Get frame number
        frame_num = self._get_frame_num(detection_sets)

        # Get all drone IDs
        drone_ids = sorted(detection_sets.keys())
        num_drones = len(drone_ids)

        if num_drones < 2:
            logger.warning(
                f"Frame {frame_num}: Only {num_drones} cameras, cannot perform cross-camera matching"
            )
            # Return all detections as unmatched
            all_detections = []
            for det_set in detection_sets.values():
                all_detections.extend(det_set.detections)

            elapsed = time.perf_counter() - start_time
            return FusionResult(
                frame_num=frame_num,
                match_groups=[],
                total_detections=len(all_detections),
                total_matches=0,
            )

        # Count total detections
        total_detections = sum(
            det_set.num_detections for det_set in detection_sets.values()
        )

        logger.debug(
            f"Frame {frame_num}: Matching {total_detections} detections "
            f"across {num_drones} cameras"
        )

        # Process all camera pairs
        pairwise_matches = {}
        num_pairs = 0

        for drone_id_a, drone_id_b in itertools.combinations(drone_ids, 2):
            num_pairs += 1

            det_set_a = detection_sets[drone_id_a]
            det_set_b = detection_sets[drone_id_b]

            # Skip if either camera has no detections
            if det_set_a.is_empty or det_set_b.is_empty:
                logger.debug(
                    f"  Pair ({drone_id_a}, {drone_id_b}): skipped (empty detection set)"
                )
                continue

            # Get projection matrices
            P_a = projection_matrices[drone_id_a]
            P_b = projection_matrices[drone_id_b]

            # Get features
            features_a = features_dict.get(drone_id_a, [])
            features_b = features_dict.get(drone_id_b, [])

            # Match this pair
            matches = self._match_pair(
                drone_id_a,
                drone_id_b,
                det_set_a,
                det_set_b,
                P_a,
                P_b,
                features_a,
                features_b,
            )

            if len(matches) > 0:
                pairwise_matches[(drone_id_a, drone_id_b)] = matches
                logger.debug(
                    f"  Pair ({drone_id_a}, {drone_id_b}): {len(matches)} matches"
                )
            else:
                logger.debug(f"  Pair ({drone_id_a}, {drone_id_b}): no matches")

        # Merge pairwise matches into groups
        match_groups = self._merge_to_groups(pairwise_matches)

        # Compute total matches
        total_matches = sum(len(matches) for matches in pairwise_matches.values())

        elapsed = time.perf_counter() - start_time

        logger.info(
            f"Fusion complete: {len(match_groups)} match groups, "
            f"{total_matches} pairwise matches, {num_pairs} pairs processed "
            f"in {elapsed * 1000:.1f}ms"
        )

        return FusionResult(
            frame_num=frame_num,
            match_groups=match_groups,
            total_detections=total_detections,
            total_matches=total_matches,
        )

    def _match_pair(
        self,
        drone_id_a: int,
        drone_id_b: int,
        det_set_a: DetectionSet,
        det_set_b: DetectionSet,
        P_a: np.ndarray,
        P_b: np.ndarray,
        features_a: list[PersonFeatures],
        features_b: list[PersonFeatures],
    ) -> list[CrossCameraMatch]:
        """
        Match detections between two cameras.

        Process:
        1. Compute fundamental matrix
        2. Geometric filtering (epipolar constraint)
        3. Appearance verification (WCH similarity + Hungarian)

        Args:
            drone_id_a: First camera ID
            drone_id_b: Second camera ID
            det_set_a: Detections from camera A
            det_set_b: Detections from camera B
            P_a: Projection matrix for camera A
            P_b: Projection matrix for camera B
            features_a: Extracted features for camera A
            features_b: Extracted features for camera B

        Returns:
            List of CrossCameraMatch objects for this pair
        """
        # Compute fundamental matrix
        F = compute_fundamental_matrix(P_a, P_b)

        # Geometric filtering: check all pairs for epipolar constraint
        geometric_candidates = []

        for i, feat_a in enumerate(features_a):
            for j, feat_b in enumerate(features_b):
                # Compute epipolar distance
                epi_dist = compute_epipolar_distance(feat_a, feat_b, F)

                # Check threshold
                if epi_dist <= self.config.epipolar_threshold:
                    geometric_candidates.append((i, j, float(epi_dist)))

        logger.debug(
            f"    Epipolar filter: {len(geometric_candidates)}/"
            f"{len(features_a) * len(features_b)} candidates "
            f"(threshold={self.config.epipolar_threshold:.1f}px)"
        )

        # Early exit if no geometric candidates
        if len(geometric_candidates) == 0:
            return []

        # Appearance verification
        detections_a = det_set_a.detections
        detections_b = det_set_b.detections

        confirmed = self.appearance_matcher.verify_candidates(
            detections_a, detections_b, geometric_candidates
        )

        logger.debug(
            f"    Appearance filter: {len(confirmed)}/{len(geometric_candidates)} "
            f"confirmed (threshold={self.config.appearance_threshold:.2f})"
        )

        # Build lookup dict for epipolar distances
        candidate_distances = {
            (idx_a, idx_b): distance for idx_a, idx_b, distance in geometric_candidates
        }

        # Create CrossCameraMatch objects
        matches = []
        for idx_a, idx_b, similarity in confirmed:
            # Get epipolar distance from candidates
            epipolar_distance = candidate_distances[(idx_a, idx_b)]

            # Get detections
            det_a = detections_a[idx_a]
            det_b = detections_b[idx_b]

            match = CrossCameraMatch(
                drone_id_a=drone_id_a,
                drone_id_b=drone_id_b,
                local_id_a=det_a.local_id,
                local_id_b=det_b.local_id,
                epipolar_distance=epipolar_distance,
                appearance_score=similarity,
            )
            matches.append(match)

        return matches

    def _merge_to_groups(
        self, pairwise_matches: dict[tuple[int, int], list[CrossCameraMatch]]
    ) -> list[MatchGroup]:
        """
        Merge pairwise matches into consistent match groups.

        Uses connected components (BFS) to find transitive closure:
        If det_a matches det_b AND det_b matches det_c, then all three
        are in the same group.

        Args:
            pairwise_matches: {(drone_id_a, drone_id_b): [CrossCameraMatch]}

        Returns:
            List of MatchGroup objects
        """
        # Build adjacency graph: {(drone_id, local_id): set of neighbors}
        graph = defaultdict(set)
        match_dict = {}  # {((drone_a, id_a), (drone_b, id_b)): CrossCameraMatch}

        for matches in pairwise_matches.values():
            for match in matches:
                node_a = (match.drone_id_a, match.local_id_a)
                node_b = (match.drone_id_b, match.local_id_b)

                graph[node_a].add(node_b)
                graph[node_b].add(node_a)

                # Store match for later retrieval
                edge = (min(node_a, node_b), max(node_a, node_b))
                match_dict[edge] = match

        # Find connected components via BFS
        visited = set()
        groups = []

        for start_node in graph.keys():
            if start_node in visited:
                continue

            # BFS to find all nodes in this component
            component = set()
            queue = [start_node]
            component.add(start_node)
            visited.add(start_node)

            while queue:
                node = queue.pop(0)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)

            # Build MatchGroup from component
            detections = sorted(component)  # List of (drone_id, local_id) tuples

            # Compute mean appearance score from all pairwise matches in this group
            scores = []
            for i in range(len(detections)):
                for j in range(i + 1, len(detections)):
                    edge = (
                        min(detections[i], detections[j]),
                        max(detections[i], detections[j]),
                    )
                    if edge in match_dict:
                        scores.append(match_dict[edge].appearance_score)

            mean_score = np.mean(scores) if len(scores) > 0 else 0.0

            group = MatchGroup(
                detections=detections, mean_appearance_score=float(mean_score)
            )
            groups.append(group)

        logger.debug(
            f"  Merged {sum(len(m) for m in pairwise_matches.values())} "
            f"pairwise matches into {len(groups)} groups"
        )

        return groups

    def _get_frame_num(self, detection_sets: dict[int, DetectionSet]) -> int:
        """
        Get frame number from detection sets.

        Args:
            detection_sets: {drone_id: DetectionSet}

        Returns:
            Frame number from first available detection set
        """
        for det_set in detection_sets.values():
            return det_set.frame_num
        return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Testing CrossCameraMatcher")
    logger.info("=" * 60)

    from detection.models import BoundingBox
    from config.settings import settings

    # Create synthetic cameras
    K = np.array(
        [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )

    # Camera 1: at origin
    R1 = np.eye(3, dtype=np.float64)
    t1 = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
    P1 = K @ np.hstack([R1, t1])

    # Camera 2: translated 2m in X
    R2 = np.eye(3, dtype=np.float64)
    t2 = np.array([[2.0], [0.0], [0.0]], dtype=np.float64)
    P2 = K @ np.hstack([R2, t2])

    # Camera 3: translated 2m in X, 2m in Y
    R3 = np.eye(3, dtype=np.float64)
    t3 = np.array([[2.0], [2.0], [0.0]], dtype=np.float64)
    P3 = K @ np.hstack([R3, t3])

    projection_matrices = {1: P1, 2: P2, 3: P3}

    # Create synthetic 3D persons
    # Person A: at (5, 3, 10) - visible in cameras 1 and 2
    X_person_a = np.array([5.0, 3.0, 10.0, 1.0], dtype=np.float64)

    # Person B: at (7, 5, 12) - visible in cameras 2 and 3
    X_person_b = np.array([7.0, 5.0, 12.0, 1.0], dtype=np.float64)

    # Person C: at (3, 2, 8) - visible only in camera 1
    X_person_c = np.array([3.0, 2.0, 8.0, 1.0], dtype=np.float64)

    # Project to image coordinates
    def project_point(X, P):
        x = P @ X
        x = x / x[2]  # Normalize
        return (float(x[0]), float(x[1]))

    # Person A projections
    a_cam1 = project_point(X_person_a, P1)
    a_cam2 = project_point(X_person_a, P2)

    # Person B projections
    b_cam2 = project_point(X_person_b, P2)
    b_cam3 = project_point(X_person_b, P3)

    # Person C projection
    c_cam1 = project_point(X_person_c, P1)

    logger.info("\nProjected positions:")
    logger.info(f"  Person A: cam1={a_cam1}, cam2={a_cam2}")
    logger.info(f"  Person B: cam2={b_cam2}, cam3={b_cam3}")
    logger.info(f"  Person C: cam1={c_cam1}")

    # Create WCH features
    np.random.seed(42)
    wch_a = np.random.randn(96).astype(np.float32)
    wch_a /= np.linalg.norm(wch_a)

    wch_b = np.random.randn(96).astype(np.float32)
    wch_b /= np.linalg.norm(wch_b)

    wch_c = np.random.randn(96).astype(np.float32)
    wch_c /= np.linalg.norm(wch_c)

    # Helper to create detection with features
    def make_detection(drone_id, local_id, center, wch, frame_num=0):
        cx, cy = center
        bbox = BoundingBox(x1=cx - 30, y1=cy - 60, x2=cx + 30, y2=cy + 60)
        det = Detection(
            bbox=bbox,
            class_id=0,
            confidence=0.9,
            drone_id=drone_id,
            frame_num=frame_num,
            local_id=local_id,
        )
        # Add small noise to WCH to simulate real variation
        noisy_wch = wch.copy() + np.random.randn(96).astype(np.float32) * 0.02
        noisy_wch /= np.linalg.norm(noisy_wch)
        det.features = noisy_wch
        return det

    def make_person_features(drone_id, local_id, center, wch, P, frame_num=0):
        cx, cy = center
        return PersonFeatures(
            drone_id=drone_id,
            frame_num=frame_num,
            local_id=local_id,
            wch=wch,
            bbox_center=center,
            projection_matrix=P,
            confidence=0.9,
        )

    # Camera 1: Person A (local_id=0), Person C (local_id=1)
    dets_cam1 = [
        make_detection(1, 0, a_cam1, wch_a),
        make_detection(1, 1, c_cam1, wch_c),
    ]
    features_cam1 = [
        make_person_features(1, 0, a_cam1, dets_cam1[0].features, P1),
        make_person_features(1, 1, c_cam1, dets_cam1[1].features, P1),
    ]

    # Camera 2: Person A (local_id=0), Person B (local_id=1)
    dets_cam2 = [
        make_detection(2, 0, a_cam2, wch_a),
        make_detection(2, 1, b_cam2, wch_b),
    ]
    features_cam2 = [
        make_person_features(2, 0, a_cam2, dets_cam2[0].features, P2),
        make_person_features(2, 1, b_cam2, dets_cam2[1].features, P2),
    ]

    # Camera 3: Person B (local_id=0)
    dets_cam3 = [
        make_detection(3, 0, b_cam3, wch_b),
    ]
    features_cam3 = [
        make_person_features(3, 0, b_cam3, dets_cam3[0].features, P3),
    ]

    # Create detection sets
    detection_sets = {
        1: DetectionSet(drone_id=1, frame_num=0, detections=dets_cam1),
        2: DetectionSet(drone_id=2, frame_num=0, detections=dets_cam2),
        3: DetectionSet(drone_id=3, frame_num=0, detections=dets_cam3),
    }

    features_dict = {
        1: features_cam1,
        2: features_cam2,
        3: features_cam3,
    }

    logger.info("\nDetection sets:")
    for drone_id, det_set in detection_sets.items():
        logger.info(f"  Camera {drone_id}: {det_set.num_detections} detections")

    # Run fusion
    matcher = CrossCameraMatcher(settings.fusion)
    result = matcher.match_frame(detection_sets, projection_matrices, features_dict)

    logger.info("\nFusion result:")
    logger.info(f"  Frame: {result.frame_num}")
    logger.info(f"  Match groups: {result.num_groups}")
    logger.info(f"  Total detections: {result.total_detections}")
    logger.info(f"  Total matches: {result.total_matches}")

    logger.info("\nMatch groups:")
    for i, group in enumerate(result.match_groups):
        logger.info(f"  Group {i + 1}: {group}")
        logger.info(f"    Cameras: {group.get_drone_ids()}")
        logger.info(f"    Detections: {group.detections}")

    # Validation checks
    logger.info("\nValidation:")

    # Check 1: Should have 2 groups (Person A and Person B)
    # Person C is unmatched, so might be in a singleton group or not in groups at all
    assert (
        result.num_groups >= 2
    ), f"Expected at least 2 groups, got {result.num_groups}"
    logger.info(f"  ✓ PASS: {result.num_groups} match groups found")

    # Check 2: Person A should be matched across cameras 1-2
    person_a_group = None
    for group in result.match_groups:
        if (1, 0) in group.detections and (2, 0) in group.detections:
            person_a_group = group
            break

    assert person_a_group is not None, "Person A not matched across cameras 1-2"
    logger.info("  ✓ PASS: Person A matched across cameras 1-2")

    # Check 3: Person B should be matched across cameras 2-3
    person_b_group = None
    for group in result.match_groups:
        if (2, 1) in group.detections and (3, 0) in group.detections:
            person_b_group = group
            break

    assert person_b_group is not None, "Person B not matched across cameras 2-3"
    logger.info("  ✓ PASS: Person B matched across cameras 2-3")

    # Check 4: Person C should NOT be in a multi-camera group
    person_c_in_multigroup = False
    for group in result.match_groups:
        if (1, 1) in group.detections and group.num_cameras > 1:
            person_c_in_multigroup = True
            break

    assert not person_c_in_multigroup, "Person C should not be in a multi-camera group"
    logger.info("  ✓ PASS: Person C not in multi-camera groups")

    # Check 5: No detection appears in multiple groups
    all_detections_in_groups = []
    for group in result.match_groups:
        all_detections_in_groups.extend(group.detections)

    unique_detections = set(all_detections_in_groups)
    assert len(all_detections_in_groups) == len(
        unique_detections
    ), "Some detections appear in multiple groups!"
    logger.info("  ✓ PASS: No detection appears in multiple groups")

    logger.info("\n" + "=" * 60)
    logger.info("All tests passed!")
