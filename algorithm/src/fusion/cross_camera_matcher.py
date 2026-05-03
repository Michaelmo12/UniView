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
from collections import defaultdict, deque

from src.fusion.models import CrossCameraMatch, MatchGroup, FusionResult
from src.fusion.fundamental_matrix import compute_fundamental_matrix
from src.fusion.epipolar_filter import compute_epipolar_distance
from src.fusion.appearance_matcher import AppearanceMatcher
from src.detection.models import Detection, DetectionSet
from src.features.models import PersonFeatures
from src.config.settings import FusionConfig

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
        detection_sets: dict[int, DetectionSet], # {drone_id: DetectionSet} for all cameras
        projection_matrices: dict[int, np.ndarray], # {drone_id: P_matrix (3, 4)} for all cameras
        features_dict: dict[int, list[PersonFeatures]], # {drone_id: [PersonFeatures]} extracted features per camera
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
        # for logging
        start_time = time.perf_counter()

        # Get frame number
        frame_num = self._get_frame_num(detection_sets)

        # Get all drone IDs
        drone_ids = sorted(detection_sets.keys())
        num_drones = len(drone_ids)

        # less than 2 drones cannot compute cross_camera_matcher
        if num_drones < 2:
            logger.warning(
                f"Frame {frame_num}: Only {num_drones} cameras, cannot perform cross-camera matching"
            )
            
            # Return all detections as unmatched
            all_detections = []
            
            # 
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
        total_detections = 0
        for det_set in detection_sets.values():
            total_detections += det_set.num_detections

        logger.debug(
            f"Frame {frame_num}: Matching {total_detections} detections "
            f"across {num_drones} cameras"
        )

        # Process all camera pairs
        pairwise_matches = {}
        num_pairs = 0
        
        # for each unique pair of drones
        for drone_id_a, drone_id_b in itertools.combinations(drone_ids, 2):
            num_pairs += 1

            # get detection set from drone a and b
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

            # Match this pair of drones and receive List of CrossCameraMatch objects for this pair
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

            # store results keyed by the drone pair tuple
            if len(matches) > 0:
                pairwise_matches[(drone_id_a, drone_id_b)] = matches
                logger.debug(
                    f"  Pair ({drone_id_a}, {drone_id_b}): {len(matches)} matches"
                )
            # no matches found so we dont store anything
            else:
                logger.debug(f"  Pair ({drone_id_a}, {drone_id_b}): no matches")

        # after all pairs are processed -> Merge pairwise matches into groups BFS
        match_groups = self._merge_to_groups(pairwise_matches)

        # Compute total matches
        total_matches = 0
        for matches in pairwise_matches.values():
            total_matches += len(matches)

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

        # ======= EPIPOLAR FILTER WORKS ON FEATURES LIST AND verify_candidates WORKS ON DETS =======
        
        # Every possible pair of detections checked against epipolar
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

        # CONVERT FROM FEATURES LIST TO DETECTIONS LIST
        # (REMINDER FEATURES MIGHT NOT HAVE ALL DETECTIONS BECAUSE OF SMALL CROPS)
        # fast lookup: local_id → Detection object
        det_by_local_id_a = {}
        for det in det_set_a.detections:
            det_by_local_id_a[det.local_id] = det

        det_by_local_id_b = {}
        for det in det_set_b.detections:
            det_by_local_id_b[det.local_id] = det

        # map feature index → Detection in same order as features_a so indices stay consistent
        feature_to_det_a = {}
        for i, feat in enumerate(features_a):
            feature_to_det_a[i] = det_by_local_id_a[feat.local_id]

        feature_to_det_b = {}
        for i, feat in enumerate(features_b):
            feature_to_det_b[i] = det_by_local_id_b[feat.local_id]

        # Appearance verification operates on detections aligned to features_a/b order
        detections_a_aligned = list(feature_to_det_a.values())
        detections_b_aligned = list(feature_to_det_b.values())

        confirmed = self.appearance_matcher.verify_candidates(
            detections_a_aligned, detections_b_aligned, geometric_candidates
        )

        logger.debug(
            f"Appearance filter: {len(confirmed)}/{len(geometric_candidates)} "
            f"confirmed (threshold={self.config.appearance_threshold:.2f})"
        )

        # confirmed only carries similarity — store epipolar distances so we can retrieve them by pair key when building CrossCameraMatch
        candidate_distances = {}
        for idx_a, idx_b, distance in geometric_candidates:
            candidate_distances[(idx_a, idx_b)] = distance

        # Create CrossCameraMatch objects
        matches = []
        # for each confirmed pair build a CrossCameraMatch with local_ids (converted from feature indices)
        for idx_a, idx_b, similarity in confirmed:
            # Get epipolar distance from candidates
            epipolar_distance = candidate_distances[(idx_a, idx_b)]

            # Resolve detections via feature index -> local_id mapping
            det_a = feature_to_det_a[idx_a]
            det_b = feature_to_det_b[idx_b]

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
        # the citys
        graph = defaultdict(set)
        
        # for each 2 detection pair we save the CrossCameraMatch object to get appearance_score later
        # how good are the roads between the citys
        match_dict = {}  # {((drone_a, id_a), (drone_b, id_b)): CrossCameraMatch}

        # for each 2 drones get the matches list ([drone1det1 = drone2det2, ect..])
        for matches in pairwise_matches.values():
            # for every match inside that list of matches
            for match in matches:
                # each node is (drone_id, local_id) — unique identifier for one detection
                node_a = (match.drone_id_a, match.local_id_a)
                node_b = (match.drone_id_b, match.local_id_b)

                # add edge in both directions so BFS can traverse freely
                graph[node_a].add(node_b)
                graph[node_b].add(node_a)

                # sorted key so (3,0)↔(4,1) and (4,1)↔(3,0) map to the same entry
                edge = (min(node_a, node_b), max(node_a, node_b))
                # used after bfs to retrieve scores
                match_dict[edge] = match

        # Find connected components via BFS
        # tracks nodes we've already processed
        visited = set()
        
        # output list of MatchGroup objects.
        groups = []

        # for each node in the graph build group from it
        for start_node in graph.keys():
            # skip if already visited
            if start_node in visited:
                continue

            # BFS to find all nodes in this component (deque for O(1) popleft)
            # component collects all nodes in this group
            component = set()
            # acts like a recursive
            bfs_queue = deque([start_node])
            
            # add that node to a fresh group
            component.add(start_node)
            
            # make the node visited 
            visited.add(start_node)
            
            
            while bfs_queue:
                node = bfs_queue.popleft()
                
                # Take the first node from the queue. Look at all its neighbors in graph
                for neighbor in graph[node]:
                    # For each unvisited neighbor — mark visited, add to component, push onto queue
                    if neighbor not in visited:
                        visited.add(neighbor)
                        component.add(neighbor)
                        bfs_queue.append(neighbor)
                        # Keep going until queue is empty = no more reachable nodes.


            # Build MatchGroup from component
            detections = sorted(component)  # List of (drone_id, local_id) tuples
            
            # --- BEGIN CONFLICT RESOLUTION --- detections = [(3,0), (3,2), (4,1), (6,0)]
            # Check if any drone has multiple detections in this group
            drone_to_locals = defaultdict(list)
            for d_id, l_id in detections:
                drone_to_locals[d_id].append(l_id)
            # drone_to_locals = { 3: [0, 2], (← conflict! drone 3 has 2 detections) 4: [1],6: [0], }

            resolved_detections = []
            for d_id, l_ids in drone_to_locals.items():
                if len(l_ids) == 1:
                    # No conflict for this drone
                    resolved_detections.append((d_id, l_ids[0]))
                else:
                    logger.warning(
                        f"Transitive Conflict: Drone {d_id} has {len(l_ids)} "
                        "detections in one group. Resolving by appearance score."
                    )

                    # Conflict! Find the "strongest" detection and discard the ghosts.
                    best_local = None
                    best_score = -1.0

                    # Build its node -> Sum its scores -> compare
                    for l_id in l_ids:
                        node = (d_id, l_id)
                        # Sum the appearance scores for this specific node with the rest of the group
                        score_sum = 0.0
                        for other_node in component:
                            # skip comparing node to itself
                            if node == other_node:
                                continue

                            # Reconstruct the edge key used in match_dict
                            # SUMMING ALL THE SCORES FROM THE EDGES WITH THE NODE
                            edge = (min(node, other_node), max(node, other_node))
                            if edge in match_dict:
                                score_sum += match_dict[edge].appearance_score

                        # Keep the node that has the strongest overall connections
                        if score_sum > best_score:
                            best_score = score_sum
                            best_local = l_id

                    resolved_detections.append((d_id, best_local))
                    # losers are absent from resolved_detections → reconstruction treats them as single-view automatically

            # Update the detections list with the resolved, conflict-free nodes
            detections = sorted(resolved_detections)
            # --- END CONFLICT RESOLUTION ---

            # Compute mean appearance score from all pairwise matches in this group
            scores = []
            # loop over every unique pair
            for i in range(len(detections)):
                for j in range(i + 1, len(detections)):
                    # looks up the CrossCameraMatch in match_dict to get its appearance_score.
                    edge = (
                        min(detections[i], detections[j]),
                        max(detections[i], detections[j]),
                    )
                    if edge in match_dict:
                        scores.append(match_dict[edge].appearance_score)
            # if no edges found do 0 else avg all
            mean_score = np.mean(scores) if len(scores) > 0 else 0.0
            
            # create the MatchGroup
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
