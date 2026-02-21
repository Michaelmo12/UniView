"""
Appearance-based matching using WCH feature similarity.

Uses cosine similarity between WCH descriptors to verify geometric candidates.
Applies Hungarian algorithm for optimal 1-to-1 assignment.

White-box: Matrix multiplication for cosine similarity (since WCH is L2-normalized),
scipy.optimize.linear_sum_assignment for optimal bipartite matching.
"""

import logging
import numpy as np
from scipy.optimize import linear_sum_assignment

from detection.models import Detection
from config.settings import FusionConfig

logger = logging.getLogger(__name__)


class AppearanceMatcher:
    """
    Verifies geometric candidates using appearance (WCH) similarity.

    Process:
    1. Extract WCH features from detections
    2. Compute cosine similarity matrix (via dot product, since WCH is L2-normalized)
    3. Run Hungarian algorithm for optimal 1-to-1 assignment
    4. Filter by appearance threshold
    """

    def __init__(self, config: FusionConfig):
        """
        Initialize appearance matcher.

        Args:
            config: FusionConfig with appearance_threshold
        """
        self.config = config

    def build_similarity_matrix(
        self, detections_a: list[Detection], detections_b: list[Detection]
    ) -> np.ndarray:
        """
        Build cosine similarity matrix from WCH features.

        Args:
            detections_a: Detections from camera A
            detections_b: Detections from camera B

        Returns:
            Similarity matrix (n, m) where entry [i, j] is cosine similarity
            between detection_a[i] and detection_b[j]. Returns zero similarity
            for detections with None features.
        """
        n = len(detections_a)
        m = len(detections_b)

        # Handle empty cases
        if n == 0 or m == 0:
            return np.zeros((n, m), dtype=np.float64)

        # Extract features from detections_a
        features_a = []
        for det in detections_a:
            if det.features is not None:
                features_a.append(det.features)
            else:
                # No features - use zero vector (will result in zero similarity)
                logger.warning(
                    f"Detection drone={det.drone_id} frame={det.frame_num} "
                    f"local_id={det.local_id} has no features"
                )
                features_a.append(np.zeros(96, dtype=np.float32))

        # Extract features from detections_b
        features_b = []
        for det in detections_b:
            if det.features is not None:
                features_b.append(det.features)
            else:
                logger.warning(
                    f"Detection drone={det.drone_id} frame={det.frame_num} "
                    f"local_id={det.local_id} has no features"
                )
                features_b.append(np.zeros(96, dtype=np.float32))

        # Stack into matrices
        A = np.vstack(features_a).astype(np.float64)  # (n, 96)
        B = np.vstack(features_b).astype(np.float64)  # (m, 96)

        # Compute similarity: A @ B.T
        # Since WCH is L2-normalized, dot product equals cosine similarity
        similarity = A @ B.T  # (n, m)

        return similarity

    def optimal_assignment(
        self,
        similarity_matrix: np.ndarray,
        candidates: list[tuple[int, int, float]],
    ) -> list[tuple[int, int, float]]:
        """
        Perform optimal 1-to-1 assignment using Hungarian algorithm.

        Only candidate pairs (those that passed epipolar filter) participate
        in the assignment. This is a sparse bipartite matching problem.

        Args:
            similarity_matrix: Full (n, m) similarity matrix
            candidates: List of (idx_a, idx_b, epipolar_distance) from geometric filter

        Returns:
            List of (idx_a, idx_b, similarity) for optimal matches above threshold
        """
        if len(candidates) == 0:
            return []

        # Extract unique indices from candidates
        idx_a_set = set()
        idx_b_set = set()
        for idx_a, idx_b, _ in candidates:
            idx_a_set.add(idx_a)
            idx_b_set.add(idx_b)

        # Create sorted lists for consistent ordering
        idx_a_list = sorted(idx_a_set)
        idx_b_list = sorted(idx_b_set)

        # Build mapping from original indices to sub-matrix indices
        a_to_sub = {orig: sub for sub, orig in enumerate(idx_a_list)}
        b_to_sub = {orig: sub for sub, orig in enumerate(idx_b_list)}

        # Create sub-matrix for candidate pairs only
        n_sub = len(idx_a_list)
        m_sub = len(idx_b_list)
        sub_similarity = np.zeros((n_sub, m_sub), dtype=np.float64)

        # Fill sub-matrix from full similarity matrix
        for idx_a, idx_b, _ in candidates:
            sub_i = a_to_sub[idx_a]
            sub_j = b_to_sub[idx_b]
            sub_similarity[sub_i, sub_j] = similarity_matrix[idx_a, idx_b]

        # Convert similarity to cost (Hungarian minimizes)
        cost_matrix = 1.0 - sub_similarity

        # Run Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Build set of valid candidate pairs for fast lookup
        valid_pairs = {(idx_a, idx_b) for idx_a, idx_b, _ in candidates}

        # Map back to original indices and filter by threshold
        result = []
        for sub_i, sub_j in zip(row_ind, col_ind):
            orig_i = idx_a_list[sub_i]
            orig_j = idx_b_list[sub_j]

            # Only keep if this pair was a geometric candidate
            if (orig_i, orig_j) not in valid_pairs:
                continue

            similarity = similarity_matrix[orig_i, orig_j]

            # Only keep if similarity exceeds threshold
            if similarity >= self.config.appearance_threshold:
                result.append((orig_i, orig_j, float(similarity)))

        logger.debug(
            f"Appearance matcher: {len(result)}/{len(candidates)} candidates confirmed "
            f"(threshold={self.config.appearance_threshold:.2f})"
        )

        return result

    def verify_candidates(
        self,
        detections_a: list[Detection],
        detections_b: list[Detection],
        candidates: list[tuple[int, int, float]],
    ) -> list[tuple[int, int, float]]:
        """
        Verify geometric candidates using appearance similarity.

        This is the main entry point for appearance verification.

        Args:
            detections_a: Detections from camera A
            detections_b: Detections from camera B
            candidates: List of (idx_a, idx_b, epipolar_distance) from geometric filter

        Returns:
            List of (idx_a, idx_b, similarity) for confirmed matches
        """
        # Early exit if no candidates
        if len(candidates) == 0:
            logger.debug("No candidates to verify")
            return []

        # Check if any detections have features
        valid_a = any(det.features is not None for det in detections_a)
        valid_b = any(det.features is not None for det in detections_b)

        if not valid_a or not valid_b:
            logger.warning(
                "No valid features available - cannot verify candidates by appearance"
            )
            return []

        # Filter candidates to only those with valid features on both sides
        # Build sets of indices with valid features
        valid_a_indices = {
            i for i, det in enumerate(detections_a) if det.features is not None
        }
        valid_b_indices = {
            i for i, det in enumerate(detections_b) if det.features is not None
        }

        # Filter candidates
        filtered_candidates = [
            (idx_a, idx_b, dist)
            for idx_a, idx_b, dist in candidates
            if idx_a in valid_a_indices and idx_b in valid_b_indices
        ]

        if len(filtered_candidates) == 0:
            logger.warning(
                f"All {len(candidates)} candidates filtered out due to missing features"
            )
            return []

        if len(filtered_candidates) < len(candidates):
            logger.debug(
                f"Filtered candidates: {len(filtered_candidates)}/{len(candidates)} "
                f"have valid features on both sides"
            )

        # Build full similarity matrix (needed for indexing by optimal_assignment)
        similarity_matrix = self.build_similarity_matrix(detections_a, detections_b)

        # Run optimal assignment on filtered candidates
        confirmed = self.optimal_assignment(similarity_matrix, filtered_candidates)

        logger.info(
            f"Appearance matcher: {len(confirmed)}/{len(filtered_candidates)} "
            f"candidates confirmed (threshold={self.config.appearance_threshold:.2f})"
        )

        return confirmed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Testing AppearanceMatcher")
    logger.info("=" * 60)

    from detection.models import BoundingBox
    from config.settings import settings

    # Create synthetic WCH features
    np.random.seed(42)

    # Person A: distinct appearance
    wch_person_a = np.random.randn(96).astype(np.float32)
    wch_person_a /= np.linalg.norm(wch_person_a)

    # Person B: different appearance
    wch_person_b = np.random.randn(96).astype(np.float32)
    wch_person_b /= np.linalg.norm(wch_person_b)

    # Helper to create detection
    def make_det(drone_id, local_id, cx, cy, features):
        bbox = BoundingBox(x1=cx - 25, y1=cy - 50, x2=cx + 25, y2=cy + 50)
        det = Detection(
            bbox=bbox,
            class_id=0,
            confidence=0.9,
            drone_id=drone_id,
            frame_num=0,
            local_id=local_id,
        )
        det.features = features
        return det

    logger.info("\nTest 1: Similarity matrix")
    dets_cam1 = [
        make_det(1, 0, 500, 300, wch_person_a.copy()),
        make_det(1, 1, 700, 400, wch_person_b.copy()),
    ]
    dets_cam2 = [
        make_det(2, 0, 600, 310, wch_person_a.copy()),
        make_det(2, 1, 800, 500, wch_person_b.copy()),
    ]

    matcher = AppearanceMatcher(settings.fusion)
    sim_matrix = matcher.build_similarity_matrix(dets_cam1, dets_cam2)

    logger.info("  Similarity matrix:")
    logger.info("  %s", sim_matrix)
    logger.info("  sim[0, 0] (A<->A): %.3f (should be ~1.0)", sim_matrix[0, 0])
    logger.info("  sim[1, 1] (B<->B): %.3f (should be ~1.0)", sim_matrix[1, 1])
    logger.info("  sim[0, 1] (A<->B): %.3f (should be low)", sim_matrix[0, 1])
    logger.info("  sim[1, 0] (B<->A): %.3f (should be low)", sim_matrix[1, 0])

    assert sim_matrix[0, 0] > 0.99, "Same person should have high similarity"
    assert sim_matrix[1, 1] > 0.99, "Same person should have high similarity"
    logger.info("  ✓ PASS: Diagonal elements have high similarity")

    logger.info("\nTest 2: Optimal assignment")
    # All pairs are geometric candidates
    candidates = [(0, 0, 2.0), (0, 1, 8.0), (1, 0, 9.0), (1, 1, 3.0)]

    confirmed = matcher.verify_candidates(dets_cam1, dets_cam2, candidates)

    logger.info("  Candidates: %d", len(candidates))
    logger.info("  Confirmed: %d", len(confirmed))
    for idx_a, idx_b, sim in confirmed:
        logger.info("    det_a[%d] <-> det_b[%d]: similarity=%.3f", idx_a, idx_b, sim)

    # Should match 0-0 (A-A) and 1-1 (B-B), not cross-matches
    assert len(confirmed) == 2, f"Should have 2 matches, got {len(confirmed)}"
    confirmed_pairs = {(idx_a, idx_b) for idx_a, idx_b, _ in confirmed}
    assert (0, 0) in confirmed_pairs, "Should match det 0 <-> det 0"
    assert (1, 1) in confirmed_pairs, "Should match det 1 <-> det 1"
    logger.info("  ✓ PASS: Correct 1-to-1 assignment")

    logger.info("\nTest 3: Appearance threshold filtering")
    # Add a low-similarity match
    wch_person_c = np.random.randn(96).astype(np.float32)
    wch_person_c /= np.linalg.norm(wch_person_c)

    dets_cam1_v2 = [make_det(1, 0, 500, 300, wch_person_a.copy())]
    dets_cam2_v2 = [
        make_det(2, 0, 600, 310, wch_person_a * 0.95 + wch_person_c * 0.3),  # Similar
        make_det(2, 1, 800, 500, wch_person_c.copy()),  # Different
    ]
    # Re-normalize
    for det in dets_cam2_v2:
        det.features = det.features / np.linalg.norm(det.features)

    candidates_v2 = [(0, 0, 5.0), (0, 1, 8.0)]
    confirmed_v2 = matcher.verify_candidates(dets_cam1_v2, dets_cam2_v2, candidates_v2)

    logger.info("  Candidates: %d", len(candidates_v2))
    logger.info("  Confirmed: %d", len(confirmed_v2))
    for idx_a, idx_b, sim in confirmed_v2:
        logger.info("    det_a[%d] <-> det_b[%d]: similarity=%.3f", idx_a, idx_b, sim)

    # Person A should match cam2 det 0, not det 1
    if len(confirmed_v2) > 0:
        assert confirmed_v2[0][1] == 0, f"Should match det 0, got det {confirmed_v2[0][1]}"
        logger.info("  ✓ PASS: Correct match selected")
    else:
        logger.warning("  ⚠ No matches confirmed (threshold too high)")

    logger.info("\nTest 4: Missing features handling")
    dets_cam1_v3 = [make_det(1, 0, 500, 300, None)]  # No features
    dets_cam2_v3 = [make_det(2, 0, 600, 310, wch_person_a.copy())]

    candidates_v3 = [(0, 0, 5.0)]
    confirmed_v3 = matcher.verify_candidates(dets_cam1_v3, dets_cam2_v3, candidates_v3)

    logger.info("  Detections with missing features")
    logger.info("  Confirmed: %d (should be 0)", len(confirmed_v3))
    assert len(confirmed_v3) == 0, "Should not match when features missing"
    logger.info("  ✓ PASS: Missing features handled correctly")

    logger.info("\n" + "=" * 60)
    logger.info("All tests passed!")
