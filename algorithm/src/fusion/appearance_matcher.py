import logging
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.detection.models import Detection
from src.config.settings import FusionConfig

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
        self.config = config

    def _extract_features(self, detections: list[Detection]) -> list[np.ndarray]:
        """
        Extract WCH features from detections, using zero vectors for missing features.

        Args:
            detections: List of Detection objects

        Returns:
            List of WCH feature vectors (96-dimensional)
        """
        features = []
        for det in detections:
            if det.features is not None:
                features.append(det.features)
            else:
                logger.warning(
                    f"Detection drone={det.drone_id} frame={det.frame_num} "
                    f"local_id={det.local_id} has no features"
                )
                features.append(np.zeros(96, dtype=np.float32))
        return features

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

        if n == 0 or m == 0:
            return np.zeros((n, m), dtype=np.float64)

        features_a = self._extract_features(detections_a)
        features_b = self._extract_features(detections_b)

        # Stack into matrices from numpy arrays (ensuring float64 for precision in similarity computation)
        A = np.vstack(features_a).astype(np.float64)  # (n, 96)
        B = np.vstack(features_b).astype(np.float64)  # (m, 96)

        # Compute similarity: A @ B.T
        # Since WCH is L2-normalized, dot product equals cosine similarity(Cosine similarity = (a · b) / (||a|| × ||b||) and ||a|| = ||b|| = 1)
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
        a_to_sub = {}
        for sub, orig in enumerate(idx_a_list):
            a_to_sub[orig] = sub

        b_to_sub = {}
        for sub, orig in enumerate(idx_b_list):
            b_to_sub[orig] = sub

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

        # Run Hungarian algorithm - must pick one value from each row and one value from each column. so we only take detections with best similarity
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Build set of valid candidate pairs for fast lookup
        valid_pairs = set()
        for idx_a, idx_b, _ in candidates:
            valid_pairs.add((idx_a, idx_b))

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
        valid_a_indices = set()
        for i, det in enumerate(detections_a):
            if det.features is not None:
                valid_a_indices.add(i)

        valid_b_indices = set()
        for i, det in enumerate(detections_b):
            if det.features is not None:
                valid_b_indices.add(i)

        # Filter candidates
        filtered_candidates = []
        for idx_a, idx_b, dist in candidates:
            if idx_a in valid_a_indices and idx_b in valid_b_indices:
                filtered_candidates.append((idx_a, idx_b, dist))

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

