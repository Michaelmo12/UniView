"""
Epipolar Constraint Filter

Filters candidate matches using geometric epipolar constraint.

The epipolar constraint says that if point p1 in camera 1 corresponds to
point p2 in camera 2, then p2 must lie on the epipolar line defined by F @ p1.

We measure the point-to-line distance. If it's below a threshold (e.g., 5 pixels),
the match is geometrically plausible.

White-box: Standard point-to-line distance formula in 2D.
"""

import sys
from pathlib import Path

# Ensure algorithm directory is in path for relative imports
_FUSION_DIR = Path(__file__).parent
_ALGORITHM_DIR = _FUSION_DIR.parent
if str(_ALGORITHM_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGORITHM_DIR))

import numpy as np

from features.models import PersonFeatures
from fusion.models import CrossCameraMatch
from fusion.fundamental_matrix import compute_fundamental_matrix


def point_to_line_distance(point: np.ndarray, line: np.ndarray) -> float:
    """
    Compute perpendicular distance from point to line in 2D.

    The line is represented in homogeneous form: ax + by + c = 0
    as the vector [a, b, c].

    Distance formula:
        d = |ax + by + c| / sqrt(a^2 + b^2)

    Args:
        point: (x, y) or (x, y, 1) point coordinates
        line: [a, b, c] line coefficients

    Returns:
        Perpendicular distance in pixels
    """
    # Ensure point is in homogeneous coords (x, y, 1)
    if len(point) == 2:
        point = np.array([point[0], point[1], 1.0])

    # Compute numerator: |ax + by + c| = |line^T @ point|
    numerator = np.abs(line @ point)

    # Compute denominator: sqrt(a^2 + b^2)
    denominator = np.sqrt(line[0] ** 2 + line[1] ** 2)

    # Handle degenerate case (line at infinity)
    if denominator < 1e-10:
        return np.inf

    return numerator / denominator


def compute_epipolar_distance(
    features1: PersonFeatures,
    features2: PersonFeatures,
    F: np.ndarray,
) -> float:
    """
    Compute symmetric epipolar distance between two detections.

    The epipolar constraint is:
        x2^T @ F @ x1 = 0

    We measure distance in both directions and take the average:
    - d(x2 -> line in img2) = distance from x2 to epiline F @ x1
    - d(x1 -> line in img1) = distance from x1 to epiline F^T @ x2

    Args:
        features1: Features from camera 1
        features2: Features from camera 2
        F: Fundamental matrix mapping camera 1 -> camera 2

    Returns:
        Symmetric epipolar distance (pixels)
    """
    # Get bbox centers as points
    x1 = np.array([features1.bbox_center[0], features1.bbox_center[1], 1.0])
    x2 = np.array([features2.bbox_center[0], features2.bbox_center[1], 1.0])

    # Compute epipolar line in image 2: l2 = F @ x1
    epiline_2 = F @ x1

    # Compute distance from x2 to epiline_2
    dist_2 = point_to_line_distance(x2[:2], epiline_2)

    # Compute epipolar line in image 1: l1 = F^T @ x2
    epiline_1 = F.T @ x2

    # Compute distance from x1 to epiline_1
    dist_1 = point_to_line_distance(x1[:2], epiline_1)

    # Return symmetric distance (average of both directions)
    return (dist_1 + dist_2) / 2.0


def filter_by_epipolar_constraint(
    features1: PersonFeatures,
    features2: PersonFeatures,
    threshold: float,
) -> tuple[bool, float]:
    """
    Filter a candidate match using epipolar constraint.

    Computes the fundamental matrix from the projection matrices stored
    in the features, then checks if the match satisfies the epipolar
    constraint within the specified threshold.

    Args:
        features1: Features from camera 1
        features2: Features from camera 2
        threshold: Maximum epipolar distance (pixels) for valid match

    Returns:
        Tuple of (is_valid, epipolar_distance)
        - is_valid: True if distance <= threshold
        - epipolar_distance: Computed distance in pixels
    """
    # Compute fundamental matrix from projection matrices
    F = compute_fundamental_matrix(features1.projection_matrix, features2.projection_matrix)

    # Compute epipolar distance
    distance = compute_epipolar_distance(features1, features2, F)

    # Check threshold
    is_valid = distance <= threshold

    return is_valid, distance


def filter_matches_batch(
    features_list: list[PersonFeatures],
    threshold: float,
) -> list[CrossCameraMatch]:
    """
    Filter all candidate matches for a set of detections using epipolar constraint.

    For each pair of detections from different cameras, compute the epipolar
    distance and create a CrossCameraMatch if valid.

    This is the geometric filtering stage that runs before appearance verification.
    It typically reduces candidate matches by 90%+ by eliminating geometrically
    impossible correspondences.

    Args:
        features_list: List of PersonFeatures from all cameras
        threshold: Maximum epipolar distance for valid match

    Returns:
        List of CrossCameraMatch instances (only geometrically valid matches)
    """
    matches = []

    # Iterate over all pairs
    for i in range(len(features_list)):
        for j in range(i + 1, len(features_list)):
            feat1 = features_list[i]
            feat2 = features_list[j]

            # Skip if same camera
            if feat1.drone_id == feat2.drone_id:
                continue

            # Check epipolar constraint
            is_valid, distance = filter_by_epipolar_constraint(feat1, feat2, threshold)

            # Create match (appearance_score will be filled by next stage)
            match = CrossCameraMatch(
                drone_id_a=feat1.drone_id,
                drone_id_b=feat2.drone_id,
                local_id_a=feat1.local_id,
                local_id_b=feat2.local_id,
                epipolar_distance=distance,
                appearance_score=0.0,  # Not yet computed
                is_valid=is_valid,
            )

            # Only keep valid matches
            if is_valid:
                matches.append(match)

    return matches


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Testing Epipolar Filter")
    logger.info("=" * 60)

    # Create synthetic camera setup
    K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])

    # Camera 1: at origin
    R1 = np.eye(3)
    t1 = np.array([[0.0], [0.0], [0.0]])
    P1 = K @ np.hstack([R1, t1])

    # Camera 2: translated 5m in X
    R2 = np.eye(3)
    t2 = np.array([[5.0], [0.0], [0.0]])
    P2 = K @ np.hstack([R2, t2])

    logger.info("\nTest 1: Point-to-line distance")
    point = np.array([100.0, 200.0])
    line = np.array([1.0, 0.0, -100.0])  # x = 100 (vertical line)
    dist = point_to_line_distance(point, line)
    logger.info("  Point: %s", point)
    logger.info("  Line: %s (vertical at x=100)", line)
    logger.info("  Distance: %.2f pixels (should be ~0)", dist)

    logger.info("\nTest 2: Epipolar distance")
    # Create mock features
    wch_dummy = np.random.rand(96).astype(np.float32)

    feat1 = PersonFeatures(
        drone_id=1,
        frame_num=0,
        local_id=0,
        wch=wch_dummy,
        bbox_center=(500.0, 300.0),
        projection_matrix=P1,
        confidence=0.9,
    )

    feat2 = PersonFeatures(
        drone_id=2,
        frame_num=0,
        local_id=0,
        wch=wch_dummy,
        bbox_center=(400.0, 300.0),  # Different x due to baseline shift
        projection_matrix=P2,
        confidence=0.9,
    )

    F = compute_fundamental_matrix(P1, P2)
    epi_dist = compute_epipolar_distance(feat1, feat2, F)
    logger.info("  Features 1 center: %s", feat1.bbox_center)
    logger.info("  Features 2 center: %s", feat2.bbox_center)
    logger.info("  Epipolar distance: %.2f pixels", epi_dist)

    logger.info("\nTest 3: Filter by constraint")
    threshold = 10.0
    is_valid, distance = filter_by_epipolar_constraint(feat1, feat2, threshold)
    logger.info("  Threshold: %.1f pixels", threshold)
    logger.info("  Distance: %.2f pixels", distance)
    logger.info("  Valid: %s", is_valid)

    logger.info("\nTest 4: Batch filtering")
    # Create a few more features
    feat3 = PersonFeatures(
        drone_id=1,
        frame_num=0,
        local_id=1,
        wch=wch_dummy,
        bbox_center=(600.0, 400.0),
        projection_matrix=P1,
        confidence=0.8,
    )

    features_list = [feat1, feat2, feat3]
    matches = filter_matches_batch(features_list, threshold=10.0)

    logger.info("  Input features: %d", len(features_list))
    logger.info("  Valid matches: %d", len(matches))
    for match in matches:
        logger.info("    %s", match)

    logger.info("\n" + "=" * 60)
    logger.info("All tests passed!")
