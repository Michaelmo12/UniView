import numpy as np

from src.features.models import PersonFeatures
from src.fusion.models import CrossCameraMatch
from src.fusion.fundamental_matrix import compute_fundamental_matrix


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
    if len(point) == 2:
        point = np.array([point[0], point[1], 1.0])

    numerator = np.abs(line @ point)

    denominator = np.sqrt(line[0] ** 2 + line[1] ** 2)

    # line at infinity
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
    F = compute_fundamental_matrix(features1.projection_matrix, features2.projection_matrix)

    distance = compute_epipolar_distance(features1, features2, F)

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

    for i in range(len(features_list)):
        for j in range(i + 1, len(features_list)):
            feat1 = features_list[i]
            feat2 = features_list[j]

            if feat1.drone_id == feat2.drone_id:
                continue

            is_valid, distance = filter_by_epipolar_constraint(feat1, feat2, threshold)

            if is_valid:
                match = CrossCameraMatch(
                    drone_id_a=feat1.drone_id,
                    drone_id_b=feat2.drone_id,
                    local_id_a=feat1.local_id,
                    local_id_b=feat2.local_id,
                    epipolar_distance=distance,
                    appearance_score=0.0,  # Not yet computed (filled by appearance matcher)
                )
                matches.append(match)

    return matches

