"""
Cross-Camera Fusion Module

Fuses detections across multiple cameras to identify unique persons.

Process:
1. Geometric filtering: Epipolar constraint removes impossible matches (90%+ reduction)
2. Appearance matching: WCH feature similarity for remaining candidates
3. Clustering: Group matches into unique person identities

Key Components:
- models: CrossCameraMatch, MatchGroup, FusionResult
- fundamental_matrix: Compute F matrices from projection matrices
- epipolar_filter: Geometric constraint filtering
- appearance_matcher: Appearance-based matching using WCH features with Hungarian assignment
- cross_camera_matcher: Full fusion pipeline orchestrator

Usage:
    from fusion import (
        CrossCameraMatch,
        MatchGroup,
        FusionResult,
        compute_fundamental_matrix,
        filter_by_epipolar_constraint,
        AppearanceMatcher,
        CrossCameraMatcher,
    )
"""

from fusion.models import (
    CrossCameraMatch,
    MatchGroup,
    FusionResult,
)

from fusion.fundamental_matrix import (
    compute_fundamental_matrix,
    compute_fundamental_matrix_batch,
)

from fusion.epipolar_filter import (
    filter_by_epipolar_constraint,
    filter_matches_batch,
    compute_epipolar_distance,
    point_to_line_distance,
)

from fusion.appearance_matcher import AppearanceMatcher

from fusion.cross_camera_matcher import CrossCameraMatcher

__all__ = [
    # Data models
    "CrossCameraMatch",
    "MatchGroup",
    "FusionResult",
    # Fundamental matrix
    "compute_fundamental_matrix",
    "compute_fundamental_matrix_batch",
    # Epipolar filtering
    "filter_by_epipolar_constraint",
    "filter_matches_batch",
    "compute_epipolar_distance",
    "point_to_line_distance",
    # Appearance matching
    "AppearanceMatcher",
    # Cross-camera fusion
    "CrossCameraMatcher",
]
