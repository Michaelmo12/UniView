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
- visual_matcher: (TODO) Appearance-based matching using WCH features
- cross_drone_matcher: (TODO) Full fusion pipeline

Usage:
    from fusion import (
        CrossCameraMatch,
        MatchGroup,
        FusionResult,
        compute_fundamental_matrix,
        filter_by_epipolar_constraint,
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
]
