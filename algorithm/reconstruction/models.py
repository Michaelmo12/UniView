"""
Reconstruction Stage Data Models

Defines data structures for 3D reconstruction from cross-camera fusion results.

Key Types:
- Point3D: A single triangulated 3D point from matched detections
- Person3D: A unique person in 3D space (from DBSCAN cluster or single-view)
- ReconstructionResult: Complete reconstruction output for one synchronized frame set
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Point3D:
    """
    A single triangulated 3D point from matched detections.

    Represents a 3D world position computed from 2+ camera views using
    triangulation (DLT via cv2.triangulatePoints). Each point comes from
    one MatchGroup and is validated via reprojection error.

    Attributes:
        position: (3,) array of world coordinates (x, y, z) in meters
        reprojection_error: Average pixel error across all views used
        num_views: Number of camera views used for triangulation (>= 2)
        source_detections: List of (drone_id, local_id) tuples that produced this point
    """

    position: np.ndarray  # (3,) world coordinates
    reprojection_error: float  # Average pixel error
    num_views: int  # Number of cameras used (>= 2)
    source_detections: list[tuple[int, int]]  # (drone_id, local_id) tuples

    def __post_init__(self):
        assert self.position.shape == (3,), (
            f"position must be (3,), got {self.position.shape}"
        )
        assert self.reprojection_error >= 0.0, (
            f"reprojection_error must be >= 0, got {self.reprojection_error}"
        )
        assert self.num_views >= 2, (
            f"num_views must be >= 2 for triangulation, got {self.num_views}"
        )

    def __repr__(self) -> str:
        return (
            f"Point3D(pos=[{self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f}], "
            f"error={self.reprojection_error:.2f}px, views={self.num_views})"
        )


@dataclass
class Person3D:
    """
    A unique person in 3D space.

    Can be created in two ways:
    1. From DBSCAN cluster of triangulated points (is_triangulated=True)
    2. From single-view detection with no matches (is_triangulated=False)

    For clustered persons: position is centroid of cluster points.
    For single-view persons: position is None or zeros (no 3D info available).

    Attributes:
        person_id: Unique ID within this frame's reconstruction (0-indexed)
        position: (3,) world coordinates. Cluster: centroid. Single-view: None or zeros.
        num_views: Total camera views (cluster: sum of point views; single-view: 1)
        source_detections: All (drone_id, local_id) tuples for this person
        is_triangulated: True if position from triangulation, False if single-view
    """

    person_id: int  # Unique ID within frame (0, 1, 2, ...)
    position: np.ndarray | None  # (3,) world coordinates or None
    num_views: int  # Total camera views
    source_detections: list[tuple[int, int]]  # All (drone_id, local_id)
    is_triangulated: bool  # True = multi-view triangulated, False = single-view

    def __repr__(self) -> str:
        if self.position is not None:
            pos_str = f"[{self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f}]"
        else:
            pos_str = "None"

        return (
            f"Person3D(id={self.person_id}, pos={pos_str}, "
            f"views={self.num_views}, triangulated={self.is_triangulated})"
        )


@dataclass
class ReconstructionResult:
    """
    Complete reconstruction output for one synchronized frame set.

    Contains all unique persons found across cameras for a single timestamp.
    Each person is either triangulated (from matched detections) or single-view
    (detected by only one camera).

    Attributes:
        frame_num: Frame number being processed
        persons: All unique persons (both triangulated and single-view)
        num_triangulated_points: How many match groups were successfully triangulated
        num_rejected_points: How many were rejected due to high reprojection error
    """

    frame_num: int
    persons: list[Person3D] = field(default_factory=list)
    num_triangulated_points: int = 0
    num_rejected_points: int = 0

    @property
    def num_persons(self) -> int:
        """Total number of unique persons found."""
        return len(self.persons)

    @property
    def triangulated_persons(self) -> list[Person3D]:
        """Filter persons that were triangulated from multiple views."""
        return [p for p in self.persons if p.is_triangulated]

    @property
    def single_view_persons(self) -> list[Person3D]:
        """Filter persons that were only seen in one camera (no matches)."""
        return [p for p in self.persons if not p.is_triangulated]

    def __repr__(self) -> str:
        return (
            f"ReconstructionResult(frame={self.frame_num}, "
            f"persons={self.num_persons}, "
            f"triangulated={len(self.triangulated_persons)}, "
            f"single_view={len(self.single_view_persons)}, "
            f"points_ok={self.num_triangulated_points}, "
            f"points_rejected={self.num_rejected_points})"
        )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Testing Reconstruction Models")
    logger.info("=" * 60)

    # Test Point3D
    logger.info("\nTest Point3D:")
    point = Point3D(
        position=np.array([1.5, 2.3, 0.8]),
        reprojection_error=2.5,
        num_views=3,
        source_detections=[(1, 0), (2, 3), (3, 1)]
    )
    logger.info(point)

    # Test Person3D (triangulated)
    logger.info("\nTest Person3D (triangulated):")
    person_tri = Person3D(
        person_id=0,
        position=np.array([1.2, 2.0, 0.5]),
        num_views=4,
        source_detections=[(1, 0), (2, 3)],
        is_triangulated=True
    )
    logger.info(person_tri)

    # Test Person3D (single-view)
    logger.info("\nTest Person3D (single-view):")
    person_single = Person3D(
        person_id=1,
        position=None,
        num_views=1,
        source_detections=[(4, 2)],
        is_triangulated=False
    )
    logger.info(person_single)

    # Test ReconstructionResult
    logger.info("\nTest ReconstructionResult:")
    result = ReconstructionResult(
        frame_num=42,
        persons=[person_tri, person_single],
        num_triangulated_points=5,
        num_rejected_points=2
    )
    logger.info(result)
    logger.info("  triangulated_persons: %d", len(result.triangulated_persons))
    logger.info("  single_view_persons: %d", len(result.single_view_persons))

    logger.info("\n" + "=" * 60)
    logger.info("All tests passed!")
