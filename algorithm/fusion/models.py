"""
Fusion Stage Data Models

Defines data structures for cross-camera matching and fusion results.

Key Types:
- CrossCameraMatch: A match between detections from two different cameras
- MatchGroup: A group of detections across cameras representing the same person
- FusionResult: Complete fusion output for one synchronized frame set
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class CrossCameraMatch:
    """
    A valid match between two detections from different cameras.

    Represents a correspondence between a person seen in camera A and camera B
    that has passed BOTH geometric (epipolar) and appearance (WCH) constraints.

    Note: Only valid matches are instantiated. Invalid candidates are discarded
    during filtering and never create CrossCameraMatch objects.
    """

    drone_id_a: int #First camera ID
    drone_id_b: int #Second camera ID
    local_id_a: int #Detection ID in camera A
    local_id_b: int #Detection ID in camera B
    epipolar_distance: float #Point-to-epiline distance (pixels) - geometric score
    appearance_score: float #WCH cosine similarity [0, 1] - appearance score

    def __post_init__(self):
        assert self.drone_id_a != self.drone_id_b, (
            f"CrossCameraMatch requires different cameras, "
            f"got drone_id_a={self.drone_id_a}, drone_id_b={self.drone_id_b}"
        )
        assert self.epipolar_distance >= 0.0, (
            f"epipolar_distance must be >= 0, got {self.epipolar_distance}"
        )
        assert 0.0 <= self.appearance_score <= 1.0, (
            f"appearance_score must be in [0, 1], got {self.appearance_score}"
        )

    def __repr__(self) -> str:
        return (
            f"CrossCameraMatch(drone_{self.drone_id_a}[{self.local_id_a}] <-> "
            f"drone_{self.drone_id_b}[{self.local_id_b}], "
            f"epi={self.epipolar_distance:.1f}px, "
            f"app={self.appearance_score:.3f})"
        )


@dataclass
class MatchGroup:
    """
    A group of detections across multiple cameras representing the same person.

    This is the output of the fusion stage: all detections that are believed
    to be the same individual across different camera views.

    The group is built by:
    1. Starting with pairwise CrossCameraMatch instances
    2. Clustering matches using transitive closure
    3. Each cluster becomes one MatchGroup
    """

    detections: list[tuple[int, int]] = field(default_factory=list) #List of (drone_id, local_id) tuples for all detections in group
    mean_appearance_score: float = 0.0 #Average appearance score across all pairwise matches

    @property
    def num_cameras(self) -> int:
        """Count distinct cameras in this group."""
        drone_ids = {drone_id for drone_id, _ in self.detections}
        return len(drone_ids)

    @property
    def num_detections(self) -> int:
        """Total number of detections in this group."""
        return len(self.detections)

    @property
    def is_empty(self) -> bool:
        """Check if group has no detections."""
        return len(self.detections) == 0

    def get_drone_ids(self) -> list[int]:
        """Get sorted list of unique drone IDs in this group."""
        drone_ids = {drone_id for drone_id, _ in self.detections}
        return sorted(drone_ids)

    def has_detection_from(self, drone_id: int) -> bool:
        """Check if this group has a detection from specified drone."""
        return any(d == drone_id for d, _ in self.detections)

    def get_detection_id(self, drone_id: int) -> int | None:
        """
        Get the local detection ID for a specific drone.

        Args:
            drone_id: Drone to query

        Returns:
            local_id if drone has detection in this group, None otherwise
        """
        for d, local_id in self.detections:
            if d == drone_id:
                return local_id
        return None

    def __repr__(self) -> str:
        """String representation for debugging."""
        drone_ids = self.get_drone_ids()
        return (
            f"MatchGroup(cameras={self.num_cameras}, "
            f"detections={self.num_detections}, "
            f"drones={drone_ids}, "
            f"mean_score={self.mean_appearance_score:.3f})"
        )


@dataclass
class FusionResult:
    """
    Complete fusion output for one synchronized frame set.

    Contains all match groups found across cameras for a single timestamp.
    Each group represents a unique person observed across one or more cameras.

    Attributes:
        frame_num: Frame number being processed
        match_groups: List of MatchGroup instances (one per unique person)
        total_detections: Total number of input detections across all cameras
        total_matches: Total number of pairwise matches found
    """

    frame_num: int
    match_groups: list[MatchGroup] = field(default_factory=list)
    total_detections: int = 0
    total_matches: int = 0

    @property
    def num_groups(self) -> int:
        """Number of unique persons found."""
        return len(self.match_groups)

    @property
    def is_empty(self) -> bool:
        """Check if no matches were found."""
        return len(self.match_groups) == 0

    def get_groups_with_min_cameras(self, min_cameras: int) -> list[MatchGroup]:
        """
        Filter match groups by minimum number of cameras.

        Args:
            min_cameras: Minimum cameras required

        Returns:
            List of MatchGroups observed by at least min_cameras cameras
        """
        return [g for g in self.match_groups if g.num_cameras >= min_cameras]

    def __repr__(self) -> str:
        return (
            f"FusionResult(frame={self.frame_num}, "
            f"groups={self.num_groups}, "
            f"detections={self.total_detections}, "
            f"matches={self.total_matches})"
        )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Testing Fusion Models")
    logger.info("=" * 60)

    logger.info("\nTest CrossCameraMatch:")
    match1 = CrossCameraMatch(
        drone_id_a=1,
        drone_id_b=2,
        local_id_a=0,
        local_id_b=3,
        epipolar_distance=2.5,
        appearance_score=0.85,
    )
    logger.info(match1)

    logger.info("\nTest MatchGroup:")
    group = MatchGroup(
        detections=[(1, 0), (2, 3), (3, 1)],
        mean_appearance_score=0.82,
    )
    logger.info(group)
    logger.info("  num_cameras: %d", group.num_cameras)
    logger.info("  num_detections: %d", group.num_detections)
    logger.info("  drone_ids: %s", group.get_drone_ids())
    logger.info("  has_detection_from(2): %s", group.has_detection_from(2))
    logger.info("  get_detection_id(2): %s", group.get_detection_id(2))

    logger.info("\nTest FusionResult:")
    result = FusionResult(
        frame_num=42,
        match_groups=[group],
        total_detections=10,
        total_matches=5,
    )
    logger.info(result)
    logger.info("  num_groups: %d", result.num_groups)
    logger.info("  groups with min 3 cameras: %d", len(result.get_groups_with_min_cameras(3)))

    logger.info("\n" + "=" * 60)
    logger.info("All tests passed!")
