"""
Tracking Stage Data Models

Public data structures for the temporal tracking stage:
- TrackState: lifecycle states for a track
- TrackedPerson: a confirmed tracked person with Kalman-smoothed position
- TrackingResult: output of one frame through the tracker
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from algorithm.reconstruction.models import Person3D


class TrackState(Enum):
    """Track lifecycle states."""

    TENTATIVE = "TENTATIVE"    # Newly created, not yet confirmed
    CONFIRMED = "CONFIRMED"    # Confirmed after n_init consecutive hits
    DELETED = "DELETED"        # Marked for removal (missed too long or tentative missed)


@dataclass
class TrackedPerson:
    """A confirmed tracked person with Kalman-smoothed position."""

    global_id: int                   # Stable ID across all frames for this person's lifetime
    position: np.ndarray             # (3,) Kalman-smoothed world position [meters]
    velocity: np.ndarray             # (3,) estimated velocity [m/frame]
    state: TrackState                # Current track state
    frames_tracked: int              # Age of track (total frames since creation)
    frames_since_update: int         # Frames since last matched detection (0 = updated this frame)
    source_person: Person3D          # The Person3D that last updated this track


@dataclass
class TrackingResult:
    """Output from one frame through PersonTracker."""

    frame_num: int
    tracked_persons: list[TrackedPerson] = field(default_factory=list)   # Only CONFIRMED tracks
    single_view_persons: list[Person3D] = field(default_factory=list)    # Pass-through, no IDs
    num_tentative: int = 0
    num_coasting: int = 0
    num_retired: int = 0

    def __repr__(self) -> str:
        return (
            f"TrackingResult(frame={self.frame_num}, "
            f"confirmed={len(self.tracked_persons)}, "
            f"tentative={self.num_tentative}, "
            f"coasting={self.num_coasting}, "
            f"retired={self.num_retired})"
        )
