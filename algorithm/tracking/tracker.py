"""
PersonTracker

Orchestrates SORT-style temporal tracking:
1. Predict all existing tracks forward (Kalman predict)
2. Associate predictions to new detections (Hungarian algorithm)
3. Update matched tracks with new measurements
4. Mark unmatched tracks as missed (coasting)
5. Create new TENTATIVE tracks for unmatched detections
6. Retire DELETED tracks and emit only CONFIRMED tracks in output

Single-view persons (is_triangulated=False or position=None) are passed through
without Kalman tracking or global ID assignment.
"""

import sys
from pathlib import Path

# Add project root to path for algorithm imports when run as script
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import logging

import numpy as np

from algorithm.config.settings import TrackingConfig, settings
from algorithm.reconstruction.models import Person3D, ReconstructionResult
from algorithm.tracking.global_id_manager import GlobalIDManager
from algorithm.tracking.kalman_filter import PersonKalmanFilter
from algorithm.tracking.models import TrackState, TrackedPerson, TrackingResult
from algorithm.tracking.track_associator import TrackAssociator

logger = logging.getLogger(__name__)


class Track:
    """Internal track representation (private to tracker.py)."""

    def __init__(self, global_id: int, initial_position: np.ndarray, config: TrackingConfig) -> None:
        self.global_id = global_id
        self.kalman = PersonKalmanFilter(initial_position, config)
        self.state = TrackState.TENTATIVE
        self.hit_streak = 0
        self.time_since_update = 0
        self.age = 0
        self.last_person: Person3D | None = None

    def predict(self) -> None:
        """Predict track forward one time step."""
        self.kalman.predict()
        self.age += 1

    def update(self, person: Person3D, config: TrackingConfig) -> None:
        """Update track with new detection."""
        self.kalman.update(person.position)
        self.time_since_update = 0
        self.hit_streak += 1
        self.last_person = person
        if self.hit_streak >= config.n_init and self.state == TrackState.TENTATIVE:
            self.state = TrackState.CONFIRMED

    def mark_missed(self, config: TrackingConfig) -> None:
        """Mark track as missed (no detection this frame)."""
        self.time_since_update += 1
        self.hit_streak = 0
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif self.state == TrackState.CONFIRMED and self.time_since_update > config.max_age:
            self.state = TrackState.DELETED

    @property
    def predicted_position(self) -> np.ndarray:
        """Get predicted position from Kalman filter."""
        return self.kalman.predicted_position

    def is_deleted(self) -> bool:
        """Check if track is marked for deletion."""
        return self.state == TrackState.DELETED

    def to_tracked_person(self) -> TrackedPerson:
        """Convert internal Track to public TrackedPerson."""
        return TrackedPerson(
            global_id=self.global_id,
            position=self.kalman.predicted_position,
            velocity=self.kalman.velocity,
            state=self.state,
            frames_tracked=self.age,
            frames_since_update=self.time_since_update,
            source_person=self.last_person,
        )


class PersonTracker:
    """
    SORT-style temporal tracker for 3D persons.

    Reads config from settings.tracking. Maintains a list of active Track
    objects and runs the full predict-associate-update-create-retire loop
    each frame.
    """

    def __init__(self) -> None:
        self.config: TrackingConfig = settings.tracking
        self.associator = TrackAssociator(self.config)
        self.id_manager = GlobalIDManager()
        self.tracks: list[Track] = []
        self.frame_count: int = 0
        self.total_retired: int = 0

    def update(self, reconstruction_result: ReconstructionResult) -> TrackingResult:
        """
        Process one frame of reconstruction output and return tracked persons.

        Args:
            reconstruction_result: Output from Phase 3 SceneReconstructor.

        Returns:
            TrackingResult with CONFIRMED tracked persons and single-view pass-throughs.
        """
        self.frame_count += 1

        # Step 1: Predict ALL existing tracks forward before association
        for track in self.tracks:
            track.predict()

        # Step 2: Separate triangulated detections from single-view
        triangulated = [
            p for p in reconstruction_result.persons
            if p.is_triangulated and p.position is not None
        ]
        single_view = [
            p for p in reconstruction_result.persons
            if not (p.is_triangulated and p.position is not None)
        ]

        # Step 3: Associate triangulated detections to existing non-deleted tracks
        active_tracks = [t for t in self.tracks if not t.is_deleted()]
        matches, unmatched_tracks, unmatched_dets = self.associator.associate(
            active_tracks, triangulated
        )

        # Step 4 (log): association results
        logger.debug(
            f"Frame {self.frame_count}: "
            f"{len(matches)} associations, "
            f"{len(unmatched_tracks)} unmatched tracks, "
            f"{len(unmatched_dets)} new detections"
        )

        # Step 5: Update matched tracks
        for track_idx, det_idx in matches:
            active_tracks[track_idx].update(triangulated[det_idx], self.config)

        # Step 6: Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            active_tracks[track_idx].mark_missed(self.config)

        # Step 7: Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = Track(
                global_id=self.id_manager.next_id(),
                initial_position=triangulated[det_idx].position,
                config=self.config,
            )
            new_track.last_person = triangulated[det_idx]
            new_track.hit_streak = 1
            self.tracks.append(new_track)

        # Step 8: Count retired tracks (DELETED state) and accumulate total
        retired_this_frame = sum(1 for t in self.tracks if t.is_deleted())
        self.total_retired += retired_this_frame

        # Step 9: Remove DELETED tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Step 10: Build TrackingResult
        confirmed_tracked = [
            t.to_tracked_person()
            for t in self.tracks
            if t.state == TrackState.CONFIRMED
        ]
        num_tentative = sum(1 for t in self.tracks if t.state == TrackState.TENTATIVE)
        num_coasting = sum(
            1 for t in self.tracks
            if t.state == TrackState.CONFIRMED and t.time_since_update > 0
        )

        # Step 11 (log): frame summary
        logger.debug(
            f"Frame {self.frame_count} complete: "
            f"{len(confirmed_tracked)} confirmed, "
            f"{num_tentative} tentative, "
            f"{num_coasting} coasting, "
            f"{retired_this_frame} retired"
        )

        return TrackingResult(
            frame_num=reconstruction_result.frame_num,
            tracked_persons=confirmed_tracked,
            single_view_persons=single_view,
            num_tentative=num_tentative,
            num_coasting=num_coasting,
            num_retired=retired_this_frame,
        )

    def reset(self) -> None:
        """Clear all state. For testing."""
        self.tracks = []
        self.id_manager.reset()
        self.frame_count = 0
        self.total_retired = 0


def _make_person(person_id: int, x: float, y: float, z: float) -> Person3D:
    """Helper for __main__ validation."""
    return Person3D(
        person_id=person_id,
        position=np.array([x, y, z], dtype=np.float64),
        num_views=2,
        source_detections=[(0, person_id), (1, person_id)],
        is_triangulated=True,
    )


def _make_frame(frame_num: int, persons: list[Person3D]) -> ReconstructionResult:
    """Helper for __main__ validation."""
    return ReconstructionResult(
        frame_num=frame_num,
        persons=persons,
        num_triangulated_points=len(persons),
        num_rejected_points=0,
    )


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    print("=" * 60)
    print("PersonTracker Synthetic Validation")
    print("=" * 60)

    tracker = PersonTracker()

    # Frame 1: 2 persons at distinct positions
    frame_1 = _make_frame(1, [
        _make_person(0, 0.0, 0.0, 5.0),
        _make_person(1, 3.0, 0.0, 5.0),
    ])
    result_1 = tracker.update(frame_1)
    print(f"Frame 1: {result_1}")

    # Frame 2: slight movement
    frame_2 = _make_frame(2, [
        _make_person(0, 0.1, 0.0, 5.0),
        _make_person(1, 3.1, 0.0, 5.0),
    ])
    result_2 = tracker.update(frame_2)
    print(f"Frame 2: {result_2}")

    # Frame 3: slight movement again (n_init=3, should confirm this frame)
    frame_3 = _make_frame(3, [
        _make_person(0, 0.2, 0.0, 5.0),
        _make_person(1, 3.2, 0.0, 5.0),
    ])
    result_3 = tracker.update(frame_3)
    print(f"Frame 3: {result_3}")

    # Frame 4: Person 1 (at origin) disappears, only person 2 remains
    frame_4 = _make_frame(4, [
        _make_person(1, 3.3, 0.0, 5.0),
    ])
    result_4 = tracker.update(frame_4)
    print(f"Frame 4: {result_4}")

    # Frame 5: Still only person 2
    frame_5 = _make_frame(5, [
        _make_person(1, 3.4, 0.0, 5.0),
    ])
    result_5 = tracker.update(frame_5)
    print(f"Frame 5: {result_5}")

    print()
    print("Running assertions...")

    # 1. Global ID stability - same person keeps same ID across frames
    person_at_origin_ids = []
    for result in [result_1, result_2, result_3]:
        for tp in result.tracked_persons:
            if abs(tp.position[0]) < 0.5 and abs(tp.position[1]) < 0.5:
                person_at_origin_ids.append(tp.global_id)
    assert len(set(person_at_origin_ids)) == 1, \
        f"Person at origin got different IDs: {person_at_origin_ids}"

    # 2. State transitions - new tracks start TENTATIVE, confirmed after n_init=3 frames
    assert result_1.num_tentative >= 1, "New tracks should start as TENTATIVE"
    assert len(result_3.tracked_persons) >= 1, \
        "Tracks should be CONFIRMED after 3 frames (n_init=3)"

    # 3. Coasting - person 1 disappears in frames 4-5, should cause coasting
    assert result_4.num_coasting >= 1 or result_5.num_coasting >= 1, \
        "Missing person should cause coasting"

    # 4. Single-view pass-through
    tracker.reset()
    single_view_result = tracker.update(ReconstructionResult(
        frame_num=10,
        persons=[Person3D(
            person_id=99,
            position=None,
            num_views=1,
            source_detections=[(1, 0)],
            is_triangulated=False,
        )],
    ))
    assert len(single_view_result.single_view_persons) == 1, \
        "Single-view persons not passed through"
    assert len(single_view_result.tracked_persons) == 0, \
        "Single-view should not be tracked (no position)"

    print()
    print("All tracking validation tests passed!")
    print("  - Global ID stability verified")
    print("  - State transitions (TENTATIVE -> CONFIRMED) working")
    print("  - Coasting behavior on missed detections working")
    print("  - Single-view pass-through working")
