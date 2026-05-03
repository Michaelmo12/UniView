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

import logging

import numpy as np

from src.config.settings import TrackingConfig, settings
from src.reconstruction.models import Person3D, ReconstructionResult
from src.tracking.global_id_manager import GlobalIDManager
from src.tracking.kalman_filter import PersonKalmanFilter
from src.tracking.models import TrackState, TrackedPerson, TrackingResult
from src.tracking.track_associator import TrackAssociator

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
        self.frame_count: int = 0 # Counter for logging and debugging

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
        triangulated = []
        single_view = []
        for person in reconstruction_result.persons:
            if person.is_triangulated and person.position is not None:
                triangulated.append(person)
            else:
                single_view.append(person)

        # Step 3: Associate triangulated detections to existing non-deleted tracks
        active_tracks = []
        for track in self.tracks:
            if not track.is_deleted():
                active_tracks.append(track)
                
        # This is where the Hungarian algorithm finds the best matches between predicted track positions and new detections, based on the cost matrix of distances. and thresholding by max_distance.
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
            #uses track.update() to update the Kalman filter with the new measurement, and resets the time_since_update and hit_streak counters. If the track was TENTATIVE and has now reached n_init hits, it transitions to CONFIRMED.
            active_tracks[track_idx].update(triangulated[det_idx], self.config)

        # Step 6: Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            #uses track.mark_missed() to increment the time_since_update counter and reset hit_streak. If a TENTATIVE track is missed, it transitions to DELETED immediately. If a CONFIRMED track exceeds max_age without an update, it also transitions to DELETED.
            active_tracks[track_idx].mark_missed(self.config)

        # Step 7: Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = Track(
                global_id=self.id_manager.next_id(),
                initial_position=triangulated[det_idx].position,
                config=self.config,
            )
            new_track.update(triangulated[det_idx], self.config)
            self.tracks.append(new_track)

        # for logging
        # Step 8: Count retired tracks (DELETED state) and accumulate total 
        retired_this_frame = 0
        for track in self.tracks:
            if track.is_deleted():
                retired_this_frame += 1

        # Step 9: Remove DELETED tracks
        remaining_tracks = []
        for track in self.tracks:
            if not track.is_deleted():
                remaining_tracks.append(track)
        self.tracks = remaining_tracks

        # Step 10: Build TrackingResult
        confirmed_tracked = []
        for track in self.tracks:
            if track.state == TrackState.CONFIRMED:
                confirmed_tracked.append(track.to_tracked_person())
        num_tentative = 0
        for track in self.tracks:
            if track.state == TrackState.TENTATIVE:
                num_tentative += 1
        num_coasting = 0
        for track in self.tracks:
            if track.state == TrackState.CONFIRMED and track.time_since_update > 0:
                num_coasting += 1

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
