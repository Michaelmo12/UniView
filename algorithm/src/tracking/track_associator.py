"""
TrackAssociator

Associates existing tracks to new detections using the Hungarian algorithm
(scipy linear_sum_assignment) on a Euclidean distance cost matrix.
"""

import logging

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.config.settings import TrackingConfig
from src.reconstruction.models import Person3D

logger = logging.getLogger(__name__)


class TrackAssociator:
    """
    Associates tracks to detections using optimal Hungarian assignment.

    Builds a Euclidean distance cost matrix between track predicted positions
    and detection positions, then solves via linear_sum_assignment. Matches
    exceeding max_distance are discarded as unmatched.
    """

    def __init__(self, config: TrackingConfig) -> None:
        self.config = config

    def associate(
        self,
        tracks: list,
        detections: list[Person3D],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Associate tracks to detections using Hungarian algorithm on Euclidean distance.

        Args:
            tracks: List of Track objects (must expose .predicted_position np.ndarray (3,)).
            detections: List of Person3D detections with valid .position arrays.

        Returns:
            matches: List of (track_idx, detection_idx) pairs within max_distance.
            unmatched_tracks: Track indices with no valid match this frame.
            unmatched_dets: Detection indices that were not matched to any track.
        """
        # Guard: empty inputs if no detections or tracks exists
        if len(tracks) == 0 or len(detections) == 0:
            return ([], list(range(len(tracks))), list(range(len(detections))))

        # Build cost matrix (Euclidean distance in meters)
        # finds the best matches between tracks and detections based on predicted position vs detected position
        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                cost_matrix[i, j] = np.linalg.norm(
                    track.predicted_position - detection.position # cost is the distance between where the Kalman says this track should be now, and where the new detection actually is.
                )
        

        # Solve assignment problem (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter matches by distance threshold
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= self.config.max_distance:
                matches.append((r, c))

        # Collect unmatched indices
        matched_track_indices = set()
        for track_idx, _ in matches:
            matched_track_indices.add(track_idx)

        matched_det_indices = set()
        for _, det_idx in matches:
            matched_det_indices.add(det_idx)

        unmatched_tracks = []
        for i in range(len(tracks)):
            if i not in matched_track_indices:
                unmatched_tracks.append(i)

        unmatched_dets = []
        for i in range(len(detections)):
            if i not in matched_det_indices:
                unmatched_dets.append(i)

        return matches, unmatched_tracks, unmatched_dets
