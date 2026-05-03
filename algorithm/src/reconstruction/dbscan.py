"""
WhiteBox DBSCAN — clusters triangulated 3D points into unique persons.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

# points that belong to no cluster
NOISE = -1

# point not yet processed
_UNVISITED = -2


class WhiteBoxDBSCAN:
    """
    DBSCAN clustering.

    eps: max distance between two points to be considered neighbors (meters in 3D space)
    min_samples: minimum neighbors a point needs to be a core point (including itself)
    """

    def __init__(self, eps: float, min_samples: int) -> None:
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Cluster points and return label array.

        X: (N, D) array of N points in D-dimensional space (D=3 for x,y,z meters)
        Returns: (N,) array — cluster ID per point (0,1,2...) or -1 for noise
        """
        # total number of points
        n = X.shape[0]

        # initialize all points as unvisited
        labels = np.full(n, _UNVISITED, dtype=np.intp)

        # next cluster label to assign
        cluster_id = 0

        # for each unvisited point — decide if it's a core point or noise
        for i in range(n):
            # skip already processed points
            if labels[i] != _UNVISITED:
                continue

            # find all points within eps distance of point i (including itself)
            neighbors = self._get_neighbors(X, i)

            # not enough neighbors — mark as noise for now (may be absorbed later by a core point's BFS)
            if len(neighbors) < self.min_samples:
                labels[i] = NOISE
            else:
                # core point — start a new cluster and expand via BFS
                self._expand_cluster(X, labels, i, neighbors, cluster_id)
                cluster_id += 1

        # defensive cleanup (shouldn't happen)
        labels[labels == _UNVISITED] = NOISE

        logger.debug(
            "DBSCAN: %d points -> %d clusters, %d noise  (eps=%.2f, min_samples=%d)",
            n,
            cluster_id,
            int(np.sum(labels == NOISE)),
            self.eps,
            self.min_samples,
        )
        return labels

    def _get_neighbors(self, X: np.ndarray, idx: int) -> np.ndarray:
        """
        Return indices of all points within eps of X[idx], including itself.
        Includes self because DBSCAN counts a point as its own neighbor.
        """
        # diff = X - X[idx] subtract point idx from every row:
        diff = X - X[idx]
        # L2 norm per row, scalar distance per point, [x,y,z]->distance each row
        dist = np.linalg.norm(diff, axis=1)
        # return indices where distance is within eps
        return np.where(dist <= self.eps)[0]

    def _expand_cluster(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        core_idx: int,
        neighbors: np.ndarray,
        cluster_id: int,
    ) -> None:
        """
        BFS from core_idx — assign cluster_id to all reachable points.
        Uses deque for O(1) popleft (list.pop(0) is O(N)).
        """
        # assign seed core point to this cluster
        labels[core_idx] = cluster_id

        # BFS queue starts with all neighbors of the seed
        queue: deque[int] = deque(neighbors)

        # for each neighbor in the queue — assign to cluster, expand if also a core point
        while queue:
            idx = queue.popleft()

            # seed already labeled — skip
            if idx == core_idx:
                continue

            if labels[idx] == NOISE:
                # was noise but reachable from core — becomes border point, no further expansion
                labels[idx] = cluster_id
                continue

            # already assigned to a cluster — skip
            if labels[idx] != _UNVISITED:
                continue

            # unvisited — assign to this cluster
            labels[idx] = cluster_id

            # check if this point is also a core point
            new_neighbors = self._get_neighbors(X, idx)
            if len(new_neighbors) >= self.min_samples:
                # also a core point — add its neighbors to expand the cluster further
                queue.extend(new_neighbors)
