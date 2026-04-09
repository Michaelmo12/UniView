"""
WhiteBox DBSCAN
===============
Density-Based Spatial Clustering of Applications with Noise.

This is a from-scratch implementation of the DBSCAN algorithm (Ester et al., 1996).
It is drop-in compatible with sklearn's DBSCAN.fit_predict() interface:

    labels = WhiteBoxDBSCAN(eps=0.5, min_samples=2).fit_predict(X)

Why write it from scratch?
--------------------------
The pipeline uses DBSCAN to cluster triangulated 3D points (~2-20 points per frame).
At this scale sklearn's overhead (input validation, compiled C extensions) is comparable
to the actual work. More importantly, having an explicit implementation makes every step
of the algorithm inspectable and explainable.

Complexity: O(N²) — for each point we scan all N points to find neighbors.
At N ≤ 20 this is negligible (<0.1 ms). sklearn uses KD-tree / ball-tree to achieve
O(N log N) but the constant overhead is larger than the gain at N < ~1000.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

# Sentinel label used by DBSCAN to mark points that belong to no cluster
NOISE = -1

# Internal sentinel: point has been visited but not yet assigned (used during BFS)
_UNVISITED = -2


class WhiteBoxDBSCAN:
    """
    DBSCAN clustering — explicit step-by-step implementation.

    Parameters
    ----------
    eps : float
        Maximum Euclidean distance between two points for them to be considered
        "neighbors". In our pipeline this is measured in meters of 3D world space.
        e.g. eps=0.5 means two triangulated positions must be within 0.5 m of each
        other to be grouped into the same person cluster.

    min_samples : int
        A point is a "core point" (anchor of a cluster) only if it has at least
        `min_samples` neighbors within distance `eps` (including itself).
        With min_samples=2: even a pair of nearby points forms a valid cluster.
        With min_samples=3: isolated pairs become noise instead.
    """

    def __init__(self, eps: float, min_samples: int) -> None:
        self.eps = eps
        self.min_samples = min_samples

    # ------------------------------------------------------------------
    # Public API — same signature as sklearn DBSCAN.fit_predict()
    # ------------------------------------------------------------------

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Cluster the input points and return a label for each point.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            N points in D-dimensional space. In our pipeline D=3 (x, y, z meters).

        Returns
        -------
        labels : np.ndarray, shape (N,)
            Integer cluster ID for each point (0, 1, 2 ...).
            Points that belong to no cluster are labeled -1 (NOISE).

        Algorithm overview (Ester et al. 1996):
        -----------------------------------------
        For each unvisited point P:
          1. Mark P as visited.
          2. Find all neighbors of P within distance eps  →  N(P).
          3. If |N(P)| < min_samples  →  P is noise for now (may be absorbed later).
          4. Otherwise P is a core point  →  start a new cluster C:
               a. Add P to C.
               b. BFS: for each neighbor Q of P (that hasn't started a new cluster):
                    - If Q is unvisited: visit Q, find N(Q).
                      If |N(Q)| >= min_samples, N(Q) is added to the BFS queue
                      (Q is also a core point, so it "expands" the cluster further).
                    - Either way, assign Q to cluster C.
        """
        n = X.shape[0]

        # labels[i] == _UNVISITED means point i has not been processed yet.
        # After the algorithm, every label is either NOISE (-1) or a cluster ID >= 0.
        labels = np.full(n, _UNVISITED, dtype=np.intp)

        cluster_id = 0  # next cluster label to assign

        for i in range(n):
            if labels[i] != _UNVISITED:
                # Already processed (assigned to a cluster or marked noise)
                continue

            # Step 1: find all neighbors of point i
            neighbors = self._get_neighbors(X, i)

            # Step 2: is point i a core point?
            if len(neighbors) < self.min_samples:
                # Not a core point — tentatively noise.
                # It might later be absorbed into another cluster's BFS,
                # but since labels[i] is still _UNVISITED at that point,
                # the BFS will re-assign it. Mark as noise for now.
                labels[i] = NOISE
            else:
                # Core point — start a new cluster and expand it via BFS
                self._expand_cluster(X, labels, i, neighbors, cluster_id)
                cluster_id += 1

        # Replace any remaining _UNVISITED labels with NOISE
        # (shouldn't happen, but defensive)
        labels[labels == _UNVISITED] = NOISE

        logger.debug(
            "DBSCAN: %d points → %d clusters, %d noise  (eps=%.2f, min_samples=%d)",
            n, cluster_id, int(np.sum(labels == NOISE)), self.eps, self.min_samples,
        )
        return labels

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_neighbors(self, X: np.ndarray, idx: int) -> np.ndarray:
        """
        Return the indices of all points within distance `eps` of X[idx],
        INCLUDING the point itself (standard DBSCAN definition).

        We compute all N distances in one vectorized operation:
            diff[j] = X[j] - X[idx]          (shape: N×D)
            dist[j] = ||diff[j]||₂            (Euclidean norm, shape: N)

        Then return indices where dist <= eps.

        Why include self?
        -----------------
        DBSCAN counts a point as its own neighbor, so the neighborhood size
        of an isolated point is 1 (just itself). With min_samples=2 that means
        a truly isolated point (no other point within eps) is noise, which is
        the correct behavior.
        """
        diff = X - X[idx]                       # (N, D) — broadcast subtraction
        dist = np.linalg.norm(diff, axis=1)     # (N,)   — L2 norm per row
        return np.where(dist <= self.eps)[0]    # indices where dist is within eps

    def _expand_cluster(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        core_idx: int,
        neighbors: np.ndarray,
        cluster_id: int,
    ) -> None:
        """
        BFS expansion: assign `cluster_id` to `core_idx` and all points
        reachable from it through a chain of core points.

        We use a deque (double-ended queue) instead of a plain list because
        deque.popleft() is O(1). list.pop(0) is O(N) — it shifts every element
        left — which makes the overall algorithm O(N³) in the worst case.

        BFS vs DFS:
        -----------
        Both work correctly for DBSCAN. BFS (queue / popleft) expands the cluster
        level-by-level from the seed point, which gives a predictable traversal
        order and identical results to the reference sklearn implementation for
        the same tie-breaking conventions.

        Parameters
        ----------
        core_idx   : index of the seed core point
        neighbors  : indices of all points within eps of core_idx (from _get_neighbors)
        cluster_id : integer label to assign to this cluster
        """
        # Assign the seed core point to this cluster
        labels[core_idx] = cluster_id

        # BFS queue — start with all neighbors of the seed
        queue: deque[int] = deque(neighbors)

        while queue:
            idx = queue.popleft()

            if idx == core_idx:
                # The seed is already labeled; skip to avoid redundant work
                continue

            if labels[idx] == NOISE:
                # This point was previously classified as noise (it had < min_samples
                # neighbors when we first visited it) but it IS reachable from this
                # core point, so it becomes a border point of this cluster.
                labels[idx] = cluster_id
                # Note: border points do NOT expand the cluster further (they are
                # not core points), so we do NOT add their neighbors to the queue.
                continue

            if labels[idx] != _UNVISITED:
                # Already assigned to this cluster (or a previously finished cluster).
                # Can happen because multiple core points may share neighbors.
                continue

            # Point is unvisited — assign to this cluster
            labels[idx] = cluster_id

            # Check if this point is also a core point
            new_neighbors = self._get_neighbors(X, idx)
            if len(new_neighbors) >= self.min_samples:
                # idx is a core point → its neighbors are also density-reachable
                # from the current cluster. Add them to the BFS queue.
                queue.extend(new_neighbors)
            # If idx is not a core point it's a border point — already labeled,
            # no further expansion.
