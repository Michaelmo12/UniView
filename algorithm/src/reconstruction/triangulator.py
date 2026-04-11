"""
Triangulator

Converts matched 2D detections across cameras into 3D world positions using
a whitebox Direct Linear Transform (DLT) via SVD. Validates triangulation
quality via reprojection error.

Key Class:
- Triangulator: Triangulates MatchGroup into Point3D with error validation
"""

import logging
import numpy as np

from src.config.settings import ReconstructionConfig
from src.config.settings import settings
from src.fusion.models import MatchGroup
from src.detection.models import DetectionSet
from src.ingestion.models import SynchronizedFrameSet
from src.reconstruction.models import Point3D

logger = logging.getLogger(__name__)

# Numerical stability threshold for homogeneous-coordinate divisions.
EPS_DENOM = 1e-12


class Triangulator:
    """
    Triangulates matched 2D detections into 3D world positions.

    Uses a whitebox DLT (SVD-based) for 2-view triangulation. Validates
    results via reprojection error and rejects poor triangulations.

    Constructor takes ReconstructionConfig with max_reprojection_error threshold.
    """

    def __init__(self, config: ReconstructionConfig):
        """
        Initialize triangulator with configuration.

        Args:
            config: ReconstructionConfig with max_reprojection_error threshold
        """
        self.config = config

    def triangulate_match_group_robust(
        self,
        match_group: MatchGroup,
        detection_sets: dict[int, DetectionSet],
        sync_set: SynchronizedFrameSet,
    ) -> tuple[list[Point3D], bool]:
        """
        Robust triangulation with iterative bad-view pruning.

        Before triangulating all C(N,2) pairs, iteratively removes the single
        view that is consistently bad across the most pairwise tests — as long
        as at least 3 views remain and the worst view exceeds
        config.prune_bad_ratio_threshold.  Falls back once only 2 views remain
        or no view is bad enough to prune.

        Args:
            match_group: MatchGroup with detections list [(drone_id, local_id), ...]
            detection_sets: Dict mapping drone_id to DetectionSet
            sync_set: SynchronizedFrameSet with calibration data

        Returns:
            (points, used_fallback)
            - points: list of Point3D from surviving views (may be empty)
            - used_fallback: True when robust pruning yielded no valid points
        """
        points_2d: list[tuple[float, float]] = []
        projection_matrices: list[np.ndarray] = []
        detection_ids: list[tuple[int, int]] = []

        for drone_id, local_id in match_group.detections:
            detection = detection_sets[drone_id].detections[local_id]
            center = detection.bbox.center

            if settings.geometry.flip_x_for_geometry:
                frame_width = (
                    settings.geometry.image_width_override
                    if settings.geometry.image_width_override > 0
                    else sync_set.frames[drone_id].frame.shape[1]
                )
                center = (float(frame_width) - float(center[0]), float(center[1]))

            points_2d.append(center)
            projection_matrices.append(
                sync_set.frames[drone_id].calibration.projection_matrix
            )
            detection_ids.append((drone_id, local_id))

        if len(points_2d) < 2:
            return [], True

        active = list(range(len(points_2d)))

        # Pre-stack all projection matrices as a single (N, 3, 4) float64 array
        # and observed 2D points as (N, 2) — built once, sliced per pruning round.
        all_Ps = np.stack(
            [projection_matrices[k].astype(np.float64) for k in range(len(points_2d))],
            axis=0,
        )  # (N, 3, 4)
        all_pts = np.array(points_2d, dtype=np.float64)  # (N, 2)

        # Iterative pruning: drop the worst view while >= 3 views remain
        while len(active) >= 3:
            bad_counts = np.zeros(len(active), dtype=np.int32)
            valid_pairs = 0

            # Slice active views once per pruning round
            act_Ps  = all_Ps[active]   # (A, 3, 4)
            act_pts = all_pts[active]  # (A, 2)

            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    ai, aj = active[i], active[j]
                    X = self._triangulate_two_view_whitebox(
                        points_2d[ai], points_2d[aj],
                        projection_matrices[ai], projection_matrices[aj],
                    )
                    if not np.isfinite(X).all():
                        continue

                    # Vectorized per-view reprojection: (A, 3, 4) @ (4,) → (A, 3)
                    X4 = np.append(X, 1.0)                      # (4,)
                    proj = act_Ps @ X4                           # (A, 3)
                    depths = proj[:, 2]                          # (A,)
                    bad_depth = np.abs(depths) < EPS_DENOM
                    with np.errstate(divide="ignore", invalid="ignore"):
                        px = np.where(bad_depth, np.inf, proj[:, 0] / depths)
                        py = np.where(bad_depth, np.inf, proj[:, 1] / depths)
                    per_view = np.hypot(px - act_pts[:, 0], py - act_pts[:, 1])

                    valid_pairs += 1
                    bad_counts += (per_view > self.config.max_reprojection_error).astype(np.int32)

            if valid_pairs == 0:
                break

            bad_ratios = bad_counts.astype(np.float64) / float(valid_pairs)
            worst_local = int(np.argmax(bad_ratios))
            worst_ratio = float(bad_ratios[worst_local])

            if worst_ratio < self.config.prune_bad_ratio_threshold:
                break

            logger.debug(
                "Pruning view index %d (drone=%d): bad_ratio=%.2f",
                active[worst_local],
                detection_ids[active[worst_local]][0],
                worst_ratio,
            )
            del active[worst_local]

            if len(active) < 2:
                return [], True

        # Triangulate all C(N,2) pairs from surviving active views
        active_points = [points_2d[k] for k in active]
        active_Ps = [projection_matrices[k] for k in active]
        active_det_ids = [detection_ids[k] for k in active]

        result_points: list[Point3D] = []
        for i in range(len(active_points)):
            for j in range(i + 1, len(active_points)):
                X = self._triangulate_two_view_whitebox(
                    active_points[i], active_points[j],
                    active_Ps[i], active_Ps[j],
                )
                if not np.isfinite(X).all():
                    continue
                err = self._compute_reprojection_error(X, active_points, active_Ps)
                if not np.isfinite(err) or err > self.config.max_reprojection_error:
                    continue
                result_points.append(
                    Point3D(
                        position=X.astype(np.float32),
                        reprojection_error=err,
                        source_detections=[active_det_ids[i], active_det_ids[j]],
                        match_group_id=match_group.group_id,
                    )
                )

        if not result_points:
            return [], True

        return result_points, False

    """The current implementation uses `cv2.triangulatePoints`, which is OpenCV's optimized
    implementation of the Direct Linear Transform algorithm. While this is efficient and
    well-tested, it is a "black box" — the internal SVD solution is hidden.

    For full transparency and academic purposes, an alternative White Box implementation
    can replace `_triangulate_two_view`"""

    def _triangulate_two_view_whitebox(
        self,
        pt1: tuple[float, float],
        pt2: tuple[float, float],
        P1: np.ndarray,
        P2: np.ndarray,
    ) -> np.ndarray:
        """
        White Box DLT implementation using explicit SVD.

        Builds the 4×4 system AX = 0 from projection equations:
          x1(P1[2]·X) = P1[0]·X  →  x1·P1[2] - P1[0] = 0
          y1(P1[2]·X) = P1[1]·X  →  y1·P1[2] - P1[1] = 0
          x2(P2[2]·X) = P2[0]·X  →  x2·P2[2] - P2[0] = 0
          y2(P2[2]·X) = P2[1]·X  →  y2·P2[2] - P2[1] = 0

        Solves via SVD: X is the right singular vector  corresponding to
        the smallest singular value (last row of Vt).
        """
        x1, y1 = pt1
        x2, y2 = pt2

        # Build matrix A (4 rows × 4 cols)
        A = np.array(
            [
                x1 * P1[2, :] - P1[0, :],  # equation from camera 1,    x-coordinate
                y1 * P1[2, :] - P1[1, :],  # equation from camera 1,    y-coordinate
                x2 * P2[2, :] - P2[0, :],  # equation from camera 2,    x-coordinate
                y2 * P2[2, :] - P2[1, :],  # equation from camera 2,    y-coordinate
            ],
            dtype=np.float64,
        )

        # Singular Value Decomposition: A = U·S·Vt
        # Solution to AX = 0 is the last column of V (last row of   Vt)
        _, _, Vt = np.linalg.svd(A)
        X_homogeneous = Vt[-1, :]  # shape (4,): [X, Y, Z, W]

        # Convert from homogeneous to Cartesian coordinates.
        # W ~= 0 means point at infinity / degenerate triangulation.
        w = float(X_homogeneous[3])
        if abs(w) < EPS_DENOM:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        point_3d = X_homogeneous[:3] / w  # [X/W, Y/W, Z/W]

        return point_3d.astype(np.float64)

    def _compute_reprojection_error(
        self,
        point_3d: np.ndarray,
        points_2d: list[tuple[float, float]],
        projection_matrices: list[np.ndarray],
    ) -> float:
        """
        Compute average reprojection error for a 3D point.

        Projects the 3D point back to each camera and measures pixel distance
        to the observed 2D point.

        Args:
            point_3d: (3,) world coordinates
            points_2d: List of (x, y) observed points in each camera
            projection_matrices: List of (3, 4) projection matrices

        Returns:
            Average L2 pixel distance across all views
        """
        # Convert to homogeneous coordinates [x, y, z, 1]
        point_4d = np.append(point_3d, 1.0)

        errors = []
        for pt_2d, P in zip(points_2d, projection_matrices):
            # Project to image: [u, v, w] = P @ [X, Y, Z, 1]
            projected_homogeneous = P @ point_4d

            # Near-zero depth makes pixel reprojection undefined.
            depth = float(projected_homogeneous[2])
            if abs(depth) < EPS_DENOM:
                return float("inf")

            # Convert to pixel coordinates: (u/w, v/w)
            projected_x = projected_homogeneous[0] / depth
            projected_y = projected_homogeneous[1] / depth

            # Compute L2 distance to observed point
            dx = projected_x - pt_2d[0]
            dy = projected_y - pt_2d[1]
            error = np.sqrt(dx**2 + dy**2)

            errors.append(error)

        # Return mean error
        return float(np.mean(errors))

