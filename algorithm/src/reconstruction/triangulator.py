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
        # Index 0 in all 3 lists = same detection.
        # the 2D pixel center of each detection
        points_2d: list[tuple[float, float]] = []
        # the P matrix for that drone
        projection_matrices: list[np.ndarray] = []
        # (drone_id, local_id) for tracking
        detection_ids: list[tuple[int, int]] = []

        for drone_id, local_id in match_group.detections:
            detection = detection_sets[drone_id].detections[local_id]
            center = detection.bbox.center

            # flip for mirrored pics
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

        # need at least 2 views to triangulate — return empty and signal fallback
        if len(points_2d) < 2:
            return [], True

        # tracks which indices are still valid after pruning.
        active = list(range(len(points_2d)))

        # change to np.array for fast grab
        # EXAMPLE: act_Ps = all_Ps[active]  # active = [0, 1, 3] → grabs rows 0, 1, 3
        ps_list = []
        for k in range(len(points_2d)):
            ps_list.append(projection_matrices[k].astype(np.float64))
        # stacks all P matrices into one (N, 3, 4) array one P per row
        all_Ps = np.stack(ps_list, axis=0)
        all_pts = np.array(points_2d, dtype=np.float64)  # (N, 2)

        # ========== find and remove bad views ===============
        # Iterative pruning: drop the worst view while >= 3 views remain
        while len(active) >= 3:
            # tracks how many pairs that view was "bad" in
            bad_counts = np.zeros(len(active), dtype=np.int32)
            # counts how many pairs produced a valid X (not NaN)
            valid_pairs = 0

            # Slice active views once per pruning round
            # so we reproject only with active
            act_Ps = all_Ps[active]  # (A, 3, 4)
            act_pts = all_pts[active]  # (A, 2)

            # for each unique pair of active views triangulate X and reproject into all active cameras to count bad views
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    ai, aj = active[i], active[j]
                    X = self._triangulate_two_view_whitebox(
                        points_2d[ai],
                        points_2d[aj],
                        projection_matrices[ai],
                        projection_matrices[aj],
                    )

                    # skip degenerate triangulation (NaN or inf)
                    if not np.isfinite(X).all():
                        continue

                    # Vectorized per-view reprojection: (A, 3, 4) @ (4,) → (A, 3)
                    X4 = np.append(X, 1.0)  # (4,) add 1 for P multiplication
                    # project X into all active cameras at once, gives (A, 3) = one [u,v,w] per camera
                    proj = act_Ps @ X4  # (A, 3)
                    # extract w (depth) from each projection
                    depths = proj[:, 2]  # (A,)
                    # flag cameras where depth≈0 (degenerate, can't divide)
                    bad_depth = np.abs(depths) < EPS_DENOM
                    # for each camera: if depth was bad → use inf, otherwise use u/w (real pixel x). Same for y.
                    with np.errstate(divide="ignore", invalid="ignore"):
                        px = np.where(bad_depth, np.inf, proj[:, 0] / depths)
                        py = np.where(bad_depth, np.inf, proj[:, 1] / depths)
                    # same as sqrt(dx² + dy²) but vectorized.
                    per_view = np.hypot(px - act_pts[:, 0], py - act_pts[:, 1])

                    # how many pairs produced a valid X.
                    valid_pairs += 1
                    # produces a boolean array, one value per active camera. True where error was too high.
                    bad_counts += (
                        per_view > self.config.max_reprojection_error
                    ).astype(np.int32)

            # If every pair produced a degenerate X (all NaN)
            if valid_pairs == 0:
                break

            # ratio 0.0-1.0: how often each camera was bad across all valid pairs
            bad_ratios = bad_counts.astype(np.float64) / float(valid_pairs)
            # index of the camera with the highest bad ratio
            worst_local = int(np.argmax(bad_ratios))
            # actual ratio value of that worst camera
            worst_ratio = float(bad_ratios[worst_local])

            # no camera is bad enough to prune — stop and triangulate with current active views
            if worst_ratio < self.config.prune_bad_ratio_threshold:
                break
            # else remove
            logger.debug(
                "Pruning view index %d (drone=%d): bad_ratio=%.2f",
                active[worst_local],
                detection_ids[active[worst_local]][0],
                worst_ratio,
            )
            del active[worst_local]

            if len(active) < 2:
                return [], True

        # triangulate all surviving pairs for real and save the results as Point3D objects.
        # Triangulate all C(N,2) pairs from surviving active views
        active_points = []
        active_Ps = []
        active_det_ids = []
        for k in active:
            active_points.append(points_2d[k])
            active_Ps.append(projection_matrices[k])
            active_det_ids.append(detection_ids[k])

        result_points: list[Point3D] = []
        
        # for each surviving pair triangulate X, validate reprojection error, save as Point3D
        for i in range(len(active_points)):
            for j in range(i + 1, len(active_points)):
                X = self._triangulate_two_view_whitebox(
                    active_points[i],
                    active_points[j],
                    active_Ps[i],
                    active_Ps[j],
                )
                
                # skip degenerate triangulation (NaN or inf)
                if not np.isfinite(X).all():
                    continue
                # validate quality — average pixel error across all active views
                err = self._compute_reprojection_error(X, active_points, active_Ps)
                
                # reject if error is invalid or too high
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

            # Compute L2 Euclidean distance to observed point
            dx = projected_x - pt_2d[0]
            dy = projected_y - pt_2d[1]
            error = np.sqrt(dx**2 + dy**2)

            errors.append(error)

        # Return mean error
        return float(np.mean(errors))
