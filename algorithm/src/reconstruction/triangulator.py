"""
Triangulator

Converts matched 2D detections across cameras into 3D world positions using
Direct Linear Transform (DLT) via cv2.triangulatePoints. Validates triangulation
quality via reprojection error.

Key Class:
- Triangulator: Triangulates MatchGroup into Point3D with error validation
"""

import logging
import numpy as np
import cv2

from src.config.settings import ReconstructionConfig
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

    Uses cv2.triangulatePoints (DLT algorithm) for 2-view or multi-view
    triangulation. Validates results via reprojection error and rejects
    poor triangulations.

    Constructor takes ReconstructionConfig with max_reprojection_error threshold.
    """

    def __init__(self, config: ReconstructionConfig):
        """
        Initialize triangulator with configuration.

        Args:
            config: ReconstructionConfig with max_reprojection_error threshold
        """
        self.config = config

    def triangulate_match_group(
        self,
        match_group: MatchGroup,
        detection_sets: dict[int, DetectionSet],
        sync_set: SynchronizedFrameSet,
    ) -> list[Point3D]:
        """
        Triangulate a match group into raw pairwise Point3Ds.

        Extracts 2D bbox centers and projection matrices for each detection in
        the match group, then triangulates all C(N,2) pairs. Each pair that
        passes reprojection error produces one Point3D. The median collapse is
        intentionally removed — DBSCAN in PersonClusterer handles consolidation.

        Args:
            match_group: MatchGroup with detections list [(drone_id, local_id), ...]
            detection_sets: Dict mapping drone_id to DetectionSet
            sync_set: SynchronizedFrameSet with calibration data

        Returns:
            List of Point3D, one per valid pairwise triangulation (may be empty)
        """
        # Extract 2D points and projection matrices (parallel lists)
        points_2d = []
        projection_matrices = []
        detection_ids = []  # keep (drone_id, local_id) aligned with points_2d

        for drone_id, local_id in match_group.detections:
            detection = detection_sets[drone_id].detections[local_id]
            center = detection.bbox.center  # (cx, cy) tuple
            points_2d.append(center)
            projection_matrices.append(
                sync_set.frames[drone_id].calibration.projection_matrix
            )
            detection_ids.append((drone_id, local_id))

        num_views = len(points_2d)

        if num_views < 2:
            logger.warning(
                "Match group has < 2 views, cannot triangulate: %s",
                match_group.detections,
            )
            return []

        # Triangulate all C(N,2) pairs, keep those below reprojection threshold
        result_points = []

        for i in range(num_views):
            for j in range(i + 1, num_views):
                point_3d = self._triangulate_two_view_whitebox(
                    points_2d[i],
                    points_2d[j],
                    projection_matrices[i],
                    projection_matrices[j],
                )

                # Skip degenerate triangulations (W=0 produces nan/inf)
                if not np.isfinite(point_3d).all():
                    logger.debug(
                        "Rejecting pair (%d,%d): degenerate triangulation (nan/inf)", i, j
                    )
                    continue

                # Validate this pair against all views in the match group
                error = self._compute_reprojection_error(
                    point_3d, points_2d, projection_matrices
                )

                if not np.isfinite(error) or error > self.config.max_reprojection_error:
                    logger.debug(
                        "Rejecting pair (%d,%d): error=%.2fpx > threshold=%.2fpx",
                        i,
                        j,
                        error,
                        self.config.max_reprojection_error,
                    )
                    continue

                result_points.append(
                    Point3D(
                        position=point_3d,
                        reprojection_error=error,
                        source_detections=[detection_ids[i], detection_ids[j]],
                        match_group_id=match_group.group_id,
                    )
                )

        return result_points

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

    def _triangulate_two_view(
        self,
        pt1: tuple[float, float],
        pt2: tuple[float, float],
        P1: np.ndarray,
        P2: np.ndarray,
    ) -> np.ndarray:
        """
        Triangulate a 3D point from two 2D correspondences.

        Uses cv2.triangulatePoints (Direct Linear Transform).

        Args:
            pt1: (x, y) in camera 1
            pt2: (x, y) in camera 2
            P1: (3, 4) projection matrix for camera 1
            P2: (3, 4) projection matrix for camera 2

        Returns:
            (3,) array of world coordinates [x, y, z]
        """
        # Convert points to float64 column vectors (required by OpenCV)
        pts1 = np.array([[pt1[0]], [pt1[1]]], dtype=np.float64)
        pts2 = np.array([[pt2[0]], [pt2[1]]], dtype=np.float64)

        # Ensure projection matrices are float64
        P1 = P1.astype(np.float64)
        P2 = P2.astype(np.float64)

        # Triangulate: returns (4, 1) homogeneous coordinates
        point_4d_homogeneous = cv2.triangulatePoints(P1, P2, pts1, pts2)

        # Convert from homogeneous to Cartesian coordinates.
        # W ~= 0 means point at infinity / degenerate triangulation.
        w = float(point_4d_homogeneous[3, 0])
        if abs(w) < EPS_DENOM:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        point_3d = point_4d_homogeneous[:3, 0] / w

        return point_3d.astype(np.float64)  # (3,) array, explicit dtype

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing Triangulator")
    logger.info("=" * 60)

    # This is a basic test - full validation is in scene_reconstructor.py
    from src.config.settings import settings

    triangulator = Triangulator(settings.reconstruction)
    logger.info("Triangulator created with config:")
    logger.info(
        "  max_reprojection_error: %.1f", settings.reconstruction.max_reprojection_error
    )

    logger.info("\nTriangulator ready for use")
