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

from algorithm.config.settings import ReconstructionConfig
from algorithm.fusion.models import MatchGroup
from algorithm.detection.models import DetectionSet
from algorithm.ingestion.models import SynchronizedFrameSet
from algorithm.reconstruction.models import Point3D

logger = logging.getLogger(__name__)


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
        sync_set: SynchronizedFrameSet
    ) -> Point3D | None:
        """
        Triangulate a match group into a 3D point.

        Extracts 2D bbox centers and projection matrices for each detection in
        the match group, performs triangulation, validates via reprojection error.

        Args:
            match_group: MatchGroup with detections list [(drone_id, local_id), ...]
            detection_sets: Dict mapping drone_id to DetectionSet
            sync_set: SynchronizedFrameSet with calibration data

        Returns:
            Point3D if triangulation successful and error below threshold, None otherwise
        """
        # Extract 2D points and projection matrices
        points_2d = []
        projection_matrices = []

        for drone_id, local_id in match_group.detections:
            # Get detection bbox center
            detection = detection_sets[drone_id].detections[local_id]
            center = detection.bbox.center  # (cx, cy) tuple
            points_2d.append(center)

            # Get projection matrix from calibration
            P = sync_set.frames[drone_id].calibration.projection_matrix
            projection_matrices.append(P)

        # Triangulate based on number of views
        num_views = len(points_2d)

        if num_views < 2:
            logger.warning(
                "Match group has < 2 views, cannot triangulate: %s",
                match_group.detections
            )
            return None

        if num_views == 2:
            # Two-view triangulation
            point_3d = self._triangulate_two_view(
                points_2d[0], points_2d[1],
                projection_matrices[0], projection_matrices[1]
            )
        else:
            # Multi-view: triangulate all pairs and take median (robust to outliers)
            triangulated_points = []
            for i in range(num_views):
                for j in range(i + 1, num_views):
                    pt_3d = self._triangulate_two_view(
                        points_2d[i], points_2d[j],
                        projection_matrices[i], projection_matrices[j]
                    )
                    triangulated_points.append(pt_3d)

            # Take median across all pairwise triangulations
            point_3d = np.median(triangulated_points, axis=0)

        # Compute reprojection error
        error = self._compute_reprojection_error(
            point_3d, points_2d, projection_matrices
        )

        # Validate against threshold
        if error > self.config.max_reprojection_error:
            logger.debug(
                "Rejecting triangulation: error=%.2fpx > threshold=%.2fpx",
                error, self.config.max_reprojection_error
            )
            return None

        # Create Point3D
        return Point3D(
            position=point_3d,
            reprojection_error=error,
            num_views=num_views,
            source_detections=match_group.detections
        )

    def _triangulate_two_view(
        self,
        pt1: tuple[float, float],
        pt2: tuple[float, float],
        P1: np.ndarray,
        P2: np.ndarray
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

        # Convert from homogeneous to Cartesian coordinates
        # Divide [X, Y, Z, W] by W to get [x, y, z]
        point_3d = point_4d_homogeneous[:3, 0] / point_4d_homogeneous[3, 0]

        return point_3d  # (3,) array

    def _compute_reprojection_error(
        self,
        point_3d: np.ndarray,
        points_2d: list[tuple[float, float]],
        projection_matrices: list[np.ndarray]
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

            # Convert to pixel coordinates: (u/w, v/w)
            projected_x = projected_homogeneous[0] / projected_homogeneous[2]
            projected_y = projected_homogeneous[1] / projected_homogeneous[2]

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
    from algorithm.config.settings import settings

    triangulator = Triangulator(settings.reconstruction)
    logger.info("Triangulator created with config:")
    logger.info("  max_reprojection_error: %.1f", settings.reconstruction.max_reprojection_error)

    logger.info("\nTriangulator ready for use")
