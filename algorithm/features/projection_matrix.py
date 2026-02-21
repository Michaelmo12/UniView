import numpy as np

from algorithm.ingestion.models import CameraCalibration


class ProjectionMatrixCalculator:
    """
    Calculator for camera projection matrices.

    The projection matrix P maps 3D world points to 2D image pixels:

        [u]       [X]
        [v] = P @ [Y]   (homogeneous coordinates)
        [w]       [Z]
                  [1]

    Then pixel coordinates: (u/w, v/w)

    P = K @ [R | t] where:
    - K is 3x3 intrinsic matrix (focal length, principal point)
    - R is 3x3 rotation matrix (world to camera orientation)
    - t is 3x1 translation vector (camera position)
    """

    def compute(self, calibration: CameraCalibration) -> np.ndarray:
        """
        Compute projection matrix from calibration.

        Args:
            calibration: CameraCalibration with K, R, t

        Returns:
            P: (3, 4) projection matrix
        """
        return calibration.projection_matrix

    def compute_batch(
        self,
        calibrations: dict[int, CameraCalibration],
    ) -> dict[int, np.ndarray]:
        """
        Compute projection matrices for multiple cameras.

        Args:
            calibrations: {drone_id: CameraCalibration}

        Returns:
            {drone_id: projection_matrix}
        """
        result = {}

        for drone_id, calib in calibrations.items():

            projection_matrix = self.compute(calib)

            result[drone_id] = projection_matrix

        return result
