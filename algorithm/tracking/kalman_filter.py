"""
PersonKalmanFilter

Wraps filterpy.kalman.KalmanFilter for a 3D constant-velocity model.
State vector: [x, y, z, vx, vy, vz]
Measurement:  [x, y, z]
"""

import logging

import numpy as np
from filterpy.kalman import KalmanFilter

from algorithm.config.settings import TrackingConfig

logger = logging.getLogger(__name__)


class PersonKalmanFilter:
    """
    Kalman filter for tracking a person's 3D position and velocity.

    Uses a constant-velocity motion model with dt=1 (one frame per step).
    State:       [x, y, z, vx, vy, vz]  (dim_x = 6)
    Measurement: [x, y, z]              (dim_z = 3)
    """

    def __init__(self, initial_position: np.ndarray, config: TrackingConfig) -> None:
        """
        Initialize the Kalman filter at a given 3D position.

        Args:
            initial_position: Shape (3,) initial [x, y, z] in world coordinates [meters].
            config: TrackingConfig with process_noise and measurement_noise.
        """
        self.kf = KalmanFilter(dim_x=6, dim_z=3)

        # State transition matrix F: constant-velocity model (dt=1)
        self.kf.F = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float64)

        # Measurement function H: observe only position [x, y, z]
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=np.float64)

        # Initial state: position from detection, velocity zero
        self.kf.x = np.zeros((6, 1), dtype=np.float64)
        self.kf.x[:3] = initial_position.reshape(3, 1)

        # Measurement noise R: triangulation variance (~0.5m)
        self.kf.R = np.eye(3, dtype=np.float64) * config.measurement_noise

        # Process noise Q: low = assumes smooth motion; velocity noise much smaller
        self.kf.Q = np.eye(6, dtype=np.float64) * config.process_noise
        self.kf.Q[3:, 3:] *= 0.01

        # Initial covariance P: moderate position uncertainty, very high velocity uncertainty
        self.kf.P = np.eye(6, dtype=np.float64)
        self.kf.P[0:3, 0:3] *= 1.0      # position uncertainty: 1m² (moderate initial uncertainty)
        self.kf.P[3:6, 3:6] *= 1000.0   # velocity uncertainty: very high (completely unknown initially)

    def predict(self) -> None:
        """Predict the state forward one time step."""
        self.kf.predict()

    def update(self, measurement: np.ndarray) -> None:
        """
        Update the filter with a new measurement.

        Args:
            measurement: Shape (3,) observed [x, y, z] position [meters].
        """
        self.kf.update(measurement.reshape(3, 1))

    @property
    def predicted_position(self) -> np.ndarray:
        """Get the current predicted position as shape (3,)."""
        return self.kf.x[:3].flatten()

    @property
    def velocity(self) -> np.ndarray:
        """Get the current estimated velocity as shape (3,)."""
        return self.kf.x[3:].flatten()
