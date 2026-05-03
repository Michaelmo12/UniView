"""
PersonKalmanFilter

Wraps filterpy.kalman.KalmanFilter for a 3D constant-velocity model.
State vector: [x, y, z, vx, vy, vz]
Measurement:  [x, y, z]
"""

import logging

import numpy as np
from filterpy.kalman import KalmanFilter

from src.config.settings import TrackingConfig

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
        # Position [x, y, z] — where the person is
        #Velocity [vx, vy, vz] — how fast they're moving per frame
        self.kf.F = np.array(
            [
                [1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        # x_new = F @ x_old

        # Measurement function H: observe only position [x, y, z]
        # what we want to receive from the triangulation
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float64,
        )

        # Initial state: position from detection, velocity zero
        self.kf.x = np.zeros((6, 1), dtype=np.float64)
        self.kf.x[:3] = initial_position.reshape(3, 1)

        # measurement noise: High R = triangulation is noisy, Kalman relies more on its own prediction. Low R = trust the measurement more.
        self.kf.R = np.eye(3, dtype=np.float64) * config.measurement_noise

        # process noise: High Q = person moves unpredictably, Kalman adapts faster to new measurements. Low Q = assumes smooth motion, filter is more stable but slower to react to sudden direction changes.
        
        self.kf.Q = np.eye(6, dtype=np.float64) * config.process_noise
        self.kf.Q[3:, 3:] *= 0.01

        # מטריצת האי-ודאות
        # confident of where we last were 3d place [x,y,z],
        # but we dont yet know the direction so we we put high initial "weight"
        # current uncertainty (high velocity uncertainty at start, shrinks as we see more frames)
        self.kf.P = np.eye(6, dtype=np.float64)
        self.kf.P[
            0:3, 0:3
        ] *= 1.0  # position uncertainty: 1m^2 (moderate initial uncertainty)
        self.kf.P[
            3:6, 3:6
        ] *= 1000.0  # velocity uncertainty: very high (completely unknown initially)

    def predict(self) -> None:
        """Predict the state forward one time step."""
        self.kf.predict() 
        # x = F @ x : new position = old position + velocity

    def update(self, measurement: np.ndarray) -> None:
        """
        Update the filter with a new measurement.
            measurement: Shape (3,) observed [x, y, z] position [meters].
        """
        self.kf.update(measurement.reshape(3, 1)) # Kalman gain, residual, posterior

    @property
    def predicted_position(self) -> np.ndarray:
        """Get the current predicted position as shape (3,)."""
        return self.kf.x[:3].flatten() # first 3 elements of the 6-state vector

    @property
    def velocity(self) -> np.ndarray:
        """Get the current estimated velocity as shape (3,)."""
        return self.kf.x[3:].flatten() # last 3 elements
