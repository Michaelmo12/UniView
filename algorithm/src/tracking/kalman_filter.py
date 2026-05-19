"""
PersonKalmanFilter — whitebox implementation, no filterpy.

Kalman filter equations (constant-velocity model, dt=1):

  Predict:
    x = F @ x              — project state forward one frame
    P = F @ P @ F.T + Q    — project uncertainty forward

  Update:
    y = z - H @ x          — residual: difference between measurement and prediction
    S = H @ P @ H.T + R    — innovation covariance (how uncertain is the residual)
    K = P @ H.T @ inv(S)   — Kalman gain: how much to trust the measurement vs prediction
    x = x + K @ y          — correct the state
    P = (I - K @ H) @ P    — correct the uncertainty

State vector x (6x1): [x, y, z, vx, vy, vz]
Measurement z (3x1):  [x, y, z]
"""

import logging

import numpy as np

from src.config.settings import TrackingConfig

logger = logging.getLogger(__name__)


class PersonKalmanFilter:
    """
    Whitebox Kalman filter for tracking a person's 3D position and velocity.

    Constant-velocity motion model with dt=1 (one frame per step).
    State:       [x, y, z, vx, vy, vz]  (6x1)
    Measurement: [x, y, z]              (3x1)
    """

    def __init__(self, initial_position: np.ndarray, config: TrackingConfig) -> None:
        # F — state transition matrix: applies velocity to position each frame
        # x_new = F @ x_old
        # [x + vx, y + vy, z + vz, vx, vy, vz]
        self.F = np.array(
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

        # H — measurement matrix: extracts position [x, y, z] from full state [x,y,z,vx,vy,vz]
        # z = H @ x
        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float64,
        )

        # R — measurement noise covariance (3x3): how much to trust the triangulated position
        # high R = noisy triangulation, rely more on prediction
        # low R  = trust the measurement more than the prediction
        self.R = np.eye(3, dtype=np.float64) * config.measurement_noise

        # Q — process noise covariance (6x6): how much unpredictable movement to allow
        # high Q = person moves erratically, filter adapts fast to new measurements
        # low Q  = smooth motion assumed, filter is stable but slow to react to sudden changes
        self.Q = np.eye(6, dtype=np.float64) * config.process_noise
        # velocity components are less noisy than position — scale down their process noise
        self.Q[3:, 3:] *= 0.01

        # P — state covariance (6x6): current uncertainty of the state estimate
        # position uncertainty: moderate at start (1 m^2)
        # velocity uncertainty: very high at start — we have no idea how fast they're moving
        # LIKE F but for confidence of it
        self.P = np.eye(6, dtype=np.float64)
        self.P[0:3, 0:3] *= 1.0
        self.P[3:6, 3:6] *= 1000.0

        # x — state vector (6x1): initial position from detection, velocity assumed zero
        self.x = np.zeros((6, 1), dtype=np.float64)
        self.x[:3] = initial_position.reshape(3, 1)

        # identity matrix reused in the update step
        self._I = np.eye(6, dtype=np.float64)

    def predict(self) -> None:
        """
        Predict step — project state and uncertainty forward one frame.

          x_new = F @ x
          כמה אני בטוח בוקטור המהירות אחרי חיזוי מגדיל את פ
          P = F @ P @ F.T + Q
        """
        # project state forward: new position = old position + velocity
        self.x = self.F @ self.x
        # project uncertainty forward and add process noise
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement: np.ndarray) -> None:
        """
        Update step — correct the prediction with a new triangulated measurement.

          y = z - H @ x          residual
          S = H @ P @ H.T + R    innovation covariance
          K = P @ H.T @ inv(S)   Kalman gain
          x = x + K @ y          corrected state
          P = (I - K @ H) @ P    corrected uncertainty

        Args:
            measurement: Shape (3,) observed [x, y, z] in world coordinates [meters].
        """
        z = measurement.reshape(3, 1)

        # איפה בן אדם נמצא פחות איפה המודל חושב שהוא נמצא מחשב מחשב פער
        y = z - self.H @ self.x

        # who do we belive more the camera R or P our guess
        S = self.H @ self.P @ self.H.T + self.R

        # K — Kalman gain: how much weight to give the measurement vs the prediction
        # high K = trust measurement more; low K = trust prediction more
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # correct the state estimate using the residual weighted by K
        self.x = self.x + K @ y

        # correct the uncertainty — shrinks because we got new information
        self.P = (self._I - K @ self.H) @ self.P

    @property
    def predicted_position(self) -> np.ndarray:
        """Current position estimate — first 3 elements of the state vector."""
        return self.x[:3].flatten()

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity estimate — last 3 elements of the state vector."""
        return self.x[3:].flatten()