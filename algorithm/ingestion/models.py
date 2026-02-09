"""
Ingestion Stage Data Models

Defines the data structures produced by the TCP receiver and used by downstream stages.

Key Types:
- CameraCalibration: Intrinsic (K) and extrinsic (R, t) camera parameters
- DroneFrame: Single frame from one drone with calibration
- SynchronizedFrameSet: Frames from all drones at the same timestamp
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class CameraCalibration:
    """
    Complete camera calibration for a single frame.

    Attributes:
        K: Intrinsic matrix (3x3)
           [[fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]]

           fx, fy = focal length in pixels (how "zoomed in" the camera is)
           cx, cy = principal point (where optical axis hits sensor, ~image center)

        R: Rotation matrix (3x3)
           Transforms world coordinates to camera coordinates.
           Columns are camera's X, Y, Z axes expressed in world frame.
           Must be orthonormal: R @ R.T = I, det(R) = 1

        t: Translation vector (3x1)
           Camera position relative to world origin, in camera coordinates.
           Note: Camera center in world coords is C = -R.T @ t

        dist: Distortion coefficients (5,)
              [k1, k2, p1, p2, k3] - radial and tangential distortion
              Used by cv2.undistort() to correct lens distortion
    """

    K: np.ndarray  # (3, 3) float32 - intrinsic matrix
    R: np.ndarray  # (3, 3) float32 - rotation matrix (world → camera)
    t: np.ndarray  # (3, 1) float32 - translation vector
    dist: np.ndarray  # (5,) float32 - distortion coefficients

    def __post_init__(self):
        assert self.K.shape == (3, 3), f"K must be 3x3, got {self.K.shape}"
        assert self.R.shape == (3, 3), f"R must be 3x3, got {self.R.shape}"
        assert self.t.shape == (3, 1), f"t must be 3x1, got {self.t.shape}"
        assert self.dist.shape == (5,), f"dist must be (5,), got {self.dist.shape}"

    @property
    def projection_matrix(self) -> np.ndarray:
        """
        Compute the 3x4 projection matrix P = K @ [R | t].

        P projects 3D world points to 2D image pixels:
            [u]       [X]
            [v] = P @ [Y]   (homogeneous coordinates)
            [w]       [Z]
                      [1]

        Then: pixel = (u/w, v/w)

        Returns:
            P: (3, 4) projection matrix
        """
        # np.hstack joins arrays horizontally
        Rt = np.hstack([self.R, self.t])  # Shape: (3, 4)

        # P = K @ [R | t]
        # K is 3x3, Rt is 3x4, result is 3x4
        P = self.K @ Rt

        return P

    # where the point is in world coordinates
    @property
    def camera_center(self) -> np.ndarray:
        """
        Compute camera center in world coordinates.

        The camera center C is the 3D point that projects to all pixels.
        It's where all viewing rays originate.

        Derivation:
        - A point X_cam in camera coords relates to world coords by:
          X_cam = R @ X_world + t
        - Camera center is at X_cam = [0, 0, 0]
        - So: 0 = R @ C + t
        - Therefore: C = -R.T @ t  (since R.T = R^(-1) for rotation matrices)

        Returns:
            C: (3,) camera center in world coordinates
        """
        # R.T is the transpose of R
        # @ is matrix multiplication
        # .flatten() converts (3,1) to (3,) for convenience
        C = -self.R.T @ self.t
        return C.flatten()


@dataclass
class DroneFrame:
    """
    A single frame received from one drone.

    This is the output of the TCP receiver for one drone at one timestamp.
    Contains everything needed for detection and reconstruction:
    - The image data (for YOLO detection)
    - Camera calibration (for 3D geometry)
    - Metadata (for synchronization and debugging)

    Attributes:
        drone_id: Which drone this frame came from (1-8)
        frame_num: Frame sequence number (0-999 in MATRIX dataset)
        timestamp: When the frame was captured (seconds since epoch)
        frame: BGR image as numpy array, shape (H, W, 3)
        calibration: Camera parameters for this frame
    """

    drone_id: int
    frame_num: int
    timestamp: float
    frame: np.ndarray  # BGR image, shape (H, W, 3), dtype uint8
    calibration: CameraCalibration  # K, R, t, dist for this frame

    def __post_init__(self):
        assert 1 <= self.drone_id <= 8, f"drone_id must be 1-8, got {self.drone_id}"
        assert (
            len(self.frame.shape) == 3
        ), f"frame must be 3D (H,W,C), got {self.frame.shape}"
        assert (
            self.frame.shape[2] == 3
        ), f"frame must have 3 channels, got {self.frame.shape[2]}"

    @property
    def frame_height(self) -> int:
        return self.frame.shape[0]

    @property
    def frame_width(self) -> int:
        return self.frame.shape[1]


@dataclass
class SynchronizedFrameSet:
    """
    A set of frames from multiple drones at approximately the same timestamp.

    The pipeline processes frames in synchronized sets:
    1. TCP receivers collect frames from all 8 drones
    2. Synchronizer groups frames by timestamp (within tolerance)
    3. Each synchronized set is processed as a unit

    This enables cross-camera matching - we can only match detections
    across cameras if they're from the same moment in time.

    Attributes:
        frame_num: The frame number this set represents
        timestamp: Reference timestamp for synchronization
        frames: Dict mapping drone_id → DroneFrame
        num_drones_expected: How many drones we expect (default 8)
    """

    frame_num: int  # Which frame number (0-999)
    timestamp: float  # Reference timestamp
    frames: dict[int, DroneFrame]  # {drone_id: DroneFrame}
    num_drones_expected: int = 8  # MATRIX has 8 cameras

    @property
    def num_drones_present(self) -> int:
        return len(self.frames)

    @property
    def is_complete(self) -> bool:
        """checks if there are frames from all expected drones in this set"""
        return self.num_drones_present == self.num_drones_expected

    @property
    def missing_drones(self) -> list[int]:
        expected = set(range(1, self.num_drones_expected + 1))
        present = set(self.frames.keys())
        return sorted(expected - present)

    @property
    def drone_ids(self) -> list[int]:
        return sorted(self.frames.keys())

    def get_frame(self, drone_id: int) -> Optional[DroneFrame]:
        return self.frames.get(drone_id)

    def get_all_frames(self) -> list[DroneFrame]:
        return [self.frames[d] for d in self.drone_ids]

    def get_all_calibrations(self) -> dict[int, CameraCalibration]:
        return {d: f.calibration for d, f in self.frames.items()}


# =============================================================================
# DEBUG / TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Ingestion Models")
    print("=" * 50)

    # Create mock calibration
    K = np.array(
        [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )

    R = np.eye(3, dtype=np.float32)  # Identity = camera aligned with world

    t = np.array([[0.0], [0.0], [5.0]], dtype=np.float32)  # 5m in front of origin

    dist = np.zeros(5, dtype=np.float32)

    calib = CameraCalibration(K=K, R=R, t=t, dist=dist)

    print("\nCameraCalibration:")
    print(f"  K shape: {calib.K.shape}")
    print(f"  R shape: {calib.R.shape}")
    print(f"  t shape: {calib.t.shape}")
    print(f"  Camera center: {calib.camera_center}")
    print(f"  Projection matrix shape: {calib.projection_matrix.shape}")

    # Create mock frame
    mock_image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    drone_frame = DroneFrame(
        drone_id=1,
        frame_num=0,
        timestamp=1234567890.123,
        frame=mock_image,
        calibration=calib,
    )

    print("\nDroneFrame:")
    print(f"  drone_id: {drone_frame.drone_id}")
    print(f"  frame_num: {drone_frame.frame_num}")
    print(f"  frame size: {drone_frame.frame_width}x{drone_frame.frame_height}")

    # Create synchronized set
    frames = {1: drone_frame}
    sync_set = SynchronizedFrameSet(
        frame_num=0, timestamp=1234567890.123, frames=frames
    )

    print("\nSynchronizedFrameSet:")
    print(f"  frame_num: {sync_set.frame_num}")
    print(
        f"  drones present: {sync_set.num_drones_present}/{sync_set.num_drones_expected}"
    )
    print(f"  is_complete: {sync_set.is_complete}")
    print(f"  missing drones: {sync_set.missing_drones}")

    print("\n" + "=" * 50)
    print("All tests passed!")
