"""
DatasetLoader - MATRIX Dataset Frame and Calibration Loader

Loads frames and per-frame calibration data from the MATRIX multi-drone dataset.
Falls back to synthetic data (random colored frames + identity calibration) when
the dataset directory is not found, so the streamer runs standalone without data.

Expected MATRIX directory structure:
    {dataset_path}/
        Drone{N}/
            frames/
                frame_0000.jpg
                frame_0001.jpg
                ...
            calibration/
                K.txt               <- 3x3 intrinsic matrix (3 rows, space-separated)
                R_frame_0000.txt    <- 3x3 rotation matrix per frame
                t_frame_0000.txt    <- 3x1 translation vector per frame (3 lines, 1 value each)

Calibration note: MATRIX dataset does not include lens distortion coefficients.
Distortion defaults to np.zeros(5, dtype=np.float32).
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads frames and calibration data from a MATRIX drone dataset folder.

    If the drone folder is not found, or if any frame/calibration file is
    missing, this loader transparently falls back to synthetic data so the
    streamer can always produce valid packets.

    Args:
        dataset_path: Root path to the MATRIX dataset (contains Drone1/, Drone2/, ...).
        drone_id:     Which drone to load (1-8).
        jpeg_quality: JPEG encoding quality for loaded frames (0-100).
    """

    SYNTHETIC_FRAME_COUNT = 100
    SYNTHETIC_WIDTH = 640
    SYNTHETIC_HEIGHT = 480

    def __init__(self, dataset_path: str, drone_id: int, jpeg_quality: int = 85) -> None:
        self.dataset_path = dataset_path
        self.drone_id = drone_id
        self.jpeg_quality = jpeg_quality

        self._drone_dir = Path(dataset_path) / f"Drone{drone_id}"
        self._frames_dir = self._drone_dir / "frames"
        self._calib_dir = self._drone_dir / "calibration"

        self._synthetic = False
        self._K_shared: np.ndarray | None = None  # Shared across frames if loaded once

        if not self._drone_dir.exists():
            logger.warning(
                "Drone%d dataset directory not found at '%s'. "
                "Falling back to synthetic frames.",
                drone_id,
                self._drone_dir,
            )
            self._synthetic = True
        else:
            logger.info(
                "DatasetLoader: Drone%d dataset found at '%s'",
                drone_id,
                self._drone_dir,
            )
            self._K_shared = self._load_K()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_frame_count(self) -> int:
        """Return the number of available frames (or synthetic count)."""
        if self._synthetic:
            return self.SYNTHETIC_FRAME_COUNT

        if not self._frames_dir.exists():
            logger.warning(
                "Drone%d: frames directory missing, using synthetic count.",
                self.drone_id,
            )
            return self.SYNTHETIC_FRAME_COUNT

        frame_files = sorted(self._frames_dir.glob("frame_*.jpg"))
        count = len(frame_files)
        if count == 0:
            logger.warning(
                "Drone%d: no frame_*.jpg files found in '%s', using synthetic count.",
                self.drone_id,
                self._frames_dir,
            )
            return self.SYNTHETIC_FRAME_COUNT

        return count

    def load_frame(
        self, frame_idx: int
    ) -> tuple[bytes, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load a single frame and its calibration data.

        Args:
            frame_idx: Zero-based frame index.

        Returns:
            (jpeg_bytes, K, R, t, dist) where:
                jpeg_bytes: JPEG-encoded image as bytes
                K:    (3,3) float32 intrinsic matrix
                R:    (3,3) float32 rotation matrix
                t:    (3,1) float32 translation vector
                dist: (5,)  float32 distortion coefficients (zeros for MATRIX)
        """
        if self._synthetic:
            return self._load_synthetic_frame(frame_idx)

        try:
            return self._load_real_frame(frame_idx)
        except Exception as exc:
            logger.warning(
                "Drone%d: failed to load real frame %d (%s). Falling back to synthetic.",
                self.drone_id,
                frame_idx,
                exc,
            )
            return self._load_synthetic_frame(frame_idx)

    # ------------------------------------------------------------------
    # Real data loading
    # ------------------------------------------------------------------

    def _load_real_frame(
        self, frame_idx: int
    ) -> tuple[bytes, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        frame_path = self._frames_dir / f"frame_{frame_idx:04d}.jpg"
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame file not found: {frame_path}")

        image = cv2.imread(str(frame_path))
        if image is None:
            raise ValueError(f"cv2.imread returned None for: {frame_path}")

        jpeg_bytes = self._encode_jpeg(image)

        K = self._K_shared if self._K_shared is not None else self._load_K()
        R = self._load_R(frame_idx)
        t = self._load_t(frame_idx)
        dist = np.zeros(5, dtype=np.float32)

        return jpeg_bytes, K, R, t, dist

    def _load_K(self) -> np.ndarray:
        """Load 3x3 intrinsic matrix from calibration/K.txt."""
        k_path = self._calib_dir / "K.txt"
        if not k_path.exists():
            logger.warning(
                "Drone%d: K.txt not found at '%s'. Using identity K.",
                self.drone_id,
                k_path,
            )
            return self._synthetic_K()

        try:
            K = np.loadtxt(str(k_path), dtype=np.float32)
            if K.shape != (3, 3):
                raise ValueError(f"Expected (3,3) but got {K.shape}")
            return K
        except Exception as exc:
            logger.warning(
                "Drone%d: failed to parse K.txt: %s. Using identity K.",
                self.drone_id,
                exc,
            )
            return self._synthetic_K()

    def _load_R(self, frame_idx: int) -> np.ndarray:
        """Load 3x3 rotation matrix for a specific frame."""
        r_path = self._calib_dir / f"R_frame_{frame_idx:04d}.txt"
        if not r_path.exists():
            logger.debug(
                "Drone%d: R_frame_%04d.txt not found. Using identity R.",
                self.drone_id,
                frame_idx,
            )
            return np.eye(3, dtype=np.float32)

        try:
            R = np.loadtxt(str(r_path), dtype=np.float32)
            if R.shape != (3, 3):
                raise ValueError(f"Expected (3,3) but got {R.shape}")
            return R
        except Exception as exc:
            logger.warning(
                "Drone%d: failed to parse R_frame_%04d.txt: %s. Using identity R.",
                self.drone_id,
                frame_idx,
                exc,
            )
            return np.eye(3, dtype=np.float32)

    def _load_t(self, frame_idx: int) -> np.ndarray:
        """Load 3x1 translation vector for a specific frame."""
        t_path = self._calib_dir / f"t_frame_{frame_idx:04d}.txt"
        if not t_path.exists():
            logger.debug(
                "Drone%d: t_frame_%04d.txt not found. Using zero t.",
                self.drone_id,
                frame_idx,
            )
            return np.zeros((3, 1), dtype=np.float32)

        try:
            t_flat = np.loadtxt(str(t_path), dtype=np.float32)
            t = t_flat.reshape(3, 1)
            return t
        except Exception as exc:
            logger.warning(
                "Drone%d: failed to parse t_frame_%04d.txt: %s. Using zero t.",
                self.drone_id,
                frame_idx,
                exc,
            )
            return np.zeros((3, 1), dtype=np.float32)

    # ------------------------------------------------------------------
    # Synthetic data generation
    # ------------------------------------------------------------------

    def _load_synthetic_frame(
        self, frame_idx: int
    ) -> tuple[bytes, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate a synthetic frame for offline testing / missing dataset."""
        # Deterministic color per drone so we can visually identify sources
        rng = np.random.default_rng(seed=self.drone_id * 1000 + frame_idx)
        color = rng.integers(50, 200, size=3).tolist()

        image = np.full(
            (self.SYNTHETIC_HEIGHT, self.SYNTHETIC_WIDTH, 3),
            color,
            dtype=np.uint8,
        )

        # Overlay text so the drone ID is visible in the JPEG stream
        label = f"Drone {self.drone_id} | Frame {frame_idx}"
        cv2.putText(
            image,
            label,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        jpeg_bytes = self._encode_jpeg(image)

        K = self._synthetic_K()
        R = np.eye(3, dtype=np.float32)
        t = np.zeros((3, 1), dtype=np.float32)
        dist = np.zeros(5, dtype=np.float32)

        return jpeg_bytes, K, R, t, dist

    @staticmethod
    def _synthetic_K() -> np.ndarray:
        """Identity-style intrinsic matrix for synthetic fallback."""
        return np.array(
            [
                [800.0, 0.0, 320.0],
                [0.0, 800.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _encode_jpeg(self, image: np.ndarray) -> bytes:
        """JPEG-encode a BGR image to bytes."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        success, buffer = cv2.imencode(".jpg", image, encode_params)
        if not success:
            raise RuntimeError("cv2.imencode failed")
        return buffer.tobytes()
