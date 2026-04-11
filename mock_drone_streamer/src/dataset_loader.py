"""
DatasetLoader - MATRIX Dataset Frame and Calibration Loader

Reads the actual MATRIX_30x30 dataset layout:

    {dataset_path}/
        image_subsets/
            D{N}/
                0000.png
                0001.png
                ...
        calibrations/
            extrinsic/
                extr_Drone{N}_{frame:04d}.xml   <- binary base64 rvec (3xfloat64) + tvec (3xfloat64)
            intrinsic/
                intr_Drone{N}_{frame:04d}.xml   <- text K matrix + distortion coefficients

Drone IDs 1-8 map to image folders D1-D8.

Calibration processing:
    - rvec is converted to R matrix via cv2.Rodrigues() in this loader
    - All output matrices are float32 numpy arrays
    - Packet builder receives (jpeg_bytes, K, R, t, dist) — no further conversion needed
"""

import base64
import logging
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads frames and per-frame calibration from the MATRIX_30x30 dataset.

    Args:
        dataset_path: Root of the MATRIX dataset
                      (the folder that contains image_subsets/ and calibrations/).
        drone_id:     Drone number 1-8.
        jpeg_quality: JPEG encoding quality for frames (0-100).
    """

    def __init__(self, dataset_path: str, drone_id: int, jpeg_quality: int = 85) -> None:
        self.dataset_path = Path(dataset_path)
        self.drone_id = drone_id
        self.jpeg_quality = jpeg_quality

        self._frames_dir = self.dataset_path / "image_subsets" / f"D{drone_id}"
        self._extr_dir = self.dataset_path / "calibrations" / "extrinsic"
        self._intr_dir = self.dataset_path / "calibrations" / "intrinsic"

        if not self._frames_dir.exists():
            raise FileNotFoundError(
                f"Drone{drone_id}: frames directory not found at '{self._frames_dir}'. "
                f"Check dataset_path points to the folder containing image_subsets/ and calibrations/."
            )

        if not self._extr_dir.exists():
            raise FileNotFoundError(
                f"Extrinsic calibration directory not found at '{self._extr_dir}'."
            )

        if not self._intr_dir.exists():
            raise FileNotFoundError(
                f"Intrinsic calibration directory not found at '{self._intr_dir}'."
            )

        logger.info(
            "DatasetLoader: Drone%d dataset ready at '%s'",
            drone_id,
            self._frames_dir,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_frame_count(self) -> int:
        """Return the number of available frames."""
        frames = sorted(self._frames_dir.glob("[0-9][0-9][0-9][0-9].png"))
        count = len(frames)
        if count == 0:
            raise FileNotFoundError(
                f"Drone{self.drone_id}: no 0000.png-style frames found in '{self._frames_dir}'."
            )
        return count

    def load_frame(
        self, frame_idx: int
    ) -> tuple[bytes, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load a single frame and its calibration.

        Args:
            frame_idx: Zero-based frame index.

        Returns:
            (jpeg_bytes, K, R, t, dist) where:
                jpeg_bytes: JPEG-encoded image bytes
                K:    (3, 3) float32 intrinsic matrix
                R:    (3, 3) float32 rotation matrix  (converted from rvec via Rodrigues)
                t:    (3, 1) float32 translation vector
                dist: (5,)   float32 distortion coefficients
        """
        return self._load_real_frame(frame_idx)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_real_frame(
        self, frame_idx: int
    ) -> tuple[bytes, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        frame_path = self._frames_dir / f"{frame_idx:04d}.png"
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame not found: {frame_path}")

        image = cv2.imread(str(frame_path))
        if image is None:
            raise ValueError(f"cv2.imread returned None for: {frame_path}")

        jpeg_bytes = self._encode_jpeg(image)
        K, dist = self._load_intrinsic(frame_idx)
        R, t = self._load_extrinsic(frame_idx)

        return jpeg_bytes, K, R, t, dist

    def _load_extrinsic(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Load extrinsic XML, decode binary rvec+tvec, convert rvec->R via Rodrigues.

        Returns:
            R: (3, 3) float32 rotation matrix
            t: (3, 1) float32 translation vector
        """
        filename = f"extr_Drone{self.drone_id}_{frame_idx:04d}.xml"
        filepath = self._extr_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Extrinsic file not found: {filepath}")

        tree = ET.parse(str(filepath))
        root = tree.getroot()

        rvec_binary = self._extract_binary_element(root, "rvec")
        tvec_binary = self._extract_binary_element(root, "tvec")

        # Each is 3 float64 values = 24 bytes; dataset stores them with a prefix,
        # so we take the last 24 bytes (same approach as mock_drone_streamer).
        rvec_vals = struct.unpack("ddd", rvec_binary[-24:])
        tvec_vals = struct.unpack("ddd", tvec_binary[-24:])

        rvec = np.array(rvec_vals, dtype=np.float64).reshape(3, 1)
        tvec = np.array(tvec_vals, dtype=np.float64).reshape(3, 1)

        R, _ = cv2.Rodrigues(rvec)
        R = R.astype(np.float32)
        t = tvec.astype(np.float32)

        return R, t

    def _load_intrinsic(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Load intrinsic XML, parse K matrix and distortion coefficients.

        Returns:
            K:    (3, 3) float32
            dist: (5,)   float32
        """
        filename = f"intr_Drone{self.drone_id}_{frame_idx:04d}.xml"
        filepath = self._intr_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Intrinsic file not found: {filepath}")

        tree = ET.parse(str(filepath))
        root = tree.getroot()

        K = None
        cam_elem = root.find(".//camera_matrix")
        if cam_elem is not None:
            data_elem = cam_elem.find(".//data")
            if data_elem is not None and data_elem.text:
                vals = [float(x) for x in data_elem.text.split()]
                K = np.array(vals, dtype=np.float32).reshape(3, 3)

        if K is None:
            raise ValueError(
                f"Drone{self.drone_id} frame {frame_idx}: could not parse K from '{filepath}'."
            )

        dist = np.zeros(5, dtype=np.float32)
        dist_elem = root.find(".//distortion_coefficients")
        if dist_elem is not None:
            data_elem = dist_elem.find(".//data")
            if data_elem is not None and data_elem.text:
                vals = [float(x) for x in data_elem.text.split()]
                n = min(len(vals), 5)
                dist[:n] = vals[:n]

        return K, dist

    @staticmethod
    def _extract_binary_element(root: ET.Element, tag: str) -> bytes:
        """Extract base64-decoded binary data from an OpenCV XML element."""
        elem = root.find(f".//{tag}")
        if elem is None:
            raise ValueError(f"<{tag}> element not found in XML")
        data_elem = elem.find(".//data")
        if data_elem is None or data_elem.get("type_id") != "binary":
            raise ValueError(f"<{tag}> data is not binary format")
        return base64.b64decode(data_elem.text.strip())

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _encode_jpeg(self, image: np.ndarray) -> bytes:
        success, buf = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not success:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()
