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

# calibration XML stores binary data as base64 text — needs decoding
import base64
import logging

# unpack binary rvec/tvec bytes from XML into Python floats
import struct

# parse calibration XML files
import xml.etree.ElementTree as ET

# file path building (/ operator for joining paths)
from pathlib import Path

# read images, encode JPEG, convert rvec→R via Rodrigues
import cv2

# matrix operations
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

    def __init__(
        self, dataset_path: str, drone_id: int, jpeg_quality: int = 85
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.drone_id = drone_id
        self.jpeg_quality = jpeg_quality

        # / joins path segments (Path operator), f"D{drone_id}" → "D1", "D2", etc.
        self._frames_dir = self.dataset_path / "image_subsets" / f"D{drone_id}"
        # folder containing per-frame R,t XML files
        self._extr_dir = self.dataset_path / "calibrations" / "extrinsic"
        # folder containing per-frame K,dist XML files
        self._intr_dir = self.dataset_path / "calibrations" / "intrinsic"

        # Fail fast with clear messages — better than cryptic errors deep in load_frame
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
        # glob finds files matching pattern — [0-9] = any digit, matches 0000.png, 0001.png, etc.
        # sorted() ensures correct order (glob order is filesystem-dependent)
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
        # :04d = zero-padded 4-digit int (e.g. 1 → "0001") to match dataset filenames
        frame_path = self._frames_dir / f"{frame_idx:04d}.png"

        if not frame_path.exists():
            raise FileNotFoundError(f"Frame not found: {frame_path}")

        # str() because cv2 is a C library — doesn't accept Path objects, only plain strings
        image = cv2.imread(str(frame_path))

        # cv2 is C-based — returns None on failure instead of raising an exception, must check manually
        if image is None:
            raise ValueError(f"cv2.imread returned None for: {frame_path}")

        # compress PNG → JPEG to reduce packet size
        jpeg_bytes = self._encode_jpeg(image)
        # load K matrix + distortion from intrinsic XML
        K, dist = self._load_intrinsic(frame_idx)
        # load R matrix + t vector from extrinsic XML (rvec converted to R via Rodrigues)
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

        # parse XML file into memory tree (str() because ET is C-based)
        tree = ET.parse(str(filepath))
        # get top-level XML element — starting point for all searches
        root = tree.getroot()

        # find <rvec> element and decode its base64 binary data
        rvec_binary = self._extract_binary_element(root, "rvec")
        # find <tvec> element and decode its base64 binary data
        tvec_binary = self._extract_binary_element(root, "tvec")

        # Each is 3 float64 values = 24 bytes; dataset stores them with a prefix,
        # so we take the last 24 bytes to skip the prefix
        # d = float64 (8 bytes), ddd = 3 float64s = 24 bytes
        rvec_vals = struct.unpack("ddd", rvec_binary[-24:])
        tvec_vals = struct.unpack("ddd", tvec_binary[-24:])

        # reshape to (3,1) column vectors — cv2.Rodrigues and matrix math (R @ t) require column vectors not flat arrays
        rvec = np.array(rvec_vals, dtype=np.float64).reshape(3, 1)
        tvec = np.array(tvec_vals, dtype=np.float64).reshape(3, 1)

        # Rodrigues converts rotation vector (3 values = axis * angle) → 3x3 rotation matrix
        # _ discards the jacobian (derivative info we don't need)
        R, _ = cv2.Rodrigues(rvec)
        # convert to float32 — packet builder and pipeline expect float32
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

        # Start with None — if XML parsing fails below, we can detect it and raise a clear error
        K = None
        # .// means "search anywhere in the tree" (not just direct children)
        cam_elem = root.find(".//camera_matrix")
        if cam_elem is not None:
            data_elem = cam_elem.find(".//data")
            if data_elem is not None and data_elem.text:
                # split() splits whitespace-separated numbers into a list of strings, then convert each to float
                raw = data_elem.text.split()
                vals = []
                for x in raw:
                    vals.append(float(x))
                # reshape flat 9-value list → 3x3 matrix
                K = np.array(vals, dtype=np.float32).reshape(3, 3)

        if K is None:
            raise ValueError(
                f"Drone{self.drone_id} frame {frame_idx}: could not parse K from '{filepath}'."
            )

        # default to zeros if distortion element missing (some cameras have no distortion)
        dist = np.zeros(5, dtype=np.float32)
        dist_elem = root.find(".//distortion_coefficients")
        if dist_elem is not None:
            data_elem = dist_elem.find(".//data")
            if data_elem is not None and data_elem.text:
                raw = data_elem.text.split()
                vals = []
                for x in raw:
                    vals.append(float(x))
                # min() guards against XML having fewer than 5 coefficients
                n = min(len(vals), 5)
                dist[:n] = vals[:n]

        return K, dist

    @staticmethod
    def _extract_binary_element(root: ET.Element, tag: str) -> bytes:
        """Extract base64-decoded binary data from an OpenCV XML element."""
        # .//{tag} searches anywhere in tree for element with this tag name
        elem = root.find(f".//{tag}")
        if elem is None:
            raise ValueError(f"<{tag}> element not found in XML")
        data_elem = elem.find(".//data")
        if data_elem is None or data_elem.get("type_id") != "binary":
            raise ValueError(f"<{tag}> data is not binary format")
        # strip() removes whitespace/newlines around the base64 text before decoding
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
        # tobytes() converts numpy buffer → plain bytes for sending over network
        return buf.tobytes()
