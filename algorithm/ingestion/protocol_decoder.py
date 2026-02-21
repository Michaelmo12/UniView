"""
Binary Protocol Decoder

Parses TCP packets from mock_drone_streamer into DroneFrame objects.

Protocol Structure (from mock_drone_streamer documentation):
============================================================

┌─────────────────────────────────────────────────────────────┐
│ HEADER (17 bytes)                                           │
├─────────────────────────────────────────────────────────────┤
│ drone_id     │ 1 byte  │ unsigned char │ Value 1-8          │
│ frame_num    │ 4 bytes │ unsigned int  │ Value 0-999        │
│ jpeg_size    │ 4 bytes │ unsigned int  │ JPEG byte count    │
│ timestamp    │ 8 bytes │ double        │ Unix timestamp     │
├─────────────────────────────────────────────────────────────┤
│ JPEG IMAGE (variable, ~80-200KB)                            │
├─────────────────────────────────────────────────────────────┤
│ jpeg_bytes   │ [jpeg_size] bytes │ Raw JPEG data            │
├─────────────────────────────────────────────────────────────┤
│ EXTRINSIC BINARY (48 bytes)                                 │
├─────────────────────────────────────────────────────────────┤
│ rvec         │ 24 bytes │ 3 doubles │ Rotation vector       │
│ tvec         │ 24 bytes │ 3 doubles │ Translation vector    │
├─────────────────────────────────────────────────────────────┤
│ INTRINSIC TEXT (variable, ~200 bytes)                       │
├─────────────────────────────────────────────────────────────┤
│ K_text_size  │ 4 bytes │ unsigned int │ Length of K string  │
│ dist_text_size│ 4 bytes │ unsigned int │ Length of dist str │
│ K_text       │ [K_text_size] bytes │ Space-separated floats │
│ dist_text    │ [dist_text_size] bytes │ Space-separated     │
└─────────────────────────────────────────────────────────────┘

"""

import struct
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from algorithm.ingestion.models import CameraCalibration, DroneFrame

# Header format string for struct.unpack()
# B = unsigned char (1 byte)  → drone_id
# I = unsigned int (4 bytes)  → frame_num
# I = unsigned int (4 bytes)  → jpeg_size
# d = double (8 bytes)        → timestamp
HEADER_FORMAT = "B I I d"

# Calculate header size from format string
# struct.calcsize() returns the number of bytes for a format
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # = 17 bytes

# Extrinsic data sizes
RVEC_SIZE = 24  # 3 doubles × 8 bytes = 24 bytes
TVEC_SIZE = 24  # 3 doubles × 8 bytes = 24 bytes
EXTRINSIC_SIZE = RVEC_SIZE + TVEC_SIZE  # = 48 bytes

# Intrinsic header size (two unsigned ints for text lengths)
INTRINSIC_HEADER_FORMAT = "I I"
INTRINSIC_HEADER_SIZE = struct.calcsize(INTRINSIC_HEADER_FORMAT)  # = 8 bytes


@dataclass
class PacketHeader:
    """
    Parsed header from a TCP packet.

    This intermediate structure holds header fields before we parse
    the rest of the packet. Useful for validation and debugging.
    """

    drone_id: int  # 1-8
    frame_num: int  # 0-999
    jpeg_size: int  # Size of JPEG data in bytes
    timestamp: float  # Unix timestamp


class ProtocolDecoder:
    """
    Decodes binary TCP packets into DroneFrame objects.

    This class handles the complete decoding pipeline:
    1. Parse header to get sizes and metadata
    2. Extract and decode JPEG image
    3. Extract and decode extrinsic parameters (rvec → R, tvec → t)
    4. Extract and decode intrinsic parameters (K matrix, distortion)
    5. Assemble into DroneFrame with CameraCalibration

    Thread Safety:
        This class is stateless and thread-safe. Multiple threads can
        share the same decoder instance or create their own.
    """

    def decode_packet(self, packet: bytes) -> DroneFrame:
        # Step 1: Validate minimum packet size
        # We need at least: header + extrinsic + intrinsic header
        min_size = HEADER_SIZE + EXTRINSIC_SIZE + INTRINSIC_HEADER_SIZE
        if len(packet) < min_size:
            raise ValueError(
                f"Packet too small: {len(packet)} bytes, "
                f"minimum required: {min_size} bytes"
            )

        # Step 2: Parse header
        header = self._parse_header(packet)

        # Step 3: Validate header values
        self._validate_header(header)

        # Step 4: Calculate byte offsets for each section
        # The packet is laid out sequentially:
        #   [header][jpeg][extrinsic][intrinsic_header][intrinsic_text]

        jpeg_start = HEADER_SIZE
        jpeg_end = jpeg_start + header.jpeg_size

        extrinsic_start = jpeg_end
        extrinsic_end = extrinsic_start + EXTRINSIC_SIZE

        intrinsic_header_start = extrinsic_end
        intrinsic_header_end = intrinsic_header_start + INTRINSIC_HEADER_SIZE

        # Step 5: Validate packet has enough bytes for JPEG
        if jpeg_end > len(packet):
            raise ValueError(
                f"Packet too small for JPEG: need {jpeg_end} bytes, "
                f"got {len(packet)} bytes"
            )

        # Step 6: Decode JPEG image
        jpeg_bytes = packet[jpeg_start:jpeg_end]
        frame = self._decode_jpeg(jpeg_bytes)

        # Step 7: Decode extrinsic parameters (R, t)
        extrinsic_bytes = packet[extrinsic_start:extrinsic_end]
        R, t = self._decode_extrinsic(extrinsic_bytes)

        # Step 8: Parse intrinsic header to get text sizes
        intrinsic_header_bytes = packet[intrinsic_header_start:intrinsic_header_end]
        K_text_size, dist_text_size = struct.unpack(
            INTRINSIC_HEADER_FORMAT, intrinsic_header_bytes
        )

        # Step 9: Extract intrinsic text data
        K_text_start = intrinsic_header_end
        K_text_end = K_text_start + K_text_size
        dist_text_start = K_text_end
        dist_text_end = dist_text_start + dist_text_size

        # Validate packet has enough bytes for intrinsic text
        if dist_text_end > len(packet):
            raise ValueError(
                f"Packet too small for intrinsic data: need {dist_text_end} bytes, "
                f"got {len(packet)} bytes"
            )

        K_text_bytes = packet[K_text_start:K_text_end]
        dist_text_bytes = packet[dist_text_start:dist_text_end]

        # Step 10: Decode intrinsic parameters (K, dist)
        K, dist = self._decode_intrinsic(K_text_bytes, dist_text_bytes)

        # Step 11: Create CameraCalibration object
        calibration = CameraCalibration(K=K, R=R, t=t, dist=dist)

        # Step 12: Create and return DroneFrame
        return DroneFrame(
            drone_id=header.drone_id,
            frame_num=header.frame_num,
            timestamp=header.timestamp,
            frame=frame,
            calibration=calibration,
        )

    def _parse_header(self, packet: bytes) -> PacketHeader:
        # Extract just the header bytes
        header_bytes = packet[:HEADER_SIZE]

        # Unpack according to format string
        # struct.unpack returns a tuple of values
        drone_id, frame_num, jpeg_size, timestamp = struct.unpack(
            HEADER_FORMAT, header_bytes
        )

        return PacketHeader(
            drone_id=drone_id,
            frame_num=frame_num,
            jpeg_size=jpeg_size,
            timestamp=timestamp,
        )

    def _validate_header(self, header: PacketHeader) -> None:
        if not 1 <= header.drone_id <= 8:
            raise ValueError(f"Invalid drone_id: {header.drone_id}, expected 1-8")

        if not 0 <= header.frame_num <= 999:
            raise ValueError(f"Invalid frame_num: {header.frame_num}, expected 0-999")

        if header.jpeg_size < 1000:
            raise ValueError(f"JPEG size too small: {header.jpeg_size} bytes")
        if header.jpeg_size > 10_000_000:
            raise ValueError(f"JPEG size too large: {header.jpeg_size} bytes")

    def _decode_jpeg(self, jpeg_bytes: bytes) -> np.ndarray:
        jpeg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)

        # cv2.imdecode() decodes the JPEG
        # cv2.IMREAD_COLOR = load as 3-channel BGR (ignore alpha if present)
        frame = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)

        if frame is None:
            raise cv2.error("Failed to decode JPEG image")

        return frame

    def _decode_extrinsic(
        self, extrinsic_bytes: bytes
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode extrinsic parameters from 48 bytes.

        The extrinsic data contains:
        - rvec (24 bytes): Rotation vector (Rodrigues form)
        - tvec (24 bytes): Translation vector

        We convert rvec to rotation matrix R using cv2.Rodrigues().

        Args:
            extrinsic_bytes: 48 bytes of extrinsic data

        Returns:
            Tuple of (R, t):
            - R: Rotation matrix (3, 3) float32
            - t: Translation vector (3, 1) float32
        """
        rvec_bytes = extrinsic_bytes[:RVEC_SIZE]
        tvec_bytes = extrinsic_bytes[RVEC_SIZE:]

        # Unpack rvec: 3 doubles
        # 'd' = double (8 bytes), 'ddd' = 3 doubles
        rvec_values = struct.unpack("ddd", rvec_bytes)
        tvec_values = struct.unpack("ddd", tvec_bytes)

        rvec = np.array(rvec_values, dtype=np.float64).reshape(3, 1)
        tvec = np.array(tvec_values, dtype=np.float64).reshape(3, 1)

        # Convert rotation vector to rotation matrix using Rodrigues formula
        # cv2.Rodrigues(rvec) returns (R, jacobian)
        # We only need R, so we ignore the jacobian
        R, _ = cv2.Rodrigues(rvec)

        # Convert to float32 for consistency with K
        R = R.astype(np.float32)
        t = tvec.astype(np.float32)

        return R, t

    def _decode_intrinsic(
        self, K_text_bytes: bytes, dist_text_bytes: bytes
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode intrinsic parameters from text strings.

        The intrinsic data is transmitted as space-separated float strings:
        - K_text: "fx 0 cx 0 fy cy 0 0 1" (9 values for 3x3 matrix)
        - dist_text: "k1 k2 p1 p2 k3" (5 distortion coefficients)

        Args:
            K_text_bytes: UTF-8 encoded camera matrix string
            dist_text_bytes: UTF-8 encoded distortion coefficients string

        Returns:
            Tuple of (K, dist):
            - K: Camera intrinsic matrix (3, 3) float32
            - dist: Distortion coefficients (5,) float32
        """
        # Decode bytes to string
        K_text = K_text_bytes.decode("utf-8")
        dist_text = dist_text_bytes.decode("utf-8")

        # Parse K matrix from space-separated values
        # "1000.0 0.0 960.0 0.0 1000.0 540.0 0.0 0.0 1.0"
        # split() without arguments splits on any whitespace
        K_values = [float(x) for x in K_text.split()]

        # Reshape 9 values into 3x3 matrix
        K = np.array(K_values, dtype=np.float32).reshape(3, 3)

        # Parse distortion coefficients
        # "0.0 0.0 0.0 0.0 0.0"
        dist_values = [float(x) for x in dist_text.split()]

        # Ensure exactly 5 coefficients (pad with zeros if needed)
        if len(dist_values) >= 5:
            dist = np.array(dist_values[:5], dtype=np.float32)
        else:
            # Pad with zeros if fewer than 5 values
            dist = np.zeros(5, dtype=np.float32)
            dist[: len(dist_values)] = dist_values

        return K, dist


if __name__ == "__main__":
    print("Testing Protocol Decoder")
    print("=" * 50)

    # 1. Create header
    drone_id = 1
    frame_num = 42
    timestamp = 1234567890.123

    # 2. Create a small test image and encode as JPEG
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[25:75, 25:75] = [0, 255, 0]  # Green square

    # cv2.imencode returns (success, buffer)
    success, jpeg_buffer = cv2.imencode(".jpg", test_image)
    jpeg_bytes = jpeg_buffer.tobytes()
    jpeg_size = len(jpeg_bytes)

    print(f"Test image: 100x100, JPEG size: {jpeg_size} bytes")

    # 3. Create mock extrinsic data
    rvec = np.array([0.1, 0.2, 0.3], dtype=np.float64)  # Small rotation
    tvec = np.array([1.0, 2.0, 5.0], dtype=np.float64)  # 5m away

    rvec_bytes = struct.pack("ddd", *rvec)
    tvec_bytes = struct.pack("ddd", *tvec)
    extrinsic_bytes = rvec_bytes + tvec_bytes

    # 4. Create mock intrinsic data
    K_text = "1000.0 0.0 50.0 0.0 1000.0 50.0 0.0 0.0 1.0"
    dist_text = "0.0 0.0 0.0 0.0 0.0"

    K_text_bytes = K_text.encode("utf-8")
    dist_text_bytes = dist_text.encode("utf-8")

    intrinsic_header = struct.pack("II", len(K_text_bytes), len(dist_text_bytes))

    # 5. Assemble complete packet
    header = struct.pack(HEADER_FORMAT, drone_id, frame_num, jpeg_size, timestamp)

    packet = (
        header
        + jpeg_bytes
        + extrinsic_bytes
        + intrinsic_header
        + K_text_bytes
        + dist_text_bytes
    )

    print(f"Total packet size: {len(packet)} bytes")
    print()

    # 6. Decode the packet
    print("Decoding packet...")
    decoder = ProtocolDecoder()

    try:
        drone_frame = decoder.decode_packet(packet)

        print("\nDecoded DroneFrame:")
        print(f"  drone_id: {drone_frame.drone_id}")
        print(f"  frame_num: {drone_frame.frame_num}")
        print(f"  timestamp: {drone_frame.timestamp}")
        print(f"  frame shape: {drone_frame.frame.shape}")
        print(f"  K[0,0] (fx): {drone_frame.calibration.K[0,0]}")
        print(f"  camera_center: {drone_frame.calibration.camera_center}")

        print("\n" + "=" * 50)
        print("All tests passed!")

    except Exception as e:
        print(f"ERROR: {e}")
        raise
