"""
Binary Protocol Decoder for Mock Drone TCP Streaming

Decodes binary packets received from drone streamers into frames and calibration data.

Packet Structure:
    Header (17 bytes):
        - drone_id: 1 byte (unsigned char, 1-8)
        - frame_num: 4 bytes (unsigned int, 0-999)
        - jpeg_size: 4 bytes (unsigned int)
        - timestamp: 8 bytes (double, time.time())

    JPEG Image (variable, ~200KB):
        - JPEG-encoded frame bytes

    Calibration (raw format):
        - Extrinsic binary (48 bytes):
            * rvec_binary: 24 bytes (3 doubles, rotation vector)
            * tvec_binary: 24 bytes (3 doubles, translation vector)
        - Intrinsic text (variable, ~200 bytes):
            * K_text_size: 4 bytes (unsigned int)
            * dist_text_size: 4 bytes (unsigned int)
            * K_text: variable UTF-8 string (space-separated floats)
            * dist_text: variable UTF-8 string (space-separated floats)

Functions:
    - recv_exact: TCP helper to receive exact number of bytes
    - unpack_packet_raw: Decode binary packet with raw calibration data
    - decode_rvec_binary: Decode rotation vector from 24 bytes
    - decode_tvec_binary: Decode translation vector from 24 bytes
    - decode_rotation_matrix: Convert rvec to rotation matrix
    - decode_translation_vector: Convert tvec to float32
    - decode_camera_matrix: Decode camera matrix from text
    - decode_distortion_coefficients: Decode distortion from text
    - get_packet_info: Get packet metadata without full decoding
"""

import struct
import sys
from pathlib import Path
import cv2
import numpy as np

# Import decoder functions from server-side calibration_loader
# Go from algorithm/stream_receiver/utils/ up to project root
_project_root = Path(__file__).parent.parent.parent.parent
_calibration_loader_path = _project_root / "mock_drone_streamer" / "utils" / "calibration_loader.py"

# Add mock_drone_streamer to path if needed
if str(_project_root / "mock_drone_streamer") not in sys.path:
    sys.path.insert(0, str(_project_root / "mock_drone_streamer"))

# Now import from utils.calibration_loader
try:
    from utils.calibration_loader import (
        decode_rvec_binary,
        decode_tvec_binary,
        decode_rotation_matrix,
        decode_translation_vector,
        decode_camera_matrix,
        decode_distortion_coefficients
    )
except ModuleNotFoundError:
    # Fallback: import directly using spec
    import importlib.util
    spec = importlib.util.spec_from_file_location("calibration_loader", _calibration_loader_path)
    calibration_loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calibration_loader)

    decode_rvec_binary = calibration_loader.decode_rvec_binary
    decode_tvec_binary = calibration_loader.decode_tvec_binary
    decode_rotation_matrix = calibration_loader.decode_rotation_matrix
    decode_translation_vector = calibration_loader.decode_translation_vector
    decode_camera_matrix = calibration_loader.decode_camera_matrix
    decode_distortion_coefficients = calibration_loader.decode_distortion_coefficients

# Struct formats
HEADER_FORMAT = 'B I I d'  # drone_id, frame_num, jpeg_size, timestamp
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 17 bytes

# Raw calibration sizes
EXTRINSIC_BINARY_SIZE = 48  # 24 bytes rvec + 24 bytes tvec
INTRINSIC_HEADER_SIZE = 8  # 4 bytes K_text_size + 4 bytes dist_text_size

# Keep old constant for backward compatibility with client.py
CALIBRATION_SIZE = EXTRINSIC_BINARY_SIZE + INTRINSIC_HEADER_SIZE  # Minimum calibration size


def recv_exact(sock, n):
    """
    Receive exactly n bytes from TCP socket.

    This helper ensures complete packet reception over TCP by reading
    in a loop until exactly n bytes are received.

    Args:
        sock: TCP socket (SOCK_STREAM)
        n: Number of bytes to receive

    Returns:
        bytes: Exactly n bytes

    Raises:
        ConnectionError: If socket closes before n bytes received
    """
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket connection closed before receiving all data")
        data += chunk
    return data


def unpack_packet_raw(packet):
    """
    Unpack binary TCP packet with RAW calibration data into frame + calibration.

    This function decodes packets that contain:
    - Raw binary extrinsic data (48 bytes: rvec + tvec)
    - Raw text intrinsic data (variable length: K_text + dist_text)

    The client-side decoding uses the same helper functions from calibration_loader.py

    Args:
        packet: bytes received from TCP socket

    Returns:
        tuple: (drone_id, frame_num, frame, K, R, t, dist, timestamp)
            - drone_id: int (1-8)
            - frame_num: int (0-999)
            - frame: numpy array BGR image (H×W×3)
            - K: Camera intrinsic matrix (3×3 numpy array, float32)
            - R: Rotation matrix (3×3 numpy array, float32)
            - t: Translation vector (3×1 numpy array, float32)
            - dist: Distortion coefficients (5, numpy array, float32)
            - timestamp: float (time.time() when packet was created)

    Raises:
        ValueError: If packet structure is invalid
        cv2.error: If JPEG decoding fails
    """
    # Validate minimum packet size
    min_size = HEADER_SIZE + EXTRINSIC_BINARY_SIZE + INTRINSIC_HEADER_SIZE
    if len(packet) < min_size:
        raise ValueError(f"Packet too small: {len(packet)} < {min_size} bytes")

    # Step 1: Unpack header
    header_data = packet[:HEADER_SIZE]
    drone_id, frame_num, jpeg_size, timestamp = struct.unpack(HEADER_FORMAT, header_data)

    # Validate header values
    if not 1 <= drone_id <= 8:
        raise ValueError(f"Invalid drone_id in packet: {drone_id}")
    if not 0 <= frame_num <= 999:
        raise ValueError(f"Invalid frame_num in packet: {frame_num}")

    # Step 2: Extract JPEG bytes
    jpeg_start = HEADER_SIZE
    jpeg_end = jpeg_start + jpeg_size

    jpeg_bytes = packet[jpeg_start:jpeg_end]

    # Decode JPEG to image
    jpeg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise cv2.error("Failed to decode JPEG frame")

    # Step 3: Extract extrinsic binary data (48 bytes)
    extr_start = jpeg_end
    extr_end = extr_start + EXTRINSIC_BINARY_SIZE

    if extr_end > len(packet):
        raise ValueError(f"Packet too small for extrinsic data")

    rvec_binary = packet[extr_start:extr_start + 24]
    tvec_binary = packet[extr_start + 24:extr_end]

    # Decode extrinsic (rvec, tvec → R, t)
    rvec = decode_rvec_binary(rvec_binary)
    tvec = decode_tvec_binary(tvec_binary)
    R = decode_rotation_matrix(rvec)
    t = decode_translation_vector(tvec)

    # Step 4: Extract intrinsic text data (variable length)
    intr_header_start = extr_end
    intr_header_end = intr_header_start + INTRINSIC_HEADER_SIZE

    if intr_header_end > len(packet):
        raise ValueError(f"Packet too small for intrinsic header")

    K_text_size, dist_text_size = struct.unpack('I I', packet[intr_header_start:intr_header_end])

    # Extract text strings
    intr_data_start = intr_header_end
    K_text_end = intr_data_start + K_text_size
    dist_text_end = K_text_end + dist_text_size

    if dist_text_end > len(packet):
        raise ValueError(f"Packet too small for intrinsic data")

    K_text_bytes = packet[intr_data_start:K_text_end]
    dist_text_bytes = packet[K_text_end:dist_text_end]

    # Decode intrinsic (text → K, dist)
    if K_text_size > 0:
        K_text = K_text_bytes.decode('utf-8')
        K = decode_camera_matrix(K_text)
    else:
        K = None

    if dist_text_size > 0:
        dist_text = dist_text_bytes.decode('utf-8')
        dist = decode_distortion_coefficients(dist_text)
    else:
        dist = np.zeros(5, dtype=np.float32)

    return drone_id, frame_num, frame, K, R, t, dist, timestamp


# Keep old function name for backward compatibility
def unpack_packet(packet):
    """
    Wrapper for unpack_packet_raw to maintain backward compatibility.
    """
    return unpack_packet_raw(packet)


def get_packet_info(packet):
    """
    Get packet information without full unpacking (for debugging).

    Args:
        packet: bytes received from TCP socket

    Returns:
        dict: Packet metadata (drone_id, frame_num, jpeg_size, timestamp, total_size)
              or None if packet is too small
    """
    if len(packet) < HEADER_SIZE:
        return None

    drone_id, frame_num, jpeg_size, timestamp = struct.unpack(HEADER_FORMAT, packet[:HEADER_SIZE])

    return {
        'drone_id': drone_id,
        'frame_num': frame_num,
        'jpeg_size': jpeg_size,
        'timestamp': timestamp,
        'total_size': len(packet),
        'expected_size': HEADER_SIZE + jpeg_size + CALIBRATION_SIZE
    }
