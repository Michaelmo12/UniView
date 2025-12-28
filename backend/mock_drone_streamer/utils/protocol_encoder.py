"""
Binary Protocol Encoder for Mock Drone TCP Streaming

Encodes frames and calibration data into binary packets for TCP transmission.

Packet Structure:
    Header (HEADER_SIZE bytes):
        - drone_id: 1 byte (unsigned char, MIN_DRONE_ID to MAX_DRONE_ID)
        - frame_num: 4 bytes (unsigned int, MIN_FRAME_NUM to MAX_FRAME_NUM)
        - jpeg_size: 4 bytes (unsigned int)
        - timestamp: 8 bytes (double, time.time())

    JPEG Image (variable, ~200KB):
        - JPEG-encoded frame bytes

    Calibration (raw format for network efficiency):
        - Extrinsic binary (EXTRINSIC_BINARY_SIZE bytes):
            * rvec_binary: RVEC_BINARY_SIZE bytes (3 doubles, rotation vector)
            * tvec_binary: TVEC_BINARY_SIZE bytes (3 doubles, translation vector)
        - Intrinsic header (INTRINSIC_HEADER_SIZE bytes):
            * K_text_size: 4 bytes (unsigned int)
            * dist_text_size: 4 bytes (unsigned int)
        - Intrinsic data (variable, ~200 bytes):
            * K_text: variable UTF-8 string (space-separated floats)
            * dist_text: variable UTF-8 string (space-separated floats)

Total: ~200KB per packet (varies by JPEG compression)

Functions:
    pack_packet_raw(drone_id, frame_num, frame, rvec_binary, tvec_binary, K_text, dist_text)
        - Encode frame + raw calibration into binary packet

    get_packet_info(packet)
        - Get packet metadata without full decoding
"""

import struct
import cv2
import numpy as np
import time

# Import protocol configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ProtocolConfig

# Struct formats
HEADER_FORMAT = 'B I I d'  # drone_id, frame_num, jpeg_size, timestamp
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# Raw calibration sizes
INTRINSIC_HEADER_SIZE = 8  # 4 bytes K_text_size + 4 bytes dist_text_size

# Minimum calibration size (doesn't include variable text data)
CALIBRATION_SIZE = ProtocolConfig.EXTRINSIC_BINARY_SIZE + INTRINSIC_HEADER_SIZE  # 56 bytes


def pack_packet_raw(drone_id, frame_num, frame, rvec_binary, tvec_binary, K_text, dist_text):
    """
    Pack frame + raw calibration data into a single binary packet.

    This function sends RAW calibration data for network efficiency:
    - Extrinsic: EXTRINSIC_BINARY_SIZE bytes binary (RVEC_BINARY_SIZE + TVEC_BINARY_SIZE)
    - Intrinsic: Text strings (space-separated floats)

    The client will decode these raw formats using calibration_loader helper functions.

    Args:
        drone_id: Drone ID (MIN_DRONE_ID to MAX_DRONE_ID)
        frame_num: Frame number (MIN_FRAME_NUM to MAX_FRAME_NUM)
        frame: numpy array BGR image (H×W×3)
        rvec_binary: RVEC_BINARY_SIZE bytes (rotation vector in binary format)
        tvec_binary: TVEC_BINARY_SIZE bytes (translation vector in binary format)
        K_text: Camera matrix as text string (or None)
        dist_text: Distortion coefficients as text string (or None)

    Returns:
        bytes: Binary packet ready for TCP transmission

    Raises:
        ValueError: If input validation fails
        cv2.error: If frame encoding fails
    """
    # Validate inputs
    if not ProtocolConfig.MIN_DRONE_ID <= drone_id <= ProtocolConfig.MAX_DRONE_ID:
        raise ValueError(f"drone_id must be {ProtocolConfig.MIN_DRONE_ID}-{ProtocolConfig.MAX_DRONE_ID}, got {drone_id}")
    if not ProtocolConfig.MIN_FRAME_NUM <= frame_num <= ProtocolConfig.MAX_FRAME_NUM:
        raise ValueError(f"frame_num must be {ProtocolConfig.MIN_FRAME_NUM}-{ProtocolConfig.MAX_FRAME_NUM}, got {frame_num}")
    if frame is None or frame.size == 0:
        raise ValueError("frame is empty or None")
    if len(rvec_binary) != ProtocolConfig.RVEC_BINARY_SIZE:
        raise ValueError(f"rvec_binary must be {ProtocolConfig.RVEC_BINARY_SIZE} bytes, got {len(rvec_binary)}")
    if len(tvec_binary) != ProtocolConfig.TVEC_BINARY_SIZE:
        raise ValueError(f"tvec_binary must be {ProtocolConfig.TVEC_BINARY_SIZE} bytes, got {len(tvec_binary)}")

    # Step 1: Encode frame to JPEG
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, ProtocolConfig.JPEG_QUALITY]
    success, jpeg_buffer = cv2.imencode('.jpg', frame, encode_params)
    if not success:
        raise cv2.error("Failed to encode frame to JPEG")

    jpeg_bytes = jpeg_buffer.tobytes()
    jpeg_size = len(jpeg_bytes)

    # Step 2: Pack header
    timestamp = time.time()
    header = struct.pack(HEADER_FORMAT, drone_id, frame_num, jpeg_size, timestamp)

    # Step 3: Pack calibration data (raw format)

    # Extrinsic: binary (48 bytes total)
    extrinsic_binary = rvec_binary + tvec_binary

    # Intrinsic: text format (variable length)
    # Handle None values
    if K_text is not None:
        K_text_bytes = K_text.encode('utf-8')
    else:
        K_text_bytes = b''
    if dist_text is not None:
        dist_text_bytes = dist_text.encode('utf-8')
    else:
        dist_text_bytes = b''

    K_text_size = len(K_text_bytes)
    dist_text_size = len(dist_text_bytes)

    # Pack intrinsic sizes + data
    intrinsic_header = struct.pack('I I', K_text_size, dist_text_size)
    intrinsic_data = K_text_bytes + dist_text_bytes

    # Step 4: Combine all parts
    packet = header + jpeg_bytes + extrinsic_binary + intrinsic_header + intrinsic_data

    return packet


def get_packet_info(packet):
    """
    Get packet information without full unpacking (for debugging).

    Args:
        packet: bytes packet data

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
