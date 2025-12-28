"""
Protocol Configuration

Binary protocol constants for TCP packet encoding/decoding.
These constants are used by protocol_encoder.py and should match
the client's protocol_decoder expectations.
"""


class ProtocolConfig:
    """Protocol configuration constants"""

    # Drone ID range
    MIN_DRONE_ID = 1
    MAX_DRONE_ID = 8

    # Frame number range
    MIN_FRAME_NUM = 0
    MAX_FRAME_NUM = 999

    # Calibration binary sizes (bytes)
    RVEC_BINARY_SIZE = 24  # 3 doubles × 8 bytes = 24 bytes (rotation vector)
    TVEC_BINARY_SIZE = 24  # 3 doubles × 8 bytes = 24 bytes (translation vector)
    EXTRINSIC_BINARY_SIZE = RVEC_BINARY_SIZE + TVEC_BINARY_SIZE  # 48 bytes total

    # JPEG encoding quality (0-100, higher = better quality but larger size)
    JPEG_QUALITY = 85
