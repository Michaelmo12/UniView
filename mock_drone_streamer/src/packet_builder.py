"""
PacketBuilder - Binary Packet Construction for ENet Drone Streamer

Packet format (little-endian):
    [Header: 13 bytes]
        uint8   drone_id        - Drone identifier (1-8)
        uint32  frame_num       - Frame sequence number
        uint64  timestamp_ns    - Capture timestamp in nanoseconds

    [Calibration: 104 bytes]
        9 x float32  K flat     - 3x3 intrinsic matrix, row-major (36 bytes)
        9 x float32  R flat     - 3x3 rotation matrix, row-major (36 bytes)
        3 x float32  t flat     - 3x1 translation vector (12 bytes)
        5 x float32  dist       - Distortion coefficients k1,k2,p1,p2,k3 (20 bytes)

    [Payload: variable]
        N bytes  jpeg_data      - JPEG-encoded frame

Total fixed portion: 13 + 104 = 117 bytes.
"""

import struct

import numpy as np


class PacketBuilder:
    """
    Builds binary packets for the ENet drone streamer protocol.

    All values are packed in little-endian byte order to match the
    ENetReceiver unpacking format exactly.

    Usage:
        pb = PacketBuilder()
        data = pb.build_packet(drone_id=1, frame_num=42,
                               timestamp_ns=int(time.time_ns()),
                               jpeg_bytes=jpeg, K=K, R=R, t=t, dist=dist)
    """

    # Format strings (little-endian)
    HEADER_FORMAT: str = "<BIQ"          # uint8 + uint32 + uint64 = 13 bytes
    CALIBRATION_FORMAT: str = "<9f9f3f5f"  # 26 x float32 = 104 bytes

    HEADER_SIZE: int = struct.calcsize(HEADER_FORMAT)         # 13
    CALIBRATION_SIZE: int = struct.calcsize(CALIBRATION_FORMAT)  # 104
    FIXED_SIZE: int = HEADER_SIZE + CALIBRATION_SIZE          # 117

    def build_packet(
        self,
        drone_id: int,
        frame_num: int,
        timestamp_ns: int,
        jpeg_bytes: bytes,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        dist: np.ndarray,
    ) -> bytes:
        """
        Build a complete binary packet.

        Args:
            drone_id:     Drone identifier (1-8, fits in uint8).
            frame_num:    Frame sequence number (fits in uint32).
            timestamp_ns: Capture timestamp in nanoseconds (fits in uint64).
            jpeg_bytes:   JPEG-encoded image bytes.
            K:            (3,3) float32 intrinsic matrix.
            R:            (3,3) float32 rotation matrix.
            t:            (3,1) float32 translation vector.
            dist:         (5,)  float32 distortion coefficients.

        Returns:
            Complete packet as bytes: header + calibration + jpeg.
        """
        header = struct.pack(
            self.HEADER_FORMAT,
            int(drone_id),
            int(frame_num),
            int(timestamp_ns),
        )

        # Flatten arrays to 1-D sequences of Python floats for struct.pack
        K_flat = K.flatten().tolist()    # 9 floats
        R_flat = R.flatten().tolist()    # 9 floats
        t_flat = t.flatten().tolist()    # 3 floats
        dist_flat = dist.flatten().tolist()  # 5 floats

        calibration = struct.pack(
            self.CALIBRATION_FORMAT,
            *K_flat,
            *R_flat,
            *t_flat,
            *dist_flat,
        )

        return header + calibration + jpeg_bytes
