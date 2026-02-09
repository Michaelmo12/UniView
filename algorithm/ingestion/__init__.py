"""
Ingestion Stage

Handles receiving, decoding, and synchronizing frames from multiple drones.

Components:
- models: Data structures (DroneFrame, CameraCalibration, SynchronizedFrameSet)
- protocol_decoder: Binary packet parsing
- tcp_receiver: TCP socket connections and frame reception
- synchronizer: Frame synchronization across drones

Usage:
    from ingestion import DroneFrame, CameraCalibration, SynchronizedFrameSet
    from ingestion import ProtocolDecoder, decode_packet
    from ingestion import TCPReceiver, FrameSynchronizer
"""

# Data models - used throughout the pipeline
from ingestion.models import (
    CameraCalibration,
    DroneFrame,
    SynchronizedFrameSet,
)

# Protocol decoder - parses binary packets
from ingestion.protocol_decoder import (
    ProtocolDecoder,
    decode_packet,
    PacketHeader,
    HEADER_SIZE,
    EXTRINSIC_SIZE,
)

# TCP receiver - connects to drone streams
from ingestion.tcp_receiver import (
    TCPReceiver,
    create_receivers,
    start_all_receivers,
    stop_all_receivers,
)

# Synchronizer - groups frames across drones
from ingestion.synchronizer import (
    FrameSynchronizer,
)

__all__ = [
    # Models
    "CameraCalibration",
    "DroneFrame",
    "SynchronizedFrameSet",
    # Decoder
    "ProtocolDecoder",
    "decode_packet",
    "PacketHeader",
    "HEADER_SIZE",
    "EXTRINSIC_SIZE",
    # TCP Receiver
    "TCPReceiver",
    "create_receivers",
    "start_all_receivers",
    "stop_all_receivers",
    # Synchronizer
    "FrameSynchronizer",
]
