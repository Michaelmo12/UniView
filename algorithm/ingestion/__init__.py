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
    from ingestion import ProtocolDecoder
    from ingestion import TCPReceiver, FrameSynchronizer
"""

# Data models - used throughout the pipeline
from algorithm.ingestion.models import (
    CameraCalibration,
    DroneFrame,
    SynchronizedFrameSet,
)

# Protocol decoder - parses binary packets
from algorithm.ingestion.protocol_decoder import (
    ProtocolDecoder,
    PacketHeader,
    HEADER_SIZE,
    EXTRINSIC_SIZE,
)

# TCP receiver - connects to drone streams
from algorithm.ingestion.tcp_receiver import (
    TCPReceiver,
    create_receivers,
    start_all_receivers,
    stop_all_receivers,
)

# Synchronizer - groups frames across drones
from algorithm.ingestion.synchronizer import (
    FrameSynchronizer,
)

__all__ = [
    # Models
    "CameraCalibration",
    "DroneFrame",
    "SynchronizedFrameSet",
    # Decoder
    "ProtocolDecoder",
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
