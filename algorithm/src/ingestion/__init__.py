"""
Ingestion Stage

Handles receiving, decoding, and synchronizing frames from multiple drones.

Components:
- models:           Data structures (DroneFrame, CameraCalibration, SynchronizedFrameSet)
- synced_receiver:  Single-thread ENet receiver + frame synchronizer (runs in child process)
- receiver_process: Manages the child process lifecycle
"""

from src.ingestion.models import (
    CameraCalibration,
    DroneFrame,
    SynchronizedFrameSet,
)

from src.ingestion.synced_receiver import (
    DroneState,
    SyncedENetReceiver,
    create_synced_receiver,
)

from src.ingestion.receiver_process import (
    ReceiverProcess,
    create_receiver_process,
)

__all__ = [
    # Models
    "CameraCalibration",
    "DroneFrame",
    "SynchronizedFrameSet",
    # Synced receiver (runs inside child process)
    "DroneState",
    "SyncedENetReceiver",
    "create_synced_receiver",
    # Process wrapper (used by main pipeline)
    "ReceiverProcess",
    "create_receiver_process",
]
