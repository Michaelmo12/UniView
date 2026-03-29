"""
Ingestion Stage

Handles receiving, decoding, and synchronizing frames from multiple drones.

Components:
- models: Data structures (DroneFrame, CameraCalibration, SynchronizedFrameSet)
- enet_receiver: ENet reliable-UDP connections and frame reception
- synchronizer: Frame synchronization across drones
"""

from src.ingestion.models import (
    CameraCalibration,
    DroneFrame,
    SynchronizedFrameSet,
)

from src.ingestion.enet_receiver import (
    ENetReceiver,
    create_enet_receivers,
    start_all_enet_receivers,
    stop_all_enet_receivers,
)

from src.ingestion.synchronizer import (
    FrameSynchronizer,
)

__all__ = [
    # Models
    "CameraCalibration",
    "DroneFrame",
    "SynchronizedFrameSet",
    # ENet Receiver
    "ENetReceiver",
    "create_enet_receivers",
    "start_all_enet_receivers",
    "stop_all_enet_receivers",
    # Synchronizer
    "FrameSynchronizer",
]
