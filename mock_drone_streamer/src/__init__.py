"""
ENet Drone Streamer - Source Modules

Components:
- dataset_loader: MATRIX dataset frame + calibration loader with synthetic fallback
- packet_builder:  Binary packet construction (13B header + 104B calibration + JPEG)
- streamer:        ENet server that sends packets to connected peers

Note: ENetStreamer is imported lazily below to avoid a hard import-time failure
when the `enet` (pyenet) native library is not installed.  DatasetLoader and
PacketBuilder have no such dependency and always import cleanly.
"""

from mock_drone_streamer.src.dataset_loader import DatasetLoader
from mock_drone_streamer.src.packet_builder import PacketBuilder

__all__ = ["DatasetLoader", "PacketBuilder", "ENetStreamer"]


def __getattr__(name: str):
    if name == "ENetStreamer":
        from mock_drone_streamer.src.streamer import ENetStreamer  # noqa: PLC0415
        return ENetStreamer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
