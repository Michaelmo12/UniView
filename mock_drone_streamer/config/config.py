"""
StreamerConfig - Configuration dataclass for the ENet drone streamer.

Each running instance represents a single drone streamer process.
Port is derived from base_port + drone_id - 1 so all 8 drones
use distinct ports (16000-16007 by default).
"""

from dataclasses import dataclass


@dataclass
class StreamerConfig:
    """
    Configuration for a single ENet drone streamer instance.

    Attributes:
        dataset_path: Root path to the MATRIX dataset directory.
                      Expected structure: {dataset_path}/Drone{N}/frames/ and calibration/
        drone_id:     Which drone this instance streams (1-8).
        host:         ENet bind address. Use "0.0.0.0" to accept connections from any host.
        base_port:    Base UDP port. Drone N listens on base_port + N - 1.
                      Default 16000 avoids conflict with TCP streamer (15000 range).
        fps:          Target frame send rate in frames per second.
        jpeg_quality: JPEG compression quality (0-100). Higher = better quality, larger packets.
        loop:         When True, loop dataset frames when the end is reached.
        num_drones:   Total number of drones, used by run_all_drones.py launcher.
    """

    # Root path to the MATRIX dataset folder
    dataset_path: str = "MATRIX"
    # Which drone this instance represents (1-8)
    drone_id: int = 1
    # Bind address — "0.0.0.0" accepts connections from any IP
    host: str = "0.0.0.0"
    # Base port — drone N listens on base_port + N - 1 (drone1=16000, drone2=16001, ...)
    base_port: int = 16000
    # Frames per second to send
    fps: float = 2.0
    # JPEG quality 0-100 — higher = better image but larger packet size
    jpeg_quality: int = 85
    # When True, restart from frame 0 when dataset is exhausted
    loop: bool = True
    # Total number of drones — used by the launcher to start all instances
    num_drones: int = 8
    # Stop after sending this many frames (0 = unlimited)
    max_frames: int = 0

    @property
    def port(self) -> int:
        # Computed from base_port + drone_id - 1 so it's always consistent
        # A regular field could be set incorrectly — property guarantees correctness
        """Compute the listening port for this drone instance."""
        return self.base_port + (self.drone_id - 1)
