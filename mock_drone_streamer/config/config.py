"""
StreamerConfig - Configuration dataclass for the ENet drone streamer.

Each running instance represents a single drone streamer process.
Port is derived from base_port + drone_id - 1 so all 8 drones
use distinct ports (16000-16007 by default).
"""

from dataclasses import dataclass, field


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

    dataset_path: str = "MATRIX"
    drone_id: int = 1
    host: str = "0.0.0.0"
    base_port: int = 16000
    fps: float = 2.0
    jpeg_quality: int = 85
    loop: bool = True
    num_drones: int = 8

    @property
    def port(self) -> int:
        """Compute the listening port for this drone instance."""
        return self.base_port + (self.drone_id - 1)
