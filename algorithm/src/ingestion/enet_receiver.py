"""
ENet Frame Receiver

Connects to an ENet drone streamer (enet_drone_streamer microservice) and receives
DroneFrame packets over reliable UDP.

Architecture:
- One ENetReceiver instance per drone (8 total for MATRIX)
- Each receiver runs in its own thread (daemon=True)
- Connects as ENet CLIENT to the streamer server
- Receives binary packets, parses them into DroneFrame + CameraCalibration objects
- Pushes decoded frames to output_queue for downstream processing
- Auto-reconnects on disconnect with configurable delay

Packet format (little-endian, matching PacketBuilder in enet_drone_streamer):
    Header (13 bytes):   <BIQ  -> drone_id uint8, frame_num uint32, timestamp_ns uint64
    Calibration (104B):  <9f9f3f5f -> K(9f) R(9f) t(3f) dist(5f)
    Payload (variable):  JPEG bytes

Public API matches TCPReceiver:
    receiver = ENetReceiver(drone_id=1, output_queue=q)
    receiver.start()
    frame = q.get(timeout=10.0)  # DroneFrame
    receiver.stop()
"""

import logging
import queue
import struct
import threading
import time
from typing import Optional

import cv2
import numpy as np

try:
    import enet  # pyenet - optional at import time; required at runtime
    _ENET_AVAILABLE = True
except ImportError:
    _ENET_AVAILABLE = False
    enet = None  # type: ignore[assignment]

from .models import CameraCalibration, DroneFrame

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Packet format constants (must match enet_drone_streamer/src/packet_builder.py)
# ---------------------------------------------------------------------------

HEADER_FORMAT: str = "<BIQ"           # uint8 + uint32 + uint64 = 13 bytes
CALIBRATION_FORMAT: str = "<9f9f3f5f"  # 26 x float32 = 104 bytes

HEADER_SIZE: int = struct.calcsize(HEADER_FORMAT)           # 13
CALIBRATION_SIZE: int = struct.calcsize(CALIBRATION_FORMAT)  # 104
FIXED_SIZE: int = HEADER_SIZE + CALIBRATION_SIZE            # 117


# =============================================================================
# ENet Receiver
# =============================================================================


class ENetReceiver:
    """
    Receives DroneFrame packets from an ENet drone streamer over reliable UDP.

    This class manages the ENet connection lifecycle:
    1. Connect to the drone stream server as an ENet client
    2. Receive packets via ENet service loop
    3. Parse binary packets into DroneFrame + CameraCalibration objects
    4. Push decoded frames to output_queue
    5. Handle disconnects and auto-reconnect

    Threading:
        - start() spawns a background daemon thread running _run()
        - stop() signals the thread to exit and waits for cleanup
        - Thread-safe: multiple receivers can run concurrently

    Args:
        drone_id:        Which drone to connect to (1-8). Port = base_port + drone_id - 1.
        output_queue:    Queue to push DroneFrame objects onto.
        host:            ENet server host to connect to.
        base_port:       Base port; drone N uses base_port + N - 1 (default 16000).
        reconnect_delay: Seconds to wait before retrying after a disconnect.
    """

    def __init__(
        self,
        drone_id: int,
        output_queue: queue.Queue,
        host: str = "127.0.0.1",
        base_port: int = 16000,
        reconnect_delay: float = 5.0,
    ) -> None:
        self.drone_id = drone_id
        self.output_queue = output_queue
        self.host = host
        self.port = base_port + (drone_id - 1)
        self.reconnect_delay = reconnect_delay

        # Threading control
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # Connection state
        self._connected = False

        # Statistics
        self.frames_received = 0
        self.bytes_received = 0
        self.errors = 0

    # ------------------------------------------------------------------
    # Public API (matches TCPReceiver)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn background thread and begin receiving."""
        if self._running:
            logger.warning(
                "ENetReceiver for drone %d already running", self.drone_id
            )
            return

        if not _ENET_AVAILABLE:
            raise ImportError(
                "pyenet is not installed. Install with: pip install pyenet"
            )

        self._stop_event.clear()
        self._running = True

        self._thread = threading.Thread(
            target=self._run,
            name=f"ENetReceiver-Drone{self.drone_id}",
            daemon=True,
        )
        self._thread.start()

        logger.info(
            "Started ENet receiver for drone %d at %s:%d",
            self.drone_id,
            self.host,
            self.port,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the receiver to stop and wait for the thread to exit."""
        if not self._running:
            return

        logger.info("Stopping ENet receiver for drone %d", self.drone_id)

        self._stop_event.set()
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        logger.info(
            "ENet receiver for drone %d stopped "
            "(frames=%d, bytes=%d, errors=%d)",
            self.drone_id,
            self.frames_received,
            self.bytes_received,
            self.errors,
        )

    def is_running(self) -> bool:
        """Return True if the background thread is active."""
        return self._running

    def is_connected(self) -> bool:
        """Return True if currently connected to the ENet server."""
        return self._connected

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """
        Main receive loop (runs in background thread).

        Creates an ENet client host, connects to the streamer server, and
        dispatches events.  On disconnect or error, waits reconnect_delay
        then retries.
        """
        logger.info(
            "ENet receiver thread started for drone %d (%s:%d)",
            self.drone_id,
            self.host,
            self.port,
        )

        while not self._stop_event.is_set():
            host = None
            peer = None
            try:
                # Create a client-side ENet host (no bind address = client mode)
                host = enet.Host(
                    None,
                    peerCount=1,
                    channelLimit=1,
                    incomingBandwidth=0,
                    outgoingBandwidth=0,
                )

                logger.info(
                    "ENet receiver: connecting to drone %d at %s:%d",
                    self.drone_id,
                    self.host,
                    self.port,
                )

                peer = host.connect(
                    enet.Address(self.host.encode(), self.port),
                    1,  # channel count
                )

                # Inner event loop - runs while connected
                connected_inner = False
                while not self._stop_event.is_set():
                    event = host.service(100)  # 100 ms timeout
                    if event is None:
                        continue

                    if event.type == enet.EVENT_TYPE_CONNECT:
                        self._connected = True
                        connected_inner = True
                        logger.info(
                            "ENet receiver: connected to drone %d server at %s:%d",
                            self.drone_id,
                            self.host,
                            self.port,
                        )

                    elif event.type == enet.EVENT_TYPE_DISCONNECT:
                        self._connected = False
                        connected_inner = False
                        logger.info(
                            "ENet receiver: disconnected from drone %d server",
                            self.drone_id,
                        )
                        break  # exit inner loop to reconnect

                    elif event.type == enet.EVENT_TYPE_RECEIVE:
                        try:
                            data = bytes(event.packet.data)
                            frame = self._parse_packet(data)
                            self.output_queue.put(frame)
                        except Exception as exc:
                            logger.error(
                                "ENet receiver: drone %d parse error: %s",
                                self.drone_id,
                                exc,
                                exc_info=True,
                            )
                            self.errors += 1

            except Exception as exc:
                self._connected = False
                logger.warning(
                    "ENet receiver: drone %d connection error: %s",
                    self.drone_id,
                    exc,
                )
                self.errors += 1

            finally:
                # Disconnect peer cleanly if still connected
                if peer is not None and self._connected:
                    try:
                        peer.disconnect()
                        if host is not None:
                            host.service(200)
                    except Exception:
                        pass
                self._connected = False

            # Wait before reconnecting (unless stop was requested)
            if not self._stop_event.is_set():
                logger.info(
                    "ENet receiver: drone %d will reconnect in %.1f seconds",
                    self.drone_id,
                    self.reconnect_delay,
                )
                self._stop_event.wait(self.reconnect_delay)

        logger.info(
            "ENet receiver thread exiting for drone %d", self.drone_id
        )

    # ------------------------------------------------------------------
    # Packet parsing
    # ------------------------------------------------------------------

    def _parse_packet(self, data: bytes) -> DroneFrame:
        """
        Parse a binary ENet packet into a DroneFrame.

        Packet layout:
            Bytes   0-12:  Header (<BIQ)  -> drone_id, frame_num, timestamp_ns
            Bytes  13-116: Calibration (<9f9f3f5f) -> K(9), R(9), t(3), dist(5)
            Bytes 117-...: JPEG bytes

        Args:
            data: Raw packet bytes from ENet.

        Returns:
            DroneFrame with decoded image and CameraCalibration.

        Raises:
            ValueError: If the packet is too short or JPEG decoding fails.
        """
        if len(data) < FIXED_SIZE:
            raise ValueError(
                f"Packet too short: {len(data)} bytes, expected >= {FIXED_SIZE}"
            )

        # 1. Unpack header (13 bytes)
        drone_id, frame_num, timestamp_ns = struct.unpack(
            HEADER_FORMAT, data[:HEADER_SIZE]
        )

        # 2. Unpack calibration (104 bytes)
        calib_values = struct.unpack(
            CALIBRATION_FORMAT, data[HEADER_SIZE:FIXED_SIZE]
        )
        # 26 floats: K(9) + R(9) + t(3) + dist(5)
        K = np.array(calib_values[0:9], dtype=np.float32).reshape(3, 3)
        R = np.array(calib_values[9:18], dtype=np.float32).reshape(3, 3)
        t = np.array(calib_values[18:21], dtype=np.float32).reshape(3, 1)
        dist = np.array(calib_values[21:26], dtype=np.float32)

        # 3. Remaining bytes are JPEG payload
        jpeg_bytes = data[FIXED_SIZE:]
        if len(jpeg_bytes) == 0:
            raise ValueError("Packet has no JPEG payload")

        # 4. Decode JPEG to BGR image
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(
                f"cv2.imdecode failed for drone {drone_id} frame {frame_num}"
            )

        # 5. Convert timestamp to seconds
        timestamp_sec = timestamp_ns / 1e9

        # 6. Build output objects
        calib = CameraCalibration(K=K, R=R, t=t, dist=dist)
        frame = DroneFrame(
            drone_id=int(drone_id),
            frame_num=int(frame_num),
            timestamp=timestamp_sec,
            frame=image,
            calibration=calib,
        )

        # 7. Update statistics
        self.frames_received += 1
        self.bytes_received += len(data)

        logger.debug(
            "ENet receiver: drone %d frame %d (%dx%d) ts=%.3f",
            drone_id,
            frame_num,
            image.shape[1],
            image.shape[0],
            timestamp_sec,
        )

        return frame


# =============================================================================
# Module-level helper functions (matching tcp_receiver.py pattern)
# =============================================================================


def create_enet_receivers(
    drone_ids: list,
    output_queue: queue.Queue,
    host: str = "127.0.0.1",
    base_port: int = 16000,
) -> dict:
    """
    Create one ENetReceiver per drone.

    Args:
        drone_ids:    List of drone IDs to create receivers for (e.g. list(range(1, 9))).
        output_queue: Shared queue that all receivers push DroneFrame objects onto.
        host:         ENet server host.
        base_port:    Base port (drone N uses base_port + N - 1).

    Returns:
        Dict mapping drone_id -> ENetReceiver.
    """
    receivers: dict[int, ENetReceiver] = {}
    for drone_id in drone_ids:
        receivers[drone_id] = ENetReceiver(
            drone_id=drone_id,
            output_queue=output_queue,
            host=host,
            base_port=base_port,
        )
    return receivers


def start_all_enet_receivers(receivers: dict) -> None:
    """Start all receivers in the given dict."""
    for drone_id, receiver in receivers.items():
        receiver.start()
    logger.info("Started %d ENet receivers", len(receivers))


def stop_all_enet_receivers(receivers: dict, timeout: float = 5.0) -> None:
    """Stop all receivers in the given dict."""
    for drone_id, receiver in receivers.items():
        receiver.stop(timeout=timeout)
    logger.info("Stopped %d ENet receivers", len(receivers))
