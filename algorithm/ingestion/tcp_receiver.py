"""
TCP Frame Receiver

Connects to mock_drone_streamer and receives DroneFrame packets over TCP.

Architecture:
- One TCPReceiver instance per drone (8 total for MATRIX)
- Each receiver runs in its own thread
- Receives complete packets and decodes them using ProtocolDecoder
- Pushes decoded frames to output queue for downstream processing

"""

import logging
import queue
import socket
import struct
import threading
from typing import Optional

from algorithm.ingestion.protocol_decoder import ProtocolDecoder, HEADER_FORMAT
from algorithm.config import settings


logger = logging.getLogger(__name__)


# =============================================================================
# TCP RECEIVER
# =============================================================================


class TCPReceiver:
    """
    Receives DroneFrame packets from mock_drone_streamer over TCP.

    This class manages the TCP connection lifecycle:
    1. Connect to drone stream
    2. Receive complete packets (handle partial reads)
    3. Decode packets into DroneFrame objects
    4. Push frames to output queue
    5. Handle disconnects and errors

    Threading:
        - start() spawns a background thread that runs the receive loop
        - stop() signals the thread to exit and waits for cleanup
        - Thread-safe: multiple receivers can run concurrently
    """

    def __init__(self, drone_id: int, output_queue: queue.Queue):

        self.drone_id = drone_id
        self.output_queue = output_queue

        # Load from config
        self.host = settings.network.host
        self.port = settings.network.base_port + (drone_id - 1)
        self.reconnect_delay = settings.network.reconnect_delay
        self.recv_timeout = settings.network.recv_timeout

        # Protocol decoder (stateless, reusable)
        self.decoder = ProtocolDecoder()

        # Threading control
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # Connection state
        self._socket: Optional[socket.socket] = None
        self._connected = False

        # Statistics
        self.frames_received = 0
        self.bytes_received = 0
        self.errors = 0

    def start(self) -> None:
        if self._running:
            logger.warning("Receiver for drone %d already running", self.drone_id)
            return

        self._stop_event.clear()
        self._running = True

        self._thread = threading.Thread(
            target=self._run,
            name=f"TCPReceiver-Drone{self.drone_id}",
            daemon=True,  # Thread exits when main program exits
        )
        self._thread.start()

        logger.info(
            "Started TCP receiver for drone %d at %s:%d",
            self.drone_id,
            self.host,
            self.port,
        )

    def stop(self, timeout: float = 5.0) -> None:
        if not self._running:
            return

        logger.info("Stopping TCP receiver for drone %d", self.drone_id)

        self._stop_event.set()
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        self._disconnect()

        logger.info(
            "TCP receiver for drone %d stopped (frames=%d, bytes=%d, errors=%d)",
            self.drone_id,
            self.frames_received,
            self.bytes_received,
            self.errors,
        )

    def is_running(self) -> bool:
        return self._running

    def is_connected(self) -> bool:
        return self._connected

    def _run(self) -> None:
        """
        Main receive loop (runs in background thread).

        Handles connection, reconnection, and frame reception.
        Exits when stop() is called.
        """
        logger.info("Receiver thread started for drone %d", self.drone_id)

        while not self._stop_event.is_set():
            try:
                # Establish connection
                if not self._connected:
                    self._connect()

                # Receive and process one frame
                if self._connected:
                    self._receive_frame()

            except ConnectionError as e:
                logger.warning("Drone %d connection lost: %s", self.drone_id, e)
                self._disconnect()
                self._wait_for_reconnect()

            except Exception as e:
                logger.error(
                    "Drone %d error in receive loop: %s",
                    self.drone_id,
                    e,
                    exc_info=True,
                )
                self.errors += 1
                self._disconnect()
                self._wait_for_reconnect()

        logger.info("Receiver thread exiting for drone %d", self.drone_id)

    def _connect(self) -> None:
        try:
            logger.info(
                "Connecting to drone %d at %s:%d", self.drone_id, self.host, self.port
            )

            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.recv_timeout)
            self._socket.connect((self.host, self.port))

            self._connected = True
            logger.info("Connected to drone %d", self.drone_id)

        except Exception as e:
            self._connected = False
            if self._socket:
                self._socket.close()
                self._socket = None
            raise ConnectionError(f"Failed to connect: {e}")

    def _disconnect(self) -> None:
        if self._socket:
            try:
                self._socket.close()
            except Exception as e:
                logger.debug("Error closing socket for drone %d: %s", self.drone_id, e)
            finally:
                self._socket = None
                self._connected = False

    def _wait_for_reconnect(self) -> None:
        if not self._stop_event.is_set():
            logger.info(
                "Drone %d will reconnect in %.1f seconds",
                self.drone_id,
                self.reconnect_delay,
            )
            self._stop_event.wait(self.reconnect_delay)

    def _receive_frame(self) -> None:
        # Read packet in three phases:
        # 1. Read header to get jpeg_size
        # 2. Read JPEG + extrinsic + intrinsic_header to get text sizes
        # 3. Read intrinsic text
        header_size = struct.calcsize(HEADER_FORMAT)
        header_bytes = self._recv_exact(header_size)

        drone_id, frame_num, jpeg_size, timestamp = struct.unpack(
            HEADER_FORMAT, header_bytes
        )

        # Phase 2: Calculate remaining packet size
        # Packet structure: [header][jpeg][extrinsic][intrinsic_header][intrinsic_text]
        # We know jpeg_size from header
        # We know extrinsic is always 48 bytes
        # We know intrinsic_header is 8 bytes (2 unsigned ints)
        # We need to read intrinsic_header to know text sizes

        extrinsic_size = 48
        intrinsic_header_size = 8

        phase2_size = jpeg_size + extrinsic_size + intrinsic_header_size
        phase2_bytes = self._recv_exact(phase2_size)

        intrinsic_header_start = jpeg_size + extrinsic_size
        intrinsic_header = phase2_bytes[
            intrinsic_header_start : intrinsic_header_start + 8
        ]

        K_text_size, dist_text_size = struct.unpack("I I", intrinsic_header)

        phase3_size = K_text_size + dist_text_size
        phase3_bytes = self._recv_exact(phase3_size)

        packet = header_bytes + phase2_bytes + phase3_bytes

        try:
            drone_frame = self.decoder.decode_packet(packet)

            self.output_queue.put(drone_frame)

            self.frames_received += 1
            self.bytes_received += len(packet)

            logger.debug(
                "Drone %d: received frame %d (%d bytes)",
                self.drone_id,
                frame_num,
                len(packet),
            )

        except Exception as e:
            logger.error(
                "Drone %d: failed to decode frame %d: %s", self.drone_id, frame_num, e
            )
            self.errors += 1
            raise

    def _recv_exact(self, num_bytes: int) -> bytes:
        """
        Receive exactly num_bytes from socket.

        TCP doesn't guarantee recv() returns all requested bytes in one call.
        This method loops until we've received exactly the requested amount.

        Args:
            num_bytes: Number of bytes to receive

        Returns:
            Bytes received (length guaranteed to be num_bytes)

        Raises:
            ConnectionError: If socket closes before receiving all bytes
        """
        chunks = []
        bytes_remaining = num_bytes

        while bytes_remaining > 0:
            try:
                chunk = self._socket.recv(bytes_remaining)

                if not chunk:
                    raise ConnectionError("Connection closed by remote host")

                chunks.append(chunk)
                bytes_remaining -= len(chunk)

            except socket.timeout:
                logger.warning(
                    "Drone %d: recv timeout (wanted %d bytes, got %d)",
                    self.drone_id,
                    num_bytes,
                    num_bytes - bytes_remaining,
                )
                raise ConnectionError("Receive timeout")

        return b"".join(chunks)


def create_receivers(
    drone_ids: list[int], output_queue: queue.Queue
) -> dict[int, TCPReceiver]:
    receivers = {}

    for drone_id in drone_ids:
        receiver = TCPReceiver(drone_id=drone_id, output_queue=output_queue)
        receivers[drone_id] = receiver

    return receivers


def start_all_receivers(receivers: dict[int, TCPReceiver]) -> None:
    for drone_id, receiver in receivers.items():
        receiver.start()
    logger.info("Started %d receivers", len(receivers))


def stop_all_receivers(receivers: dict[int, TCPReceiver], timeout: float = 5.0) -> None:
    for drone_id, receiver in receivers.items():
        receiver.stop(timeout=timeout)
    logger.info("Stopped %d receivers", len(receivers))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    logger.info("Testing TCP Receiver")
    logger.info("=" * 60)
    logger.info("")
    logger.info("IMPORTANT: Start mock_drone_streamer first!")
    logger.info("  cd ../mock_drone_streamer")
    logger.info("  python server.py")
    logger.info("")

    # Connect to mock_drone_streamer (using config settings)
    test_queue = queue.Queue()

    receiver = TCPReceiver(drone_id=1, output_queue=test_queue)

    logger.info("Connecting to %s:%d (from config)", receiver.host, receiver.port)

    try:
        receiver.start()

        # Receive 10 frames
        for i in range(10):
            try:
                frame = test_queue.get(timeout=5.0)
                logger.info(
                    "Received frame %d from drone %d (%dx%d)",
                    frame.frame_num,
                    frame.drone_id,
                    frame.frame_width,
                    frame.frame_height,
                )
            except queue.Empty:
                logger.warning("No frame received within timeout")
                break

    finally:
        receiver.stop()
        logger.info("Test complete")
