"""
ingestion/synced_receiver.py
----------------------------
Receives ENet packets from all drones and synchronizes them by frame_num.

Each packet arrives as raw binary bytes: [header 13B][calibration 104B][JPEG].
parse_enet_packet unpacks the bytes into a DroneFrame.
SyncBuffer groups frames by frame_num — once all drones arrive, outputs a SynchronizedFrameSet.
If a drone does not arrive within sync_timeout — outputs a partial set without it.
Everything runs in a single background thread.
"""

# Print messages at severity levels (DEBUG, INFO, WARNING, ERROR)
import logging

# Thread-safe FIFO queue — used to pass SynchronizedFrameSets to the algorithm (two threads writing/reading)
import queue

# Unpack binary bytes into Python values using format strings (e.g. "<BIQ")
import struct

# Create the background daemon thread that runs the ENet receive loop
import threading

# time.time() — used to measure sync_timeout (how long we wait for all drones per frame_num)
import time

# dict that auto-creates an empty value for new keys — used in SyncBuffer to group frames by frame_num
from collections import defaultdict

# dataclass: auto-generates __init__ from fields.
from dataclasses import dataclass

# Optional[X] = X or None — used in return type hints
from typing import Optional

# cv2.imdecode — converts raw JPEG bytes into a numpy BGR image array
import cv2

# Arrays for K, R, t matrices and image data
import numpy as np

# enet is optional — if not installed, code loads fine and only fails when .start() is called
try:
    import enet

    _ENET_AVAILABLE = True
except ImportError:
    _ENET_AVAILABLE = False
    enet = None  # type: ignore[assignment]

# Data models this file produces: one frame from one drone, and a synchronized set from all drones
from src.ingestion.models import CameraCalibration, DroneFrame, SynchronizedFrameSet

# sync_timeout and max_buffer_size config values
from src.config.settings import settings

# Module-level logger — identifies log messages as coming from this file
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Packet format constants (must match enet_drone_streamer/src/packet_builder.py)
# ---------------------------------------------------------------------------
#
_HEADER_FORMAT: str = "<BIQ"  # uint8 + uint32 + uint64 = 13 bytes
_CALIBRATION_FORMAT: str = "<9f9f3f5f"  # 26 × float32 = 104 bytes

_HEADER_SIZE: int = struct.calcsize(_HEADER_FORMAT)  # 13
_CALIBRATION_SIZE: int = struct.calcsize(_CALIBRATION_FORMAT)  # 104
_FIXED_SIZE: int = _HEADER_SIZE + _CALIBRATION_SIZE  # 117


# ---------------------------------------------------------------------------
# Packet parser
# ---------------------------------------------------------------------------


def parse_enet_packet(data: bytes) -> DroneFrame:
    """
    Parse raw ENet packet bytes into a DroneFrame.

    Packet layout:
        Bytes   0-12:  Header (<BIQ)       → drone_id, frame_num, timestamp_ns
        Bytes  13-116: Calibration (<9f…)  → K(9), R(9), t(3), dist(5) as float32
        Bytes 117-...: JPEG payload

    Returns:
        DroneFrame with decoded BGR image and CameraCalibration.

    Raises:
        ValueError: packet too short, empty JPEG, or imdecode failure.
    """
    # Guard: must be at least large enough to contain header + calibration
    if len(data) < _FIXED_SIZE:
        raise ValueError(
            f"Packet too short: {len(data)} bytes, expected >= {_FIXED_SIZE}"
        )

    # 1. Header
    # slice the packet header bytes and unpack into drone_id, frame_num, timestamp_ns
    drone_id, frame_num, timestamp_ns = struct.unpack(
        _HEADER_FORMAT, data[:_HEADER_SIZE]
    )

    # 2. Calibration (26 floats)
    calib_values = struct.unpack(_CALIBRATION_FORMAT, data[_HEADER_SIZE:_FIXED_SIZE])
    K = np.array(calib_values[0:9], dtype=np.float32).reshape(3, 3)
    R = np.array(calib_values[9:18], dtype=np.float32).reshape(3, 3)
    t = np.array(calib_values[18:21], dtype=np.float32).reshape(3, 1)
    dist = np.array(calib_values[21:26], dtype=np.float32)

    # 3. JPEG payload
    jpeg_bytes = data[_FIXED_SIZE:]
    if len(jpeg_bytes) == 0:
        raise ValueError("Packet has no JPEG payload")

    # 4. Decode JPEG → BGR
    # jpeg_bytes is the raw bytes of the JPEG image. so we use buf to treat them as a numpy array of uint8, which is what cv2.imdecode expects. cv2.imdecode then decodes the JPEG bytes into an image array in BGR format.
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cv2.imdecode failed for drone {drone_id} frame {frame_num}")

    calib = CameraCalibration(K=K, R=R, t=t, dist=dist)
    return DroneFrame(
        drone_id=int(drone_id),
        frame_num=int(frame_num),
        timestamp=timestamp_ns / 1e9,  # convert from nanoseconds to seconds
        frame=image,
        calibration=calib,
    )


# ---------------------------------------------------------------------------
# DroneState — per-drone connection state + statistics
# ---------------------------------------------------------------------------


@dataclass
class DroneState:
    """
    Tracks connection state and receive statistics for a single drone.

    One instance exists per drone. Updated by SyncedENetReceiver as
    CONNECT/DISCONNECT/RECEIVE events arrive. Not a thread — all updates
    happen synchronously inside the receiver thread.
    """

    drone_id: int
    host: str
    port: int
    connected: bool = False
    frames_received: int = 0
    bytes_received: int = 0
    errors: int = 0

    def is_connected(self) -> bool:
        """True while the drone is connected."""
        return self.connected


# ---------------------------------------------------------------------------
# SyncBuffer — groups DroneFrames into SynchronizedFrameSets
# ---------------------------------------------------------------------------


class SyncBuffer:
    """
    Groups incoming DroneFrames by frame_num into SynchronizedFrameSets.

    Not a thread — all methods are called synchronously from inside
    SyncedENetReceiver._run().

    Strategy:
    - Buffer frames until all expected drones contribute → output immediately.
    - If not all drones arrive within sync_timeout seconds → output partial set.
    - If buffer grows beyond max_buffer_size → drop oldest frame.
    """

    # Max number of completed frame_nums to remember — late arrivals for these are dropped
    _MAX_OUTPUT_HISTORY = 1000

    def __init__(
        self,
        drone_ids: list[int],
        sync_timeout: float,
        max_buffer_size: int,
    ) -> None:
        # list of expected drone IDs
        self._drone_ids = drone_ids
        # How many drones we expect per frame_num
        self._num_drones = len(drone_ids)
        # How long to wait for all drones before outputting a partial set (seconds)
        self._sync_timeout = sync_timeout
        # Max number of frame_nums buffered at once — oldest dropped if exceeded
        self._max_buffer_size = max_buffer_size

        # Main buffer: frame_num → {drone_id → DroneFrame}
        # defaultdict auto-creates an empty dict when a new frame_num is first seen
        self._buffer: dict[int, dict[int, DroneFrame]] = defaultdict(dict)
        # Tracks when the first drone arrived for each frame_num — used to detect timeout
        self._arrival_times: dict[int, float] = {}
        # Set of frame_nums already output — used to drop late-arriving frames
        self._outputted: set[int] = set()

        # Statistics - Counters for how many complete vs partial sets were output
        self.sets_complete = 0
        self.sets_partial = 0

    def add(self, frame: DroneFrame) -> Optional[SynchronizedFrameSet]:
        """
        Add a DroneFrame to the buffer.

        Returns a complete SynchronizedFrameSet immediately if all expected
        drones have now arrived for this frame_num. Returns None otherwise
        (frame is held in the buffer until timeout or completion).
        """
        frame_num = frame.frame_num
        drone_id = frame.drone_id

        # Drop late arrivals (frame already output)
        if frame_num in self._outputted:
            logger.warning(
                "SyncBuffer: late frame %d from drone %d — already output, dropping",
                frame_num,
                drone_id,
            )
            return None

        # drop duplicates - defensive programming shouldnt happen
        if drone_id in self._buffer[frame_num]:
            logger.warning(
                "SyncBuffer: duplicate frame %d from drone %d — ignoring",
                frame_num,
                drone_id,
            )
            return None

        # buffer the frame
        self._buffer[frame_num][drone_id] = frame

        # Record first arrival time
        if frame_num not in self._arrival_times:
            self._arrival_times[frame_num] = time.time()

        num_present = len(self._buffer[frame_num])
        logger.debug(
            "SyncBuffer: buffered frame %d from drone %d (%d/%d drones)",
            frame_num,
            drone_id,
            num_present,
            self._num_drones,
        )

        # Complete set — output immediately
        if num_present == self._num_drones:
            return self._pop_and_build(frame_num, complete=True)

        return None

    def flush_timed_out(self) -> list[SynchronizedFrameSet]:
        """
        Check all buffered frame_nums for timeout expiry.
        Returns a list of partial SynchronizedFrameSets for timed-out frames.
        Called on every iteration of the ENet service loop.
        """
        # compare it against when each frame_num first arrived
        now = time.time()

        results = []

        timed_out = []
        # t is when the first drone arrived for that frame_num (fn).
        for fn, t in self._arrival_times.items():
            # how many seconds have passed
            # if that exceeds sync_timeout add to timed_out list.
            if now - t > self._sync_timeout:
                timed_out.append(fn)

        # For each timed-out frame_num — build a partial SynchronizedFrameSet with whatever drones arrived, and add it to results.
        for frame_num in timed_out:
            n = len(self._buffer[frame_num])
            logger.warning(
                "SyncBuffer: timeout for frame %d (%d/%d drones) — partial output",
                frame_num,
                n,
                self._num_drones,
            )
            # complete=False tells _pop_and_build this is a partial set.
            sync_set = self._pop_and_build(frame_num, complete=False)
            if sync_set is not None:
                results.append(sync_set)

        return results

    def flush_all(self) -> list[SynchronizedFrameSet]:
        """
        Output all remaining buffered frame sets (called on shutdown).
        Returns a list of partial SynchronizedFrameSets.
        """
        # If buffer is empty — nothing to flush, return early.
        if not self._buffer:
            return []

        logger.info("SyncBuffer: flushing %d remaining frame sets", len(self._buffer))
        results = []
        # put keys in list to avoid "dictionary changed size during iteration" error when we pop inside the loop
        for frame_num in list(self._buffer.keys()):
            sync_set = self._pop_and_build(frame_num, complete=False)
            if sync_set is not None:
                results.append(sync_set)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pop_and_build(
        self, frame_num: int, complete: bool
    ) -> Optional[SynchronizedFrameSet]:
        """Remove frame_num from buffer and build a SynchronizedFrameSet."""
        # .pop(key, default) — removes the key from the dict and returns its value. If key doesn't exist, returns the default.
        frames = self._buffer.pop(frame_num, {})
        arrival_time = self._arrival_times.pop(frame_num, None)

        if not frames:
            return None
        # use iter like a cursor in order to convert the .values() view object to then use next() to get the first frame.
        first_frame = next(iter(frames.values()))
        latency = time.time() - arrival_time if arrival_time else 0.0

        sync_set = SynchronizedFrameSet(
            frame_num=frame_num,
            timestamp=first_frame.timestamp,
            frames=frames,
            expected_drone_ids=self._drone_ids,
        )

        # for monitoring
        if complete:
            self.sets_complete += 1
            logger.debug(
                "SyncBuffer: complete set %d (all %d drones, latency=%.3fs)",
                frame_num,
                self._num_drones,
                latency,
            )
        else:
            self.sets_partial += 1
            logger.info(
                "SyncBuffer: partial set %d (%d/%d drones, missing=%s, latency=%.3fs)",
                frame_num,
                len(frames),
                self._num_drones,
                sync_set.missing_drones,
                latency,
            )

        # Track as outputted to drop any future late arrivals
        self._outputted.add(frame_num)
        if len(self._outputted) > self._MAX_OUTPUT_HISTORY:
            # removes the oldest frame_num
            self._outputted.discard(min(self._outputted))

        # Enforce buffer size limit (drop oldest if over limit)
        self._enforce_limit()

        return sync_set

    def _enforce_limit(self) -> None:
        """Drop the oldest buffered frame set if buffer exceeds max_buffer_size."""
        # If buffer is within limit — nothing to do
        if len(self._buffer) <= self._max_buffer_size:
            return

        # Find the frame_num that arrived first (been waiting the longest)
        # min() with key=self._arrival_times.__getitem__ finds the key with the smallest value in arrival_times
        oldest = min(self._arrival_times, key=self._arrival_times.__getitem__)

        # Remove the oldest frame_num from the buffer — we're dropping it
        self._buffer.pop(oldest, None)

        # Remove its arrival time entry — no longer tracking it
        self._arrival_times.pop(oldest, None)

        # Mark it as outputted so any late packets for this frame_num get dropped
        self._outputted.add(oldest)

        # Log so we know the system is overwhelmed
        logger.error(
            "SyncBuffer: buffer overflow (%d frames) — dropped frame %d",
            self._max_buffer_size,
            oldest,
        )


# ---------------------------------------------------------------------------
# SyncedENetReceiver — single thread: receive + sync
# ---------------------------------------------------------------------------


class SyncedENetReceiver:
    """
    Single background daemon thread that receives frames from all drones and
    synchronizes them into SynchronizedFrameSets.

    Runs in a single background thread to minimize CPU competition with OpenVINO's thread pool - yolo.

    Flow (inside the single thread):
        ENet event → parse_enet_packet() → DroneFrame
                   → SyncBuffer.add()    → SynchronizedFrameSet (if complete)
        Every loop → SyncBuffer.flush_timed_out() → partial sets
        All ready sets → pushed to output_queue

    Usage::

        receiver, states = create_synced_receiver(
            drone_ids=[3, 4, 6, 7],
            output_queue=sync_queue,
        )
        receiver.start()
        sync_set = sync_queue.get(timeout=10.0)
        receiver.stop()
    """

    def __init__(
        self,
        drone_ids: list[int],  # which drones to connect
        output_queue: queue.Queue,  # where to put completed SynchronizedFrameSets
        host: str = "127.0.0.1",  # loopback address
        base_port: int = 16000,  # drone N connects on base_port + N - 1
        sync_timeout: float = 1.0,  # seconds to wait before outputting partial set
        max_buffer_size: int = 10,  # max frame_nums buffered at once
        reconnect_delay: float = 5.0,  # seconds to wait before reconnecting after disconnect
    ) -> None:
        self._output_queue = output_queue
        self._host = host
        self._base_port = base_port
        self._reconnect_delay = reconnect_delay

        # One DroneState per drone — tracks connection status and receive stats
        self._states: dict[int, DroneState] = {}
        for d in drone_ids:
            self._states[d] = DroneState(
                drone_id=d,
                host=host,
                port=base_port + d - 1,
            )

        # creates sync buffer (not a thread) where frames are grouped by frame_num until complete or timeout, then output as SynchronizedFrameSet
        self._sync_buffer = SyncBuffer(
            drone_ids=drone_ids,
            sync_timeout=sync_timeout,
            max_buffer_size=max_buffer_size,
        )

        # Thread control
        # thread-safe flag. When the main thread wants to stop the receiver
        self._stop_event = threading.Event()
        # thread is none untill .start() is called
        self._thread: Optional[threading.Thread] = None
        # init mode
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the background daemon thread."""
        if self._running:
            logger.warning("SyncedENetReceiver already running")
            return
        if not _ENET_AVAILABLE:
            raise ImportError(
                "pyenet is not installed. Install with: pip install pyenet"
            )

        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run,  # the method we want the thread to run
            name="SyncedENetReceiver",  # thread name for logging
            daemon=True,  # Daemon thread will automatically exit when main program exits
        )
        self._thread.start()
        logger.info(
            "SyncedENetReceiver started for drones %s",
            sorted(self._states.keys()),
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the thread to stop and wait for it to exit."""
        # not running
        if not self._running:
            return

        # signal the thread to stop
        logger.info("SyncedENetReceiver stopping...")
        self._stop_event.set()
        self._running = False

        # check if thread exists and is alive, then join with timeout to wait for it to finish
        if self._thread and self._thread.is_alive():
            # main thread pauses here and waits for the background thread to finish max timeout seconds.
            self._thread.join(timeout=timeout)

        total_frames, total_errors = 0, 0
        for s in self._states.values():
            total_frames += s.frames_received
            total_errors += s.errors

        total_sets = self._sync_buffer.sets_complete + self._sync_buffer.sets_partial
        if total_sets > 0:
            complete_pct = self._sync_buffer.sets_complete / total_sets * 100
        else:
            complete_pct = 0
        logger.info(
            "SyncedENetReceiver stopped: frames=%d errors=%d "
            "sets=%d (complete=%d [%.1f%%] partial=%d)",
            total_frames,
            total_errors,
            total_sets,
            self._sync_buffer.sets_complete,
            complete_pct,
            self._sync_buffer.sets_partial,
        )

    # True if the background thread is currently running
    def is_running(self) -> bool:
        return self._running

    # access to the DroneState dict
    @property
    def drone_states(self) -> dict[int, DroneState]:
        return self._states

    # How many SynchronizedFrameSets were output with all drones present
    @property
    def sets_complete(self) -> int:
        return self._sync_buffer.sets_complete

    # How many SynchronizedFrameSets were output with only some drones (timeout)
    @property
    def sets_partial(self) -> int:
        return self._sync_buffer.sets_partial

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """
        Main loop (runs in background thread).

        Creates a single ENet Host with one peer per drone.
        Dispatches received packets through parse_enet_packet → SyncBuffer.
        Pushes ready SynchronizedFrameSets to output_queue.
        Auto-reconnects on disconnect/error.
        """
        logger.info("SyncedENetReceiver thread started")

        while not self._stop_event.is_set():
            enet_host = None
            # maps each ENet peer object to its DroneState. When a packet arrives, we know which drone sent it.
            peer_to_state: dict = {}

            try:
                # One ENet host handles all drones
                enet_host = enet.Host(
                    None,  # no bind address (we're a client connecting out, not a server listening)
                    peerCount=len(
                        self._states
                    ),  # how many peers (drones) we'll connect to
                    channelLimit=1,  # one channel per peer is enough (we only send one type of data)
                    incomingBandwidth=0,  # 0 means unlimited bandwidth
                    outgoingBandwidth=0,
                )

                # Connect one peer per drone
                for state in self._states.values():
                    peer = enet_host.connect(
                        enet.Address(state.host.encode(), state.port), 1
                    )
                    peer_to_state[peer] = state
                    logger.info(
                        "SyncedENetReceiver: connecting drone %d → %s:%d",
                        state.drone_id,
                        state.host,
                        state.port,
                    )

                # Event loop
                while not self._stop_event.is_set():
                    # waits up to 200ms for an ENet event (connect/disconnect/receive). Returns None if no events.
                    event = enet_host.service(200)

                    # Always check for timed-out frames each iteration
                    for sync_set in self._sync_buffer.flush_timed_out():
                        self._push_to_queue(sync_set)

                    if event is None:
                        continue

                    # look up which drone sent this event. event.peer is the ENet peer object that sent the event. We use it as a key to get the corresponding DroneState from peer_to_state.
                    state = peer_to_state.get(event.peer)
                    if state is None:
                        continue

                    self._handle_event(event, state)

                    # Any disconnect event means the host is stale — mark all drones disconnected and break
                    if event.type == enet.EVENT_TYPE_DISCONNECT:
                        for s in self._states.values():
                            if s != state:
                                logger.info("SyncedENetReceiver: drone %d disconnected", s.drone_id)
                            s.connected = False
                        break

            except Exception as exc:
                # mark all drones as disconnected. If the connection crashed, we can't assume any drone is still connected.
                for state in self._states.values():
                    state.connected = False
                logger.warning("SyncedENetReceiver: connection error: %s", exc)

            finally:
                # Flush remaining buffered frames before reconnect
                for sync_set in self._sync_buffer.flush_all():
                    self._push_to_queue(sync_set)

                for state in self._states.values():
                    state.connected = False

                if enet_host is not None:
                    try:
                        enet_host.flush()
                    except Exception:
                        pass

            # Only reconnect if we crashed — not if .stop() was called
            # wait() returns immediately if stop_event is set during the delay (unlike time.sleep)
            if not self._stop_event.is_set():
                logger.info(
                    "SyncedENetReceiver: reconnecting in %.1fs",
                    self._reconnect_delay,
                )
                self._stop_event.wait(self._reconnect_delay)

        logger.info("SyncedENetReceiver thread exiting")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_event(self, event, state: DroneState) -> None:
        """Dispatch a single ENet event to the appropriate handler."""
        # a drone just connected
        if event.type == enet.EVENT_TYPE_CONNECT:
            state.connected = True
            logger.info("SyncedENetReceiver: drone %d connected", state.drone_id)

        # a drone disconnected
        elif event.type == enet.EVENT_TYPE_DISCONNECT:
            state.connected = False
            logger.info("SyncedENetReceiver: drone %d disconnected", state.drone_id)

        # a packet arrived
        elif event.type == enet.EVENT_TYPE_RECEIVE:
            self._handle_receive(event, state)

    def _handle_receive(self, event, state: DroneState) -> None:
        """Parse a received packet and feed it into the sync buffer."""
        try:
            # converts ENet's internal packet data to a Python bytes object
            data = bytes(event.packet.data)
            # unpacks header, calibration, JPEG → returns a DroneFrame
            frame = parse_enet_packet(data)

            # update states
            state.frames_received += 1
            state.bytes_received += len(data)

            # add to buffer. Returns a complete set if all drones arrived, otherwise None.
            sync_set = self._sync_buffer.add(frame)
            if sync_set is not None:
                self._push_to_queue(sync_set)

        # if anything goes wrong parsing
        except Exception as exc:
            state.errors += 1
            logger.error(
                "SyncedENetReceiver: drone %d receive error: %s",
                state.drone_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Queue output
    # ------------------------------------------------------------------

    def _push_to_queue(self, sync_set: SynchronizedFrameSet) -> None:
        """
        Push a SynchronizedFrameSet to the output queue.
        If the queue is full, drop the oldest item and push the new one.
        the priority is keep the newest frame, drop the oldest.
        """
        try:
            # puts the item in the queue without waiting. If the queue is full, raises queue.Full immediately instead of blocking.
            self._output_queue.put_nowait(sync_set)
        except queue.Full:
            try:
                self._output_queue.get_nowait()  # drop oldest
            except queue.Empty:
                pass
            try:
                self._output_queue.put_nowait(
                    sync_set
                )  # try again to put the new item after making space. If it fails again, we just give up and drop the new item too.
            except queue.Full:
                pass


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_synced_receiver(
    drone_ids: list[int],
    output_queue: queue.Queue,
    host: str = "127.0.0.1",
    base_port: int = 16000,
) -> tuple[SyncedENetReceiver, dict[int, DroneState]]:
    """
    Factory function a function whose only job is to create and return an object. Instead of calling SyncedENetReceiver(...) directly with all its parameters, callers use this simpler function.
    Create a SyncedENetReceiver configured from settings.

    Returns:
        (receiver, drone_states_dict) where drone_states_dict maps
        drone_id → DroneState (connection status and receive stats per drone).
    """
    receiver = SyncedENetReceiver(
        drone_ids=drone_ids,
        output_queue=output_queue,
        host=host,
        base_port=base_port,
        sync_timeout=settings.ingestion.sync_timeout,
        max_buffer_size=settings.ingestion.max_buffer_size,
    )
    return receiver, receiver.drone_states
