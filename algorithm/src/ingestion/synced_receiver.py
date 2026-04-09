"""
Synced ENet Receiver
====================

Combines frame receiving and frame synchronization into a **single background
thread**, replacing the previous two-thread design (MultiENetReceiver +
FrameSynchronizer).

Architecture
------------
Old design (3 threads total):
    Main thread  →  MultiENetReceiver thread  →  receiver_queue
                 →  FrameSynchronizer thread  →  sync_queue

New design (2 threads total):
    Main thread  →  SyncedENetReceiver thread  →  sync_queue

Components
----------
parse_enet_packet(data)
    Module-level function. Parses raw ENet packet bytes into a DroneFrame.
    No side-effects — pure input/output.

DroneState
    Dataclass holding per-drone connection state and statistics.
    Exposes the same public API as the old ENetReceiver / DroneProxy so that
    callers (wait_for_connections, stats loops) work without changes.

SyncBuffer
    Not a thread. Groups DroneFrames by frame_num into SynchronizedFrameSets.
    Called synchronously inside SyncedENetReceiver._run().
    Handles: immediate output on complete set, timeout-based partial output,
    buffer size enforcement.

SyncedENetReceiver
    Single background daemon thread.
    Connects to all drones on one ENet Host (one peer per drone).
    On each received packet: parse → add to SyncBuffer → push ready sets to output_queue.
    On each loop iteration: flush timed-out sets.
    On stop: flush all remaining buffered sets.

create_synced_receiver(drone_ids, output_queue, ...)
    Factory function. Returns (SyncedENetReceiver, Dict[int, DroneState]).
    The dict has the same shape as the old receivers dict.
"""

from __future__ import annotations

import logging
import queue
import struct
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

try:
    import enet
    _ENET_AVAILABLE = True
except ImportError:
    _ENET_AVAILABLE = False
    enet = None  # type: ignore[assignment]

from src.ingestion.models import CameraCalibration, DroneFrame, SynchronizedFrameSet
from src.config.settings import settings


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Packet format constants (must match enet_drone_streamer/src/packet_builder.py)
# ---------------------------------------------------------------------------

_HEADER_FORMAT: str = "<BIQ"           # uint8 + uint32 + uint64 = 13 bytes
_CALIBRATION_FORMAT: str = "<9f9f3f5f"  # 26 × float32 = 104 bytes

_HEADER_SIZE: int = struct.calcsize(_HEADER_FORMAT)           # 13
_CALIBRATION_SIZE: int = struct.calcsize(_CALIBRATION_FORMAT)  # 104
_FIXED_SIZE: int = _HEADER_SIZE + _CALIBRATION_SIZE            # 117


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
    if len(data) < _FIXED_SIZE:
        raise ValueError(
            f"Packet too short: {len(data)} bytes, expected >= {_FIXED_SIZE}"
        )

    # 1. Header
    drone_id, frame_num, timestamp_ns = struct.unpack(
        _HEADER_FORMAT, data[:_HEADER_SIZE]
    )

    # 2. Calibration (26 floats)
    calib_values = struct.unpack(_CALIBRATION_FORMAT, data[_HEADER_SIZE:_FIXED_SIZE])
    K    = np.array(calib_values[0:9],  dtype=np.float32).reshape(3, 3)
    R    = np.array(calib_values[9:18], dtype=np.float32).reshape(3, 3)
    t    = np.array(calib_values[18:21], dtype=np.float32).reshape(3, 1)
    dist = np.array(calib_values[21:26], dtype=np.float32)

    # 3. JPEG payload
    jpeg_bytes = data[_FIXED_SIZE:]
    if len(jpeg_bytes) == 0:
        raise ValueError("Packet has no JPEG payload")

    # 4. Decode JPEG → BGR
    buf   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"cv2.imdecode failed for drone {drone_id} frame {frame_num}"
        )

    calib = CameraCalibration(K=K, R=R, t=t, dist=dist)
    return DroneFrame(
        drone_id=int(drone_id),
        frame_num=int(frame_num),
        timestamp=timestamp_ns / 1e9,
        frame=image,
        calibration=calib,
    )


# ---------------------------------------------------------------------------
# DroneState — per-drone connection state + statistics
# ---------------------------------------------------------------------------

@dataclass
class DroneState:
    """
    Holds connection state and receive statistics for one drone.

    Exposes the same public API as the old ENetReceiver / DroneProxy so that
    existing callers (wait_for_connections, stats loops) work without changes.

    Not a thread — lifecycle is managed by SyncedENetReceiver.
    """

    drone_id: int
    host: str
    port: int
    connected: bool = False
    frames_received: int = 0
    bytes_received: int = 0
    errors: int = 0

    # ------------------------------------------------------------------
    # Public API matching ENetReceiver / DroneProxy
    # ------------------------------------------------------------------

    def is_connected(self) -> bool:
        """True while connected to the drone's ENet server."""
        return self.connected

    def is_running(self) -> bool:
        """Always True — alive as long as the parent SyncedENetReceiver runs."""
        return True

    def start(self) -> None:
        """No-op. SyncedENetReceiver.start() manages everything."""
        pass

    def stop(self, timeout: float = 5.0) -> None:
        """No-op. SyncedENetReceiver.stop() manages everything."""
        pass


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

    # How many recently outputted frame_nums to remember (to drop late arrivals)
    _MAX_OUTPUT_HISTORY = 1000

    def __init__(
        self,
        drone_ids: list[int],
        sync_timeout: float,
        max_buffer_size: int,
    ) -> None:
        self._drone_ids      = drone_ids
        self._num_drones     = len(drone_ids)
        self._sync_timeout   = sync_timeout
        self._max_buffer_size = max_buffer_size

        # {frame_num: {drone_id: DroneFrame}}
        self._buffer: dict[int, dict[int, DroneFrame]] = defaultdict(dict)
        # {frame_num: first_arrival_time}
        self._arrival_times: dict[int, float] = {}
        # recently outputted frame_nums (to drop late arrivals)
        self._outputted: set[int] = set()

        # Statistics
        self.sets_complete = 0
        self.sets_partial  = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, frame: DroneFrame) -> Optional[SynchronizedFrameSet]:
        """
        Add a DroneFrame to the buffer.

        Returns a complete SynchronizedFrameSet immediately if all expected
        drones have now arrived for this frame_num. Returns None otherwise
        (frame is held in the buffer until timeout or completion).
        """
        frame_num = frame.frame_num
        drone_id  = frame.drone_id

        # Drop late arrivals (frame already output)
        if frame_num in self._outputted:
            logger.warning(
                "SyncBuffer: late frame %d from drone %d — already output, dropping",
                frame_num, drone_id,
            )
            return None

        # Drop duplicates
        if drone_id in self._buffer[frame_num]:
            logger.warning(
                "SyncBuffer: duplicate frame %d from drone %d — ignoring",
                frame_num, drone_id,
            )
            return None

        # Buffer the frame
        self._buffer[frame_num][drone_id] = frame

        # Record first arrival time
        if frame_num not in self._arrival_times:
            self._arrival_times[frame_num] = time.time()

        num_present = len(self._buffer[frame_num])
        logger.debug(
            "SyncBuffer: buffered frame %d from drone %d (%d/%d drones)",
            frame_num, drone_id, num_present, self._num_drones,
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
        now     = time.time()
        results = []

        timed_out = [
            fn for fn, t in self._arrival_times.items()
            if now - t > self._sync_timeout
        ]

        for frame_num in timed_out:
            n = len(self._buffer[frame_num])
            logger.warning(
                "SyncBuffer: timeout for frame %d (%d/%d drones) — partial output",
                frame_num, n, self._num_drones,
            )
            sync_set = self._pop_and_build(frame_num, complete=False)
            if sync_set is not None:
                results.append(sync_set)

        return results

    def flush_all(self) -> list[SynchronizedFrameSet]:
        """
        Output all remaining buffered frame sets (called on shutdown).
        Returns a list of partial SynchronizedFrameSets.
        """
        if not self._buffer:
            return []

        logger.info("SyncBuffer: flushing %d remaining frame sets", len(self._buffer))
        results = []
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
        frames       = self._buffer.pop(frame_num, {})
        arrival_time = self._arrival_times.pop(frame_num, None)

        if not frames:
            return None

        first_frame = next(iter(frames.values()))
        latency     = time.time() - arrival_time if arrival_time else 0.0

        sync_set = SynchronizedFrameSet(
            frame_num=frame_num,
            timestamp=first_frame.timestamp,
            frames=frames,
            expected_drone_ids=self._drone_ids,
        )

        if complete:
            self.sets_complete += 1
            logger.debug(
                "SyncBuffer: complete set %d (all %d drones, latency=%.3fs)",
                frame_num, self._num_drones, latency,
            )
        else:
            self.sets_partial += 1
            logger.info(
                "SyncBuffer: partial set %d (%d/%d drones, missing=%s, latency=%.3fs)",
                frame_num, len(frames), self._num_drones,
                sync_set.missing_drones, latency,
            )

        # Track as outputted to drop any future late arrivals
        self._outputted.add(frame_num)
        if len(self._outputted) > self._MAX_OUTPUT_HISTORY:
            self._outputted.discard(min(self._outputted))

        # Enforce buffer size limit (drop oldest if over limit)
        self._enforce_limit()

        return sync_set

    def _enforce_limit(self) -> None:
        """Drop the oldest buffered frame set if buffer exceeds max_buffer_size."""
        if len(self._buffer) <= self._max_buffer_size:
            return

        oldest = min(self._arrival_times, key=self._arrival_times.__getitem__)
        self._buffer.pop(oldest, None)
        self._arrival_times.pop(oldest, None)
        self._outputted.add(oldest)

        logger.error(
            "SyncBuffer: buffer overflow (%d frames) — dropped frame %d",
            self._max_buffer_size, oldest,
        )


# ---------------------------------------------------------------------------
# SyncedENetReceiver — single thread: receive + sync
# ---------------------------------------------------------------------------

class SyncedENetReceiver:
    """
    Single background daemon thread that receives frames from all drones and
    synchronizes them into SynchronizedFrameSets.

    Replaces MultiENetReceiver + FrameSynchronizer (2 threads → 1 thread),
    reducing background thread competition with OpenVINO's internal thread pool.

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
        drone_ids: list[int],
        output_queue: queue.Queue,
        host: str = "127.0.0.1",
        base_port: int = 16000,
        sync_timeout: float = 1.0,
        max_buffer_size: int = 10,
        reconnect_delay: float = 5.0,
    ) -> None:
        self._output_queue    = output_queue
        self._host            = host
        self._base_port       = base_port
        self._reconnect_delay = reconnect_delay

        # Per-drone state objects (same public API as old receivers dict)
        self._states: dict[int, DroneState] = {
            d: DroneState(
                drone_id=d,
                host=host,
                port=base_port + d - 1,
            )
            for d in drone_ids
        }

        # Sync buffer (not a thread)
        self._sync_buffer = SyncBuffer(
            drone_ids=drone_ids,
            sync_timeout=sync_timeout,
            max_buffer_size=max_buffer_size,
        )

        # Thread control
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
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
            target=self._run,
            name="SyncedENetReceiver",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "SyncedENetReceiver started for drones %s",
            sorted(self._states.keys()),
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the thread to stop and wait for it to exit."""
        if not self._running:
            return

        logger.info("SyncedENetReceiver stopping...")
        self._stop_event.set()
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        total_frames = sum(s.frames_received for s in self._states.values())
        total_errors = sum(s.errors for s in self._states.values())
        total_sets   = self._sync_buffer.sets_complete + self._sync_buffer.sets_partial
        complete_pct = (
            self._sync_buffer.sets_complete / total_sets * 100
            if total_sets > 0 else 0
        )
        logger.info(
            "SyncedENetReceiver stopped: frames=%d errors=%d "
            "sets=%d (complete=%d [%.1f%%] partial=%d)",
            total_frames, total_errors,
            total_sets,
            self._sync_buffer.sets_complete, complete_pct,
            self._sync_buffer.sets_partial,
        )

    def is_running(self) -> bool:
        return self._running

    @property
    def drone_states(self) -> dict[int, DroneState]:
        """Dict mapping drone_id → DroneState. Same shape as old receivers dict."""
        return self._states

    @property
    def sets_complete(self) -> int:
        return self._sync_buffer.sets_complete

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
            enet_host    = None
            peer_to_state: dict = {}

            try:
                # One ENet host handles all drones
                enet_host = enet.Host(
                    None,
                    peerCount=len(self._states),
                    channelLimit=1,
                    incomingBandwidth=0,
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
                        state.drone_id, state.host, state.port,
                    )

                # Event loop
                while not self._stop_event.is_set():
                    event = enet_host.service(200)

                    # Always check for timed-out frames each iteration
                    for sync_set in self._sync_buffer.flush_timed_out():
                        self._push_to_queue(sync_set)

                    if event is None:
                        continue

                    state = peer_to_state.get(event.peer)
                    if state is None:
                        continue

                    self._handle_event(event, state)

            except Exception as exc:
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
        if event.type == enet.EVENT_TYPE_CONNECT:
            state.connected = True
            logger.info(
                "SyncedENetReceiver: drone %d connected", state.drone_id
            )

        elif event.type == enet.EVENT_TYPE_DISCONNECT:
            state.connected = False
            logger.info(
                "SyncedENetReceiver: drone %d disconnected", state.drone_id
            )

        elif event.type == enet.EVENT_TYPE_RECEIVE:
            self._handle_receive(event, state)

    def _handle_receive(self, event, state: DroneState) -> None:
        """Parse a received packet and feed it into the sync buffer."""
        try:
            data  = bytes(event.packet.data)
            frame = parse_enet_packet(data)

            state.frames_received += 1
            state.bytes_received  += len(data)

            # Add to sync buffer — returns a complete set immediately if ready
            sync_set = self._sync_buffer.add(frame)
            if sync_set is not None:
                self._push_to_queue(sync_set)

        except Exception as exc:
            state.errors += 1
            logger.error(
                "SyncedENetReceiver: drone %d receive error: %s",
                state.drone_id, exc,
            )

    # ------------------------------------------------------------------
    # Queue output
    # ------------------------------------------------------------------

    def _push_to_queue(self, sync_set: SynchronizedFrameSet) -> None:
        """
        Push a SynchronizedFrameSet to the output queue.
        If the queue is full, drop the oldest item and push the new one.
        """
        try:
            self._output_queue.put_nowait(sync_set)
        except queue.Full:
            try:
                self._output_queue.get_nowait()   # drop oldest
            except queue.Empty:
                pass
            try:
                self._output_queue.put_nowait(sync_set)
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
    Create a SyncedENetReceiver configured from settings.

    Returns:
        (receiver, drone_states_dict) where drone_states_dict maps
        drone_id → DroneState and has the same public API as the old
        receivers dict (is_connected(), frames_received, etc.).
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
