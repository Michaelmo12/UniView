"""
Receiver Process
================

Moves the ENet receiver (SyncedENetReceiver) into a **separate OS process**,
completely isolating its CPU usage from OpenVINO's internal OpenMP thread pool.

Why a separate process?
-----------------------
OpenVINO uses OpenMP internally and tries to claim all available CPU cores.
Any background thread in the *same* process (including cv2.imdecode threads)
competes with OpenVINO's thread pool — the OS scheduler gives them equal
priority, so OpenVINO gets preempted and runs slower.

A separate ``multiprocessing.Process`` has its own OS scheduler slot.
OpenVINO in the algorithm process gets its full CPU allocation with no
competition from ENet receiver work.

Components
----------
_receiver_worker(...)
    Module-level function (must be at module level for multiprocessing pickling).
    Runs inside the child process. Creates a SyncedENetReceiver and forwards
    SynchronizedFrameSets to the parent via a multiprocessing.Queue.
    Also writes per-drone connection flags into a shared multiprocessing.Array
    so the parent can check is_connected() without IPC overhead.

ConnectionProxy
    Exposes the same is_connected() / is_running() API as DroneState,
    but reads from the shared multiprocessing.Array instead of the real
    DroneState (which lives in the child process).

ReceiverProcess
    Manages the child process lifecycle: start(), stop(), is_running().
    Owns the multiprocessing.Queue, stop_event, and connected_flags array.

create_receiver_process(drone_ids, output_queue, ...)
    Factory. Returns (ReceiverProcess, dict[int, ConnectionProxy]).
    The dict has the same shape as the old receivers dict so all callers
    (wait_for_connections, stats loops) work without changes.
"""

from __future__ import annotations

import logging
import multiprocessing
import multiprocessing.synchronize
import os
import queue
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker function — runs in the child process
# ---------------------------------------------------------------------------

def _receiver_worker(
    drone_ids: list[int],
    mp_queue: multiprocessing.Queue,
    connected_flags: multiprocessing.Array,
    host: str,
    base_port: int,
    sync_timeout: float,
    max_buffer_size: int,
    stop_event: multiprocessing.synchronize.Event,
) -> None:
    """
    Entry point for the child process.

    Creates a SyncedENetReceiver with a local threading.Queue, then
    forwards SynchronizedFrameSets into mp_queue (multiprocessing.Queue).
    Keeps connected_flags up to date so the parent can check is_connected().
    Exits when stop_event is set.
    """
    # Suppress noisy logging inside the child unless DEBUG is requested
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    # Import here so the import only happens inside the child process
    from src.config.settings import settings
    from src.ingestion.synced_receiver import SyncedENetReceiver

    # Local threading.Queue: SyncedENetReceiver → this forward loop
    local_queue: queue.Queue = queue.Queue(maxsize=4)

    receiver = SyncedENetReceiver(
        drone_ids=drone_ids,
        output_queue=local_queue,
        host=host,
        base_port=base_port,
        sync_timeout=sync_timeout,
        max_buffer_size=max_buffer_size,
    )
    receiver.start()

    drone_index = {d: i for i, d in enumerate(drone_ids)}

    def _sync_flags() -> None:
        """Copy DroneState.connected into the shared flags array."""
        for drone_id, state in receiver.drone_states.items():
            idx = drone_index.get(drone_id)
            if idx is not None:
                connected_flags[idx] = int(state.is_connected())

    try:
        while not stop_event.is_set():
            # Sync connection flags to shared memory periodically
            _sync_flags()

            try:
                sync_set = local_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Forward to parent — drop oldest if full
            try:
                mp_queue.put_nowait(sync_set)
            except Exception:
                try:
                    mp_queue.get_nowait()   # drop oldest
                    mp_queue.put_nowait(sync_set)
                except Exception:
                    pass

    finally:
        _sync_flags()
        receiver.stop(timeout=5.0)


# ---------------------------------------------------------------------------
# ConnectionProxy — parent-side view of a drone's connection state
# ---------------------------------------------------------------------------

class ConnectionProxy:
    """
    Per-drone connection state readable from the parent process.

    The real DroneState lives inside the child process and cannot be shared
    directly. This class reads from a multiprocessing.Array of bytes (one
    byte per drone) that the child process updates on CONNECT/DISCONNECT.

    Exposes the same public API as DroneState / ENetReceiver so that
    wait_for_connections() and any stats loops work without changes.
    """

    def __init__(
        self,
        drone_id: int,
        connected_flags: multiprocessing.Array,
        index: int,
    ) -> None:
        self.drone_id        = drone_id
        self._flags          = connected_flags
        self._index          = index
        # These counters are not shared — parent doesn't need per-drone counts
        self.frames_received = 0
        self.bytes_received  = 0
        self.errors          = 0

    def is_connected(self) -> bool:
        """True while the child process reports the drone as connected."""
        return bool(self._flags[self._index])

    def is_running(self) -> bool:
        """Always True — proxy is alive as long as ReceiverProcess runs."""
        return True

    def start(self) -> None:
        """No-op. ReceiverProcess.start() manages everything."""
        pass

    def stop(self, timeout: float = 5.0) -> None:
        """No-op. ReceiverProcess.stop() manages everything."""
        pass


# ---------------------------------------------------------------------------
# ReceiverProcess — manages the child process lifecycle
# ---------------------------------------------------------------------------

class ReceiverProcess:
    """
    Manages a child process running SyncedENetReceiver.

    The child process is a daemon — it is automatically killed if the parent
    exits unexpectedly.

    Usage::

        mp_queue = multiprocessing.Queue(maxsize=2)
        proc, proxies = create_receiver_process(
            drone_ids=[3, 4, 6, 7],
            output_queue=mp_queue,
        )
        proc.start()
        sync_set = mp_queue.get(timeout=10.0)
        proc.stop()
    """

    def __init__(
        self,
        drone_ids: list[int],
        output_queue: multiprocessing.Queue,
        host: str,
        base_port: int,
        sync_timeout: float,
        max_buffer_size: int,
    ) -> None:
        self._drone_ids      = drone_ids
        self._output_queue   = output_queue
        self._host           = host
        self._base_port      = base_port
        self._sync_timeout   = sync_timeout
        self._max_buffer_size = max_buffer_size

        self._stop_event = multiprocessing.Event()

        # One byte per drone: 0 = disconnected, 1 = connected
        self._connected_flags = multiprocessing.Array('b', len(drone_ids))

        self._process: multiprocessing.Process | None = None

        # ConnectionProxy objects — one per drone, same API as DroneState
        self.drone_states: dict[int, ConnectionProxy] = {
            d: ConnectionProxy(d, self._connected_flags, i)
            for i, d in enumerate(drone_ids)
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the child process."""
        if self._process is not None and self._process.is_alive():
            logger.warning("ReceiverProcess already running")
            return

        self._stop_event.clear()

        self._process = multiprocessing.Process(
            target=_receiver_worker,
            args=(
                self._drone_ids,
                self._output_queue,
                self._connected_flags,
                self._host,
                self._base_port,
                self._sync_timeout,
                self._max_buffer_size,
                self._stop_event,
            ),
            daemon=True,
            name="ENetReceiverProcess",
        )
        self._process.start()
        logger.info(
            "ReceiverProcess started (PID %d) for drones %s",
            self._process.pid,
            self._drone_ids,
        )

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the child process to stop and wait for it to exit."""
        self._stop_event.set()

        if self._process is None:
            return

        if self._process.is_alive():
            self._process.join(timeout=timeout)

        if self._process.is_alive():
            logger.warning(
                "ReceiverProcess (PID %d) did not stop in %.1fs — terminating",
                self._process.pid, timeout,
            )
            self._process.terminate()
            self._process.join(timeout=3.0)

        logger.info("ReceiverProcess stopped")

    def is_running(self) -> bool:
        """True if the child process is alive."""
        return self._process is not None and self._process.is_alive()


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_receiver_process(
    drone_ids: list[int],
    output_queue: multiprocessing.Queue,
    host: str = "127.0.0.1",
    base_port: int = 16000,
) -> tuple[ReceiverProcess, dict[int, ConnectionProxy]]:
    """
    Create a ReceiverProcess configured from settings.

    Returns:
        (proc, drone_proxies_dict) where drone_proxies_dict maps
        drone_id → ConnectionProxy and has the same public API as the old
        receivers dict (is_connected(), etc.).
    """
    from src.config.settings import settings

    proc = ReceiverProcess(
        drone_ids=drone_ids,
        output_queue=output_queue,
        host=host,
        base_port=base_port,
        sync_timeout=settings.ingestion.sync_timeout,
        max_buffer_size=settings.ingestion.max_buffer_size,
    )
    return proc, proc.drone_states
