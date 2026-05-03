"""
Receiver Process
================

Moves the ENet receiver (SyncedENetReceiver) into a **separate OS process**,
completely isolating its CPU usage from OpenVINO's internal OpenMP thread pool.

Why a separate process?
-----------------------
OpenVINO uses OpenMP internally and tries to claim all available CPU cores.
Any background thread in the *same* process
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

ReceiverProcess
    Manages the child process lifecycle: start(), stop(), is_running().
    Owns the multiprocessing.Queue and stop_event.

create_receiver_process(drone_ids, output_queue, ...)
    Factory. Returns a ReceiverProcess configured from settings.
"""

# Standard logging library — lets us print messages at severity levels (DEBUG, INFO, WARNING, ERROR)
import logging

# Standard library for creating and managing separate OS processes and cross-process queues
import multiprocessing

# Imported explicitly so multiprocessing.synchronize.Event can be used as a type hint
import multiprocessing.synchronize

# Standard thread-safe FIFO queue — used for local_queue inside the child process (thread-to-thread)
import queue


# Module-level logger — identifies log messages as coming from this file
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker function — runs in the child process
# ---------------------------------------------------------------------------


def _receiver_worker(
    drone_ids: list[int],  # Which drones to listen to (e.g. [3,4,6,7])
    mp_queue: multiprocessing.Queue,  # Cross-process pipe — child puts frames here, parent reads them
    host: str,  # IP address to bind ENet server on
    base_port: int,  # Port base (drone N listens on base_port + N - 1)
    sync_timeout: float,  # How long to wait for all drones to have a frame before giving up on sync
    max_buffer_size: int,  # Max frames to buffer per drone before dropping old ones
    stop_event: multiprocessing.synchronize.Event,  # Flag the parent sets to tell the child to shut down
) -> None:
    """
    Entry point for the child process.

    Creates a SyncedENetReceiver with a local threading.Queue, then
    forwards SynchronizedFrameSets into mp_queue (multiprocessing.Queue).
    Exits when stop_event is set.
    """
    # Configure logging for this child process — fresh process has no logging setup
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    # Raise child process priority so ENet packets are not dropped under CPU load
    try:
        import os
        import psutil

        # On Windows, tells the OS scheduler to give this process higher CPU priority
        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        print(f"[priority] receiver child process PID {p.pid} set to {p.nice()}")
    except Exception as e:
        # Priority is best-effort — child still runs if psutil is missing or OS rejects it
        print(f"[priority] failed to set priority: {e}")

    # Import inside child process — on Windows, spawn creates a fresh interpreter so imports must happen here
    from src.ingestion.synced_receiver import SyncedENetReceiver

    # Thread-safe queue local to the child — SyncedENetReceiver writes here, forward loop reads from here
    local_queue = queue.Queue(maxsize=4)

    # Create the ENet receiver with the local queue — it will put SynchronizedFrameSets into local_queue
    receiver = SyncedENetReceiver(
        drone_ids=drone_ids,
        output_queue=local_queue,
        host=host,
        base_port=base_port,
        sync_timeout=sync_timeout,
        max_buffer_size=max_buffer_size,
    )
    # Start the background thread that runs the ENet receive loop
    receiver.start()

    # Run until the parent sets stop_event
    try:
        while not stop_event.is_set():
            try:
                # Block up to 0.5s waiting for a SynchronizedFrameSet from SyncedENetReceiver
                sync_set = local_queue.get(timeout=0.5)
            except queue.Empty:
                # Nothing arrived in 0.5s — loop back and check stop_event again
                continue

            # Try to forward the frame set to the parent's cross-process queue
            try:
                # Non-blocking put — raises Full if mp_queue has no space
                mp_queue.put_nowait(sync_set)
            except Exception:
                try:
                    # mp_queue is full — remove the oldest frame set to make room
                    mp_queue.get_nowait()
                    # Now put the new frame set in
                    mp_queue.put_nowait(sync_set)
                except Exception:
                    # Both attempts failed — discard this frame set silently
                    pass

    finally:
        # Stop the ENet receiver thread cleanly, wait up to 5 seconds
        receiver.stop(timeout=5.0)


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
        proc = create_receiver_process(
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
        # Store all parameters — passed to _receiver_worker when child process spawns
        self._drone_ids = drone_ids
        self._output_queue = output_queue
        self._host = host
        self._base_port = base_port
        self._sync_timeout = sync_timeout
        self._max_buffer_size = max_buffer_size

        # Cross-process shutdown signal — parent calls .set(), child checks .is_set() in its loop
        self._stop_event = multiprocessing.Event()

        # Holds the child process object after start() is called — None until then
        self._process: multiprocessing.Process | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the child process."""
        if self._process is not None and self._process.is_alive():
            logger.warning("ReceiverProcess already running")
            return

        # sets flag to False
        self._stop_event.clear()

        self._process = multiprocessing.Process(
            target=_receiver_worker,
            args=(
                self._drone_ids,
                self._output_queue,
                self._host,
                self._base_port,
                self._sync_timeout,
                self._max_buffer_size,
                self._stop_event,
            ),
            daemon=True,
            name="ENetReceiverProcess",
        )
        # spawns the OS process
        self._process.start()
        logger.info(
            "ReceiverProcess started (PID %d) for drones %s",
            self._process.pid,
            self._drone_ids,
        )

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the child process to stop and wait for it to exit."""
        # sets flag to True
        self._stop_event.set()

        # if stop() is called before start(), nothing to do
        if self._process is None:
            return

        if self._process.is_alive():
            # blocks the parent for up to 10 seconds, waiting for the child to exit cleanly on its own.
            self._process.join(timeout=timeout)

        # If after 10 seconds the child is still alive — it didn't stop cleanly. terminate() sends force kill.
        if self._process.is_alive():
            logger.warning(
                "ReceiverProcess (PID %d) did not stop in %.1fs — terminating",
                self._process.pid,
                timeout,
            )
            self._process.terminate()
            # waits up to 3 more seconds for it to die
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
) -> ReceiverProcess:
    """
    Create a ReceiverProcess configured from settings.

    Returns:
        ReceiverProcess ready to start.
    """
    from src.config.settings import settings

    return ReceiverProcess(
        drone_ids=drone_ids,
        output_queue=output_queue,
        host=host,
        base_port=base_port,
        sync_timeout=settings.ingestion.sync_timeout,
        max_buffer_size=settings.ingestion.max_buffer_size,
    )
