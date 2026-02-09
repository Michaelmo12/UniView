"""
Frame Synchronizer

Groups DroneFrames from multiple drones into SynchronizedFrameSet objects.

Problem:
- Frames from different drones arrive independently via separate TCP streams
- Frames might not arrive in perfect sync (network jitter, processing delays)
- We need to group frames that were captured at approximately the same time

Solution:
- Buffer incoming frames and group them by frame_num (or timestamp)
- Wait for all drones to contribute a frame (or timeout)
- Output complete SynchronizedFrameSet objects downstream

"""

import logging
import queue
import threading
import time
from collections import defaultdict
from typing import Optional

try:
    from ingestion.models import DroneFrame, SynchronizedFrameSet
    from config import settings
except ImportError:
    from models import DroneFrame, SynchronizedFrameSet
    import sys

    sys.path.insert(0, "..")
    from config import settings


logger = logging.getLogger(__name__)


class FrameSynchronizer:
    """
    Synchronizes DroneFrames across multiple drones into SynchronizedFrameSet objects.

    Strategy:
    - Group frames by frame_num (MATRIX dataset has synchronized frame numbers)
    - Buffer frames until all drones contribute OR timeout expires
    - Output complete or partial sets downstream

    Threading:
        - start() spawns background thread that processes input queue
        - stop() signals thread to exit
    """

    MAX_OUTPUTTED_HISTORY = 1000

    def __init__(self, input_queue: queue.Queue, output_queue: queue.Queue):
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.num_drones = settings.ingestion.num_drones
        self.sync_timeout = settings.ingestion.sync_timeout
        self.max_buffer_size = settings.ingestion.max_buffer_size

        # Frame buffer: {frame_num: {drone_id: DroneFrame}}
        self._buffer: dict[int, dict[int, DroneFrame]] = defaultdict(dict)
        
        self._arrival_times: dict[int, float] = {}

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        self._outputted_frames: set[int] = set()

        self.frames_processed = 0
        self.sets_complete = 0
        self.sets_partial = 0

    def start(self) -> None:
        if self._running:
            logger.warning("Synchronizer already running")
            return

        self._stop_event.clear()
        self._running = True

        # daemon thread automatically terminates when the main program exits
        self._thread = threading.Thread(
            target=self._run, name="FrameSynchronizer", daemon=True
        )
        self._thread.start()

        logger.info(
            "Started frame synchronizer (num_drones=%d, timeout=%.3fs)",
            self.num_drones,
            self.sync_timeout,
        )

    def stop(self, timeout: float = 5.0) -> None:
        if not self._running:
            return

        logger.info("Stopping frame synchronizer")

        self._stop_event.set()
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        self._flush_all()

        total_sets = self.sets_complete + self.sets_partial
        complete_pct = (self.sets_complete / total_sets * 100) if total_sets > 0 else 0

        logger.info("Frame synchronizer stopped:")
        logger.info(
            "  Total sets: %d (complete=%d [%.1f%%], partial=%d [%.1f%%])",
            total_sets,
            self.sets_complete,
            complete_pct,
            self.sets_partial,
            100 - complete_pct,
        )

    def is_running(self) -> bool:
        return self._running

    def _run(self) -> None:
        """
        Main synchronization loop (runs in background thread).

        Processes frames from input queue and checks for timeouts.
        """
        logger.info("Synchronizer thread started")

        while not self._stop_event.is_set():
            self._process_incoming_frames()
            self._check_timeouts()
            self._enforce_buffer_limit()
            # Brief sleep to avoid busy loop
            time.sleep(0.01)

        logger.info("Synchronizer thread exiting")

    def _process_incoming_frames(self) -> None:
        """Process all available frames from input queue."""
        # Drain queue without blocking
        while not self._stop_event.is_set():
            try:
                frame = self.input_queue.get_nowait()
                self._add_frame(frame)
                self.frames_processed += 1

            except queue.Empty:
                break

    def _add_frame(self, frame: DroneFrame) -> None:
        frame_num = frame.frame_num
        drone_id = frame.drone_id

        if frame_num in self._outputted_frames:
            logger.warning(
                "Late frame %d from drone %d (already output) - dropping",
                frame_num,
                drone_id,
            )
            return

        if drone_id in self._buffer[frame_num]:
            logger.warning(
                "Duplicate frame: drone %d, frame %d (ignoring)", drone_id, frame_num
            )
            return

        self._buffer[frame_num][drone_id] = frame

        if frame_num not in self._arrival_times:
            self._arrival_times[frame_num] = time.time()

        num_drones_present = len(self._buffer[frame_num])

        logger.debug(
            "Buffered frame %d from drone %d (now have %d/%d drones)",
            frame_num,
            drone_id,
            num_drones_present,
            self.num_drones,
        )

        if num_drones_present == self.num_drones:
            logger.debug(
                "Complete set %d (all %d drones, outputting immediately)",
                frame_num,
                self.num_drones,
            )
            self._output_set(frame_num, complete=True)
        # Otherwise wait for timeout

    def _check_timeouts(self) -> None:
        current_time = time.time()

        timed_out = []
        for frame_num, arrival_time in self._arrival_times.items():
            if current_time - arrival_time > self.sync_timeout:
                timed_out.append(frame_num)

        # Output timed-out sets
        for frame_num in timed_out:
            num_drones = len(self._buffer[frame_num])
            logger.warning(
                "Timeout for frame %d (%d/%d drones) - outputting partial set",
                frame_num,
                num_drones,
                self.num_drones,
            )
            self._output_set(frame_num, complete=False)

    def _enforce_buffer_limit(self) -> None:
        if len(self._buffer) <= self.max_buffer_size:
            return

        oldest_frame_num = min(
            self._arrival_times.keys(), key=lambda fn: self._arrival_times[fn]
        )

        self._buffer.pop(oldest_frame_num, None)
        self._arrival_times.pop(oldest_frame_num, None)
        self._outputted_frames.add(oldest_frame_num)

        logger.error(
            "Buffer overflow (%d frames) - dropping frame %d (SYSTEM OVERLOAD)",
            self.max_buffer_size,
            oldest_frame_num,
        )

    def _output_set(self, frame_num: int, complete: bool) -> None:
        frames = self._buffer.pop(frame_num, {})
        arrival_time = self._arrival_times.pop(frame_num, None)

        if not frames:
            logger.warning("Attempted to output empty frame set %d", frame_num)
            return

        first_frame = next(iter(frames.values()))
        timestamp = first_frame.timestamp
        
        latency = time.time() - arrival_time if arrival_time else 0.0

        sync_set = SynchronizedFrameSet(
            frame_num=frame_num,
            timestamp=timestamp,
            frames=frames,
            num_drones_expected=self.num_drones,
        )

        if complete:
            self.sets_complete += 1
        else:
            self.sets_partial += 1

        num_drones = len(frames)

        if complete:
            logger.debug(
                "Output complete set %d (all %d drones, latency=%.3fs)",
                frame_num,
                self.num_drones,
                latency,
            )
        else:
            logger.info(
                "Output partial set %d (%d/%d drones, missing: %s, latency=%.3fs)",
                frame_num,
                num_drones,
                self.num_drones,
                sync_set.missing_drones,
                latency,
            )

        self._outputted_frames.add(frame_num)

        if len(self._outputted_frames) > self.MAX_OUTPUTTED_HISTORY:
            min_frame = min(self._outputted_frames)
            self._outputted_frames.discard(min_frame)

        self.output_queue.put(sync_set)

    def _flush_all(self) -> None:
        if not self._buffer:
            return

        logger.info("Flushing %d buffered frame sets", len(self._buffer))

        for frame_num in list(self._buffer.keys()):
            self._output_set(frame_num, complete=False)


# =============================================================================
# DEBUG / TESTING
# =============================================================================

if __name__ == "__main__":

    try:
        from ingestion.tcp_receiver import TCPReceiver
    except ImportError:
        from tcp_receiver import TCPReceiver

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    logger.info("Testing Frame Synchronizer with REAL Streamer")
    logger.info("=" * 60)
    logger.info("")
    logger.info("IMPORTANT: Start mock_drone_streamer first!")
    logger.info("  cd ../mock_drone_streamer")
    logger.info("  python server.py")
    logger.info("")
    logger.info("This test will:")
    logger.info("  - Connect to 3 drones (ports 15000-15002)")
    logger.info("  - Synchronize frames across all 3 drones")
    logger.info("  - Receive 10 complete synchronized sets")
    logger.info("")
    logger.info("=" * 60)

    # Queues
    receiver_queue = queue.Queue()  # TCP receivers → Synchronizer
    output_queue = queue.Queue()  # Synchronizer → Output

    # Create TCP receivers for 3 drones (using config)
    receivers = {}
    for drone_id in [1, 2, 3, 4, 5, 6, 7, 8]:
        receiver = TCPReceiver(drone_id=drone_id, output_queue=receiver_queue)
        receivers[drone_id] = receiver

    # Create synchronizer (using config)
    sync = FrameSynchronizer(input_queue=receiver_queue, output_queue=output_queue)

    logger.info(
        "Using config: num_drones=%d, sync_timeout=%.3fs",
        sync.num_drones,
        sync.sync_timeout,
    )

    try:
        # Start all components
        logger.info("Starting synchronizer...")
        sync.start()

        logger.info("Starting TCP receivers...")
        for drone_id, receiver in receivers.items():
            receiver.start()
            logger.info(
                "  Drone %d connecting to port %d...", drone_id, 15000 + drone_id - 1
            )

        time.sleep(1.0)  # Wait for connections

        logger.info("")
        logger.info("Receiving synchronized frame sets...")
        logger.info("-" * 60)

        # Receive 10 synchronized sets
        for i in range(10):
            try:
                sync_set = output_queue.get(timeout=5.0)

                logger.info(
                    "Set %2d: frame_num=%4d, complete=%s, drones=%s, missing=%s",
                    i + 1,
                    sync_set.frame_num,
                    "✓" if sync_set.is_complete else "✗",
                    sync_set.drone_ids,
                    sync_set.missing_drones if not sync_set.is_complete else "none",
                )

                # Show one frame's details
                if i == 0:
                    first_drone = sync_set.drone_ids[0]
                    frame = sync_set.frames[first_drone]
                    logger.info(
                        "  Sample frame: drone=%d, size=%dx%d, calibration=K[0,0]=%.1f",
                        frame.drone_id,
                        frame.frame_width,
                        frame.frame_height,
                        frame.calibration.K[0, 0],
                    )

            except queue.Empty:
                logger.warning("Timeout waiting for synchronized set")
                break

        logger.info("-" * 60)
        logger.info("")
        logger.info("Test Results:")
        logger.info("  Frames processed: %d", sync.frames_processed)
        logger.info("  Complete sets: %d", sync.sets_complete)
        logger.info("  Partial sets: %d", sync.sets_partial)
        logger.info("")

        # Show receiver stats
        logger.info("Receiver Statistics:")
        for drone_id, receiver in receivers.items():
            logger.info(
                "  Drone %d: frames=%d, bytes=%d, errors=%d",
                drone_id,
                receiver.frames_received,
                receiver.bytes_received,
                receiver.errors,
            )

        logger.info("")
        logger.info("=" * 60)
        logger.info("Test complete!")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")

    finally:
        # Cleanup
        logger.info("\nStopping receivers...")
        for receiver in receivers.values():
            receiver.stop()

        logger.info("Stopping synchronizer...")
        sync.stop()

        logger.info("Cleanup complete")
