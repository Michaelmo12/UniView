"""
Frame Synchronization Buffer for Multi-Drone Streams

Synchronizes frames from 8 drones by frame number.
Returns frames from the highest available frame_num, allowing partial sets
(e.g., if only 5 of 8 drones have data, display those 5 frames).
"""

import threading
from collections import defaultdict


class FrameBuffer:
    """
    Thread-safe buffer for synchronizing frames from multiple drones.

    The buffer collects frames from all drones and returns frames from the
    highest available frame number. Allows partial synchronization - if only
    some drones have data for a given frame, those frames are returned rather
    than waiting for all 8 drones.
    """

    def __init__(self, num_drones=8, max_buffer_size=10):
        """
        Initialize frame buffer.

        Args:
            num_drones: Number of drones to synchronize (default: 8)
            max_buffer_size: Maximum frames to buffer per drone (default: 10)
        """
        self.num_drones = num_drones
        self.max_buffer_size = max_buffer_size

        # Buffers: {drone_id: {frame_num: data}} - using drone IDs 1-8
        self.buffers = {i + 1: {} for i in range(num_drones)}

        # Current frame number we're looking for
        self.current_frame = 0

        # Thread lock for synchronization
        self.lock = threading.Lock()

        # Statistics
        self.stats = {
            'frames_received': defaultdict(int),  # Per drone
            'frames_synchronized': 0,
            'frames_dropped': defaultdict(int),  # Per drone (old frames)
        }

    def add_frame(self, drone_id, frame_num, data):
        """
        Add a received frame to the buffer.

        Args:
            drone_id: Drone ID (1-8)
            frame_num: Frame number (0-999)
            data: Frame data dictionary containing:
                - 'frame': numpy array image
                - 'K': camera intrinsic matrix
                - 'R': rotation matrix
                - 't': translation vector
                - 'dist': distortion coefficients
                - 'timestamp': when packet was created

        Returns:
            bool: True if frame was added, False if dropped (too old or invalid)
        """
        with self.lock:
            # Validate drone_id
            if not 1 <= drone_id <= self.num_drones:
                return False

            # Drop frames that are too old (behind current_frame)
            if frame_num < self.current_frame:
                self.stats['frames_dropped'][drone_id] += 1
                return False

            # Drop frames that are too far ahead (buffer overflow)
            if frame_num >= self.current_frame + self.max_buffer_size:
                # This might indicate a problem, but we'll buffer it anyway
                pass

            # Add frame to buffer
            self.buffers[drone_id][frame_num] = data
            self.stats['frames_received'][drone_id] += 1

            # Note: Cleanup happens in get_synchronized_set() after display
            self._cleanup_old_frames(drone_id, self.current_frame)

            return True

    def get_synchronized_set(self):
        """
        Get a synchronized set of frames (partial synchronization allowed).

        Returns frames from the highest frame number where at least one drone
        has data. Allows partial synchronization - if only 5 of 8 drones have
        frame 130, it will return those 5 frames rather than waiting for all 8.

        Returns:
            tuple: (frame_num, frames_dict) or (None, None) if no frames available
                - frame_num: The frame number being returned
                - frames_dict: {drone_id: data} for available drones (1-8 entries)

        Note:
            After returning frames, current_frame advances and old frames are
            cleaned up automatically.
        """
        with self.lock:
            # Count how many drones have each frame number
            frame_counts = defaultdict(int)
            for drone_id in range(1, self.num_drones + 1):
                for frame_num in self.buffers[drone_id]:
                    frame_counts[frame_num] += 1

            # If no frames at all, return None
            if not frame_counts:
                return None, None

            # Find the highest frame number with at least 1 drone
            # (prioritize newer frames, even if fewer drones have them)
            best_frame = max(frame_counts.keys())

            # Collect all available data for this frame
            frames = {}
            for drone_id in range(1, self.num_drones + 1):
                if best_frame in self.buffers[drone_id]:
                    frames[drone_id] = self.buffers[drone_id][best_frame]

            # Advance current frame to best_frame + 1
            self.current_frame = best_frame + 1

            # Clean up frames older than best_frame
            for drone_id in range(1, self.num_drones + 1):
                self._cleanup_old_frames(drone_id, best_frame)

            # Update stats (only count as "synchronized" if all 8 present)
            if len(frames) == self.num_drones:
                self.stats['frames_synchronized'] += 1

            return best_frame, frames

    def _cleanup_old_frames(self, drone_id, current_frame_num):
        """
        Remove frames older than current_frame_num for a specific drone.

        Args:
            drone_id: Drone ID (1-8)
            current_frame_num: Frame number threshold (remove frames < this)
        """
        # Find frames to remove
        to_remove = [f for f in self.buffers[drone_id] if f < current_frame_num]

        # Remove them
        for frame_num in to_remove:
            del self.buffers[drone_id][frame_num]

    def get_stats(self):
        """
        Get buffer statistics.

        Returns:
            dict: Statistics including:
                - frames_received: dict {drone_id: count}
                - frames_synchronized: int (total complete sets)
                - frames_dropped: dict {drone_id: count}
                - buffer_sizes: dict {drone_id: current buffer size}
                - current_frame: int (current frame being sought)
        """
        with self.lock:
            return {
                'frames_received': dict(self.stats['frames_received']),
                'frames_synchronized': self.stats['frames_synchronized'],
                'frames_dropped': dict(self.stats['frames_dropped']),
                'buffer_sizes': {i + 1: len(self.buffers[i + 1]) for i in range(self.num_drones)},
                'current_frame': self.current_frame
            }

    def reset(self):
        """Reset the buffer (clear all frames and stats)"""
        with self.lock:
            self.buffers = {i + 1: {} for i in range(self.num_drones)}
            self.current_frame = 0
            self.stats = {
                'frames_received': defaultdict(int),
                'frames_synchronized': 0,
                'frames_dropped': defaultdict(int),
            }

    def get_buffer_status(self):
        """
        Get current buffer status for debugging.

        Returns:
            str: Human-readable buffer status
        """
        with self.lock:
            lines = []
            lines.append(f"Current frame: {self.current_frame}")
            lines.append(f"Synchronized: {self.stats['frames_synchronized']}")
            lines.append("Per-drone status:")

            for i in range(self.num_drones):
                drone_id = i + 1
                received = self.stats['frames_received'][drone_id]
                dropped = self.stats['frames_dropped'][drone_id]
                buffered = len(self.buffers[drone_id])

                # Get buffered frame numbers
                frame_nums = sorted(self.buffers[drone_id].keys())
                if frame_nums:
                    frames_str = f"{frame_nums[0]}-{frame_nums[-1]}" if len(frame_nums) > 1 else str(frame_nums[0])
                else:
                    frames_str = "empty"

                lines.append(f"  Drone {drone_id}: rx={received:4d}, drop={dropped:3d}, buf={buffered:2d} [{frames_str}]")

            return "\n".join(lines)


def test_frame_buffer():
    """Test frame buffer synchronization"""
    print("Testing FrameBuffer synchronization...")
    print("=" * 60)

    buffer = FrameBuffer(num_drones=8, max_buffer_size=10)

    # Test 1: Add frames in order
    print("\nTest 1: Add frames in order (all drones, frame 0)")
    for drone_id in range(8):
        data = {
            'frame': f"frame_{drone_id}_0",
            'K': f"K_{drone_id}",
            'R': f"R_{drone_id}",
            't': f"t_{drone_id}",
            'dist': f"dist_{drone_id}",
            'timestamp': 0.0
        }
        buffer.add_frame(drone_id, 0, data)

    frame_num, frames = buffer.get_synchronized_set()
    assert frame_num == 0, "Should get frame 0"
    assert len(frames) == 8, "Should have all 8 drones"
    print(f"✅ Got synchronized frame {frame_num} with {len(frames)} drones")

    # Test 2: Out of order frames
    print("\nTest 2: Add frames out of order")
    buffer.add_frame(0, 2, {'timestamp': 0.0})
    buffer.add_frame(1, 2, {'timestamp': 0.0})
    buffer.add_frame(2, 1, {'timestamp': 0.0})

    frame_num, frames = buffer.get_synchronized_set()
    assert frame_num is None, "Should not sync yet (incomplete set)"
    print(f"✅ Correctly returned None (incomplete set)")

    # Complete frame 1
    for drone_id in range(8):
        buffer.add_frame(drone_id, 1, {'timestamp': 0.0})

    frame_num, frames = buffer.get_synchronized_set()
    assert frame_num == 1, "Should get frame 1"
    print(f"✅ Got synchronized frame {frame_num}")

    # Test 3: Drop old frames
    print("\nTest 3: Old frames are dropped")
    result = buffer.add_frame(0, 0, {'timestamp': 0.0})
    assert result is False, "Old frame should be dropped"
    print(f"✅ Old frame correctly dropped")

    # Test 4: Statistics
    print("\nTest 4: Statistics")
    stats = buffer.get_stats()
    print(f"  Frames synchronized: {stats['frames_synchronized']}")
    print(f"  Frames received (drone 0): {stats['frames_received'][0]}")
    print(f"  Frames dropped (drone 0): {stats['frames_dropped'][0]}")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")


if __name__ == "__main__":
    test_frame_buffer()
