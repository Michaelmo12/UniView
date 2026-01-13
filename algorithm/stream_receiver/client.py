"""
Stream Receiver Service - Receiver and Display

Receives frames from 8 drone streamers via TCP, synchronizes them, and displays
in a 2×4 grid using OpenCV. Supports partial synchronization - displays frames
from available drones even if not all 8 drones have data for the current frame.
"""

import sys
import io
import socket

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import time
import threading
import cv2
import numpy as np
import struct

from utils.protocol_decoder import unpack_packet_raw, recv_exact, HEADER_FORMAT, HEADER_SIZE, CALIBRATION_SIZE
from utils.frame_synchronizer import FrameBuffer


# Configuration
NUM_DRONES = 8
BASE_PORT = 15000
SOCKET_TIMEOUT = 5.0  # 5 seconds timeout

# Global flag for graceful shutdown
running = True


def receiver_worker(drone_id, port, frame_buffer):
    """
    Worker thread that receives packets from a single drone.

    Args:
        drone_id: Drone ID (1-8)
        port: TCP port to listen on
        frame_buffer: Shared FrameBuffer instance for synchronization
    """
    global running

    print(f"📺 Drone {drone_id}: Connecting to server on port {port}...")

    packets_received = 0
    packets_failed = 0
    last_frame_num = -1

    # Main loop - supports reconnection
    while running:
        client_sock = None

        try:
            # Connect to server
            try:
                client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_sock.settimeout(5.0)  # Connection timeout
                client_sock.connect(('127.0.0.1', port))
                client_sock.settimeout(SOCKET_TIMEOUT)  # Receive timeout
                print(f"📺 Drone {drone_id}: Connected to server")
            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                # Server not available, retry
                print(f"📺 Drone {drone_id}: Connection failed, retrying in 2s...")
                time.sleep(2)
                continue

            # Receive loop
            while running:
                try:
                    # Read header (17 bytes)
                    header = recv_exact(client_sock, HEADER_SIZE)
                    d_id, frame_num, jpeg_size, timestamp = struct.unpack(HEADER_FORMAT, header)

                    # Read JPEG bytes
                    jpeg_bytes = recv_exact(client_sock, jpeg_size)

                    # Read extrinsic binary (48 bytes)
                    extrinsic_bytes = recv_exact(client_sock, 48)

                    # Read intrinsic header (8 bytes: K_text_size, dist_text_size)
                    intrinsic_header = recv_exact(client_sock, 8)
                    K_text_size, dist_text_size = struct.unpack('I I', intrinsic_header)

                    # Read intrinsic text data (variable length)
                    intrinsic_text_bytes = recv_exact(client_sock, K_text_size + dist_text_size)

                    # Reconstruct full packet
                    packet = header + jpeg_bytes + extrinsic_bytes + intrinsic_header + intrinsic_text_bytes
                    packets_received += 1

                    # Unpack packet
                    try:
                        d_id, frame_num, frame, K, R, t, dist, timestamp = unpack_packet_raw(packet)

                        # Verify drone_id matches (sanity check)
                        if d_id != drone_id:
                            print(f"⚠️  Drone {drone_id}: Received packet from wrong drone (ID {d_id})")
                            continue

                        # Create data dictionary
                        data = {
                            'frame': frame,
                            'K': K,
                            'R': R,
                            't': t,
                            'dist': dist,
                            'timestamp': timestamp
                        }

                        # Add to buffer
                        frame_buffer.add_frame(drone_id, frame_num, data)

                        # Log progress (only when frame advances)
                        if frame_num != last_frame_num and frame_num % 100 == 0:
                            print(f"📥 Drone {drone_id}: Received frame {frame_num:04d}")
                            last_frame_num = frame_num

                    except Exception as e:
                        print(f"❌ Drone {drone_id}: Failed to unpack packet: {e}")
                        packets_failed += 1

                except socket.timeout:
                    if running:
                        print(f"⚠️  Drone {drone_id}: Timeout (no data for {SOCKET_TIMEOUT}s)")
                    break

                except ConnectionError as e:
                    if running:
                        print(f"⚠️  Drone {drone_id}: Connection lost - will reconnect")
                    break

                except Exception as e:
                    if running:
                        print(f"❌ Drone {drone_id}: Receiver error: {e}")
                    break

        except Exception as e:
            if running:
                print(f"❌ Drone {drone_id}: Connection error: {e}")
            time.sleep(1)  # Wait before retry

        finally:
            if client_sock:
                try:
                    client_sock.close()
                except Exception:
                    pass
                client_sock = None

    # Cleanup when receiver thread exits
    success_rate = (packets_received / (packets_received + packets_failed) * 100) if (packets_received + packets_failed) > 0 else 0
    print(f"✅ Drone {drone_id}: Receiver stopped - {packets_received} received, {packets_failed} failed ({success_rate:.1f}% success)")


def create_grid_display(frames_dict, frame_num, fps):
    """
    Create a 2×4 grid display of all 8 drone streams.

    Args:
        frames_dict: {drone_id: data} for all 8 drones
        frame_num: Current synchronized frame number
        fps: Current display FPS

    Returns:
        numpy array: Combined grid image (1920×540)
    """
    # Grid layout: 2 rows × 4 columns
    grid_rows = 2
    grid_cols = 4
    cell_width = 480
    cell_height = 270

    # Create empty grid
    grid = np.zeros((grid_rows * cell_height, grid_cols * cell_width, 3), dtype=np.uint8)

    # Fill grid with drone frames (drone IDs 1-8)
    for i in range(NUM_DRONES):
        drone_id = i + 1  # Drone IDs: 1, 2, 3, 4, 5, 6, 7, 8
        row = i // grid_cols  # Grid positions: 0-based
        col = i % grid_cols

        if drone_id in frames_dict:
            data = frames_dict[drone_id]
            frame = data['frame']
            K = data['K']
            R = data['R']
            t = data['t']

            # Resize frame to cell size
            resized = cv2.resize(frame, (cell_width, cell_height))

            # Add overlay text
            # Line 1: Drone ID and frame number
            text1 = f"Drone {drone_id} | Frame {frame_num:04d}"
            cv2.putText(resized, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # Line 2: Intrinsic parameters (fx, fy, cx, cy)
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            text2 = f"Intrinsic: fx={fx:.0f} fy={fy:.0f}"
            cv2.putText(resized, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       0.45, (0, 255, 0), 1, cv2.LINE_AA)

            # Line 3: Principal point
            text3 = f"Principal: cx={cx:.0f} cy={cy:.0f}"
            cv2.putText(resized, text3, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                       0.45, (0, 255, 0), 1, cv2.LINE_AA)

            # Line 4: Extrinsic translation (camera position)
            tx = t[0, 0]
            ty = t[1, 0]
            tz = t[2, 0]
            text4 = f"Position: x={tx:.2f} y={ty:.2f} z={tz:.2f}"
            cv2.putText(resized, text4, (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                       0.45, (255, 255, 0), 1, cv2.LINE_AA)

            # Place in grid
            y_start = row * cell_height
            x_start = col * cell_width
            grid[y_start:y_start+cell_height, x_start:x_start+cell_width] = resized

        else:
            # Missing frame - show black with red text
            y_start = row * cell_height
            x_start = col * cell_width
            cv2.putText(grid, f"Drone {drone_id}", (x_start + 10, y_start + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(grid, "NO DATA", (x_start + 10, y_start + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # Add global info at bottom
    info_text = f"Synchronized Frame: {frame_num:04d} | Display FPS: {fps:.1f}"
    cv2.putText(grid, info_text, (10, grid.shape[0] - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return grid


def display_loop(frame_buffer):
    """
    Main display loop that shows synchronized frames in real-time.

    Args:
        frame_buffer: FrameBuffer instance for getting synchronized frames
    """
    global running

    print("📺 Display: Starting...")

    # Create window
    window_name = "UniView - 8 Drone Swarm"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 540)

    # FPS tracking
    fps_history = []
    last_time = time.time()
    frames_displayed = 0
    no_data_count = 0

    # Display pacing - show frames at consistent 2 FPS
    target_fps = 2.0
    frame_interval = 1.0 / target_fps  # 0.5 seconds
    last_display_time = time.time()

    print("\n" + "=" * 70)
    print("🎥 Display Active")
    print("=" * 70)
    print("Press ESC to quit")
    print()

    # Track last displayed frame to avoid showing black screen
    last_displayed_frame_num = -1
    last_displayed_frames = None

    try:
        while running:
            loop_start = time.time()

            # Get synchronized frame set (only if enough time passed for next frame)
            current_time = time.time()
            time_since_last = current_time - last_display_time

            if time_since_last >= frame_interval:
                # Ready for next frame - try to get it
                frame_num, frames = frame_buffer.get_synchronized_set()

                if frames is not None and frame_num != last_displayed_frame_num:
                    # New frame available - display it

                    # Calculate actual FPS
                    dt = current_time - last_time
                    if dt > 0:
                        fps = 1.0 / dt
                        fps_history.append(fps)
                        if len(fps_history) > 30:
                            fps_history.pop(0)
                    last_time = current_time
                    last_display_time = current_time

                    avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0

                    # Create grid display
                    grid = create_grid_display(frames, frame_num, avg_fps)

                    # Show frame
                    cv2.imshow(window_name, grid)
                    frames_displayed += 1
                    no_data_count = 0

                    # Remember this frame
                    last_displayed_frame_num = frame_num
                    last_displayed_frames = frames

                    # Log every 100 frames
                    if frame_num % 100 == 0:
                        print(f"✅ Displayed synchronized frame {frame_num:04d} (FPS: {avg_fps:.1f})")

                elif last_displayed_frames is not None:
                    # No new frame yet - keep showing last frame to avoid black screen
                    avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
                    grid = create_grid_display(last_displayed_frames, last_displayed_frame_num, avg_fps)
                    cv2.imshow(window_name, grid)

            else:
                # Not ready for next frame yet - keep showing current frame
                if last_displayed_frames is not None:
                    avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
                    grid = create_grid_display(last_displayed_frames, last_displayed_frame_num, avg_fps)
                    cv2.imshow(window_name, grid)
                else:
                    # No frames received yet - show waiting placeholder
                    no_data_count += 1

                    # Show waiting message with buffer status every 10 iterations
                    if no_data_count == 1 or no_data_count % 10 == 0:
                        stats = frame_buffer.get_stats()
                        print(f"⏳ Waiting... (current_frame={stats['current_frame']}, buffer_sizes={stats['buffer_sizes']})")

                    # Create placeholder image
                    placeholder = np.zeros((540, 1920, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Waiting for data from drones...",
                               (650, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

                    # Show buffer status
                    stats = frame_buffer.get_stats()
                    y_pos = 320
                    for i in range(NUM_DRONES):
                        drone_id = i + 1  # Drone IDs: 1-8
                        received = stats['frames_received'].get(drone_id, 0)
                        buffered = stats['buffer_sizes'].get(drone_id, 0)
                        status_text = f"Drone {drone_id}: {received} received, {buffered} buffered"
                        cv2.putText(placeholder, status_text, (650, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.5, (200, 200, 200), 1, cv2.LINE_AA)
                        y_pos += 25

                    cv2.imshow(window_name, placeholder)

                    # Sleep briefly to avoid busy-waiting
                    time.sleep(0.01)

            # Check for quit (ESC key)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("\n⚠️  User requested quit (ESC pressed)")
                running = False
                break

            # Limit display loop rate to avoid excessive CPU usage
            loop_time = time.time() - loop_start
            if loop_time < 0.01:  # Max ~100 FPS display rate
                time.sleep(0.01 - loop_time)

    except KeyboardInterrupt:
        print("\n⚠️  Display interrupted by user")

    finally:
        cv2.destroyAllWindows()
        print(f"\n✅ Display stopped - {frames_displayed} frames displayed")


def main():
    """
    Main client entry point.

    Starts 8 receiver threads and the display loop.
    """
    global running

    print("=" * 70)
    print("📺 STREAM RECEIVER SERVICE - Receiver and Display")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Drones: {NUM_DRONES}")
    print(f"  Ports: {BASE_PORT}-{BASE_PORT + NUM_DRONES - 1}")
    print(f"  Protocol: Custom Binary TCP")
    print(f"  Socket timeout: {SOCKET_TIMEOUT}s")
    print("=" * 70)
    print()
    print("⚠️  NOTE: Start this receiver FIRST, then start the drone streamer")
    print()

    # Create shared frame buffer
    frame_buffer = FrameBuffer(num_drones=NUM_DRONES, max_buffer_size=10)

    # Create receiver threads (drone IDs 1-8)
    threads = []
    for i in range(NUM_DRONES):
        drone_id = i + 1  # Drone IDs: 1, 2, 3, 4, 5, 6, 7, 8
        port = BASE_PORT + i  # Ports: 15000, 15001, ..., 15007
        thread = threading.Thread(
            target=receiver_worker,
            args=(drone_id, port, frame_buffer),
            name=f"Receiver{drone_id}",
            daemon=True
        )
        threads.append(thread)

    # Start all receiver threads
    print(f"Starting {NUM_DRONES} receiver threads...")
    for thread in threads:
        thread.start()
        time.sleep(0.05)

    print(f"✅ All {NUM_DRONES} receivers listening\n")

    try:
        # Run display loop in main thread
        display_loop(frame_buffer)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")

    finally:
        running = False

        # Wait for receiver threads to stop
        print("\nWaiting for receivers to stop...")
        for thread in threads:
            thread.join(timeout=1.0)

        # Print final statistics
        print("\nFinal Statistics:")
        stats = frame_buffer.get_stats()
        print(f"  Frames synchronized: {stats['frames_synchronized']}")
        for i in range(NUM_DRONES):
            drone_id = i + 1  # Drone IDs: 1-8
            received = stats['frames_received'].get(drone_id, 0)
            dropped = stats['frames_dropped'].get(drone_id, 0)
            print(f"  Drone {drone_id}: {received} received, {dropped} dropped")

        print("\n✅ Client stopped")


if __name__ == "__main__":
    main()
