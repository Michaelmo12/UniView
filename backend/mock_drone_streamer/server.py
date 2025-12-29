"""
Mock Drone Streamer Service - TCP Streaming Server

Simulates 8 drones streaming frames + calibration data at 2 FPS.
Each drone runs in its own thread, connecting to ports 15000-15007.
"""

# Standard library imports
import socket
import sys
import threading
import time

# Local imports
from config import DatasetConfig, NetworkConfig, ServerConfig
from utils.calibration_loader import load_extrinsic_binary, load_intrinsic_text
from utils.frame_loader import load_frame
from utils.protocol_encoder import pack_packet_raw

# Global flag for graceful shutdown
running = True

# Synchronized start coordination
start_event = threading.Event()
ready_lock = threading.Lock()
threads_ready = 0


def drone_worker(drone_id, port, dataset_path):
    """
    Worker thread for a single drone.

    Streams frames + calibration data at 2 FPS via TCP.
    Works independently - continues streaming even if client disconnects.

    Args:
        drone_id: Drone ID (0-7)
        port: TCP port to connect to
        dataset_path: Path to MATRIX dataset
    """
    global running, threads_ready

    print(f"[DRONE {drone_id}] Starting on port {port}...")

    # Create listening socket (passive server - waits for clients to connect)
    #create ipv4 tcp socket
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(('0.0.0.0', port))
    server_sock.listen(1)
    server_sock.settimeout(0.1)  # Non-blocking accept with 100ms timeout

    # Signal ready
    with ready_lock:
        threads_ready += 1
        print(f"[DRONE {drone_id}] Ready ({threads_ready}/{ServerConfig.NUM_DRONES})")
        if threads_ready == ServerConfig.NUM_DRONES:
            print(f"\n{'='*70}")
            print(f"[SERVER] All {ServerConfig.NUM_DRONES} drones ready! Starting stream in 1 second...")
            print(f"{'='*70}\n")
            time.sleep(ServerConfig.STARTUP_DELAY)
            start_event.set()

    # Wait for synchronized start
    start_event.wait()
    print(f"[DRONE {drone_id}] Streaming")

    # Statistics
    frames_sent = 0
    frames_failed = 0
    total_bytes = 0

    # Client connection
    client_sock = None

    try:
        # Loop through all frames continuously
        for frame_num in range(ServerConfig.TOTAL_FRAMES):
            if not running:
                break

            # Record exact start time for precise 2 FPS timing
            frame_start_time = time.time()

            # Try to accept new client connection (non-blocking)
            if client_sock is None:
                try:
                    client_sock, addr = server_sock.accept()
                    client_sock.setblocking(True)
                    print(f"[DRONE {drone_id}] Client connected")
                except socket.timeout:
                    # No client waiting - that's fine, continue processing
                    pass
                except OSError:
                    pass

            # Load frame
            frame = load_frame(drone_id, frame_num, dataset_path)
            if frame is None:
                print(f"[WARNING] Drone {drone_id}: Failed to load frame {frame_num}")
                frames_failed += 1
                time.sleep(ServerConfig.FRAME_INTERVAL)
                continue

            # Load raw calibration data (for network efficiency)
            dataset_root = DatasetConfig.get_path()

            # Load extrinsic binary data (48 bytes: rvec + tvec)
            try:
                rvec_binary, tvec_binary = load_extrinsic_binary(drone_id, frame_num, dataset_root)
            except FileNotFoundError:
                print(f"[WARNING] Drone {drone_id}: Extrinsic not found for frame {frame_num}")
                frames_failed += 1
                time.sleep(ServerConfig.FRAME_INTERVAL)
                continue
            except Exception as e:
                print(f"[ERROR] Drone {drone_id}: Error loading extrinsic for frame {frame_num}: {e}")
                frames_failed += 1
                time.sleep(ServerConfig.FRAME_INTERVAL)
                continue

            # Load intrinsic text data (K matrix and distortion coefficients as JSON strings)
            try:
                K_text, dist_text = load_intrinsic_text(drone_id, frame_num, dataset_root)
            except FileNotFoundError:
                print(f"[WARNING] Drone {drone_id}: Intrinsic not found")
                frames_failed += 1
                time.sleep(ServerConfig.FRAME_INTERVAL)
                continue
            except Exception as e:
                print(f"[ERROR] Drone {drone_id}: Error loading intrinsic: {e}")
                frames_failed += 1
                time.sleep(ServerConfig.FRAME_INTERVAL)
                continue

            # Pack packet with raw calibration data (always, even if not connected)
            try:
                packet = pack_packet_raw(drone_id, frame_num, frame, rvec_binary, tvec_binary, K_text, dist_text)
            except Exception as e:
                print(f"[ERROR] Drone {drone_id}: Error packing frame {frame_num}: {e}")
                frames_failed += 1
                time.sleep(ServerConfig.FRAME_INTERVAL)
                continue

            # Try to send if client is connected
            if client_sock:
                try:
                    client_sock.sendall(packet)
                    frames_sent += 1
                    total_bytes += len(packet)

                    # Log progress every N frames (configurable)
                    if frame_num % ServerConfig.PROGRESS_LOG_INTERVAL == 0:
                        avg_size = total_bytes / frames_sent / 1024.0
                        print(f"[DRONE {drone_id}] Frame {frame_num:04d}, {len(packet)/1024:.1f} KB (avg: {avg_size:.1f} KB)")

                except (ConnectionError, BrokenPipeError, OSError, socket.error):
                    print(f"[DRONE {drone_id}] Client disconnected")
                    try:
                        client_sock.close()
                    except Exception:
                        pass
                    client_sock = None
                except Exception as e:
                    # Catch any other send errors to keep server running
                    print(f"[WARNING] Drone {drone_id}: Send error ({type(e).__name__}: {e})")
                    try:
                        client_sock.close()
                    except Exception:
                        pass
                    client_sock = None

            # Log progress every N frames when no client connected (configurable)
            if not client_sock and frame_num % ServerConfig.NO_CLIENT_LOG_INTERVAL == 0:
                print(f"[DRONE {drone_id}] Processing frame {frame_num:04d}")

            # Precise timing: maintain EXACTLY configured FPS regardless of connection status
            elapsed = time.time() - frame_start_time
            sleep_time = max(0, ServerConfig.FRAME_INTERVAL - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"[ERROR] Drone {drone_id}: Fatal error: {e}")

    finally:
        # Close client socket if connected
        if client_sock:
            try:
                client_sock.close()
            except Exception:
                pass

        # Close server socket
        try:
            server_sock.close()
        except Exception:
            pass

        # Print final statistics
        total_frames_processed = frames_sent + frames_failed
        success_rate = (frames_sent / total_frames_processed * 100) if total_frames_processed > 0 else 0
        avg_size = total_bytes / frames_sent / 1024.0 if frames_sent > 0 else 0
        print(f"[DRONE {drone_id}] Complete - {frames_sent} sent, {frames_failed} failed ({success_rate:.1f}% success), avg {avg_size:.1f} KB/packet")


def main():
    """
    Main server entry point.

    Starts 8 drone worker threads and handles graceful shutdown.
    """
    global running

    # Force flush to ensure output appears immediately
    sys.stdout.flush()

    print("=" * 70)
    print("MOCK DRONE STREAMER SERVICE")
    print("=" * 70)
    print("Configuration:")
    print(f"  Dataset: {DatasetConfig.DATASET_PATH}")
    print(f"  Drones: {ServerConfig.NUM_DRONES}")
    print(f"  Ports: {NetworkConfig.get_port_range(ServerConfig.NUM_DRONES)}")
    print(f"  Frames: {ServerConfig.TOTAL_FRAMES} per drone")
    print(f"  Stream rate: {ServerConfig.FPS} FPS ({ServerConfig.FRAME_INTERVAL*1000:.0f}ms per frame)")
    print("=" * 70)
    print()

    # Verify dataset exists
    if not DatasetConfig.validate_path():
        print(f"[ERROR] Dataset not found at {DatasetConfig.DATASET_PATH}")
        return

    # Reset synchronization state
    global threads_ready, start_event
    threads_ready = 0
    start_event.clear()

    # Create worker threads (drone IDs 1-8)
    threads = []
    for i in range(ServerConfig.NUM_DRONES):
        drone_id = i + 1  # Drone IDs: 1, 2, 3, 4, 5, 6, 7, 8
        port = NetworkConfig.get_port(drone_id)  # Ports: 15000, 15001, ..., 15007
        thread = threading.Thread(
            target=drone_worker,
            args=(drone_id, port, DatasetConfig.DATASET_PATH),
            name=f"Drone{drone_id}",
            daemon=True
        )
        threads.append(thread)

    # Start all threads
    print(f"Starting {ServerConfig.NUM_DRONES} drone threads...")
    for thread in threads:
        thread.start()

    print("\n[SERVER] STARTED")

    try:
        # Keep main thread alive
        while any(t.is_alive() for t in threads):
            time.sleep(1)

        print("\n[SERVER] All drones finished streaming")

    except KeyboardInterrupt:
        print("\n\n[SERVER] Interrupted by user, shutting down...")
        running = False

        # Wait for threads to finish (with timeout)
        print("Waiting for drones to stop...")
        for thread in threads:
            thread.join(timeout=ServerConfig.THREAD_JOIN_TIMEOUT)

        print("[SERVER] Server stopped")


if __name__ == "__main__":
    main()
