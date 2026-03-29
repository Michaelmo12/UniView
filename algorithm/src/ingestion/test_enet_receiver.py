"""
ENetReceiver Test / Demo Script

Two modes:

1. LIVE mode (default):
   Connects to a running enet_drone_streamer and receives 5 real frames.
   Start the streamer first:
       python enet_drone_streamer/main.py --drone-id 1

   Run:
       python -m algorithm.src.ingestion.test_enet_receiver

2. UNIT TEST mode (--unit-test flag):
   Builds a synthetic packet in memory, parses it with _parse_packet(),
   and asserts the decoded fields match.  Does NOT require a running server.

   Run:
       python algorithm/src/ingestion/test_enet_receiver.py --unit-test
"""

import argparse
import logging
import queue
import struct
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap: load sibling modules by file path to avoid triggering the
# broken algorithm/src/ingestion/__init__.py (which has stale src.ingestion.* imports).
# This approach works whether the script is run directly or via -m.
# ---------------------------------------------------------------------------
import importlib.util as _ilu
import os as _os
import types as _types

_here = _os.path.dirname(_os.path.abspath(__file__))
_repo_root = _os.path.abspath(_os.path.join(_here, "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def _load_module_by_path(fqname: str, filepath: str):
    """Load a Python module from an explicit file path, bypassing package __init__."""
    if fqname in sys.modules:
        return sys.modules[fqname]
    # Ensure parent package entries exist in sys.modules (without running __init__)
    parts = fqname.split(".")
    for i in range(1, len(parts)):
        pkg_name = ".".join(parts[:i])
        if pkg_name not in sys.modules:
            pkg = _types.ModuleType(pkg_name)
            pkg.__path__ = []  # mark as package
            pkg.__package__ = pkg_name
            sys.modules[pkg_name] = pkg
    spec = _ilu.spec_from_file_location(fqname, filepath)
    mod = _ilu.module_from_spec(spec)
    mod.__package__ = ".".join(fqname.split(".")[:-1])
    sys.modules[fqname] = mod
    spec.loader.exec_module(mod)
    return mod


_models_mod = _load_module_by_path(
    "algorithm.src.ingestion.models",
    _os.path.join(_here, "models.py"),
)
_receiver_mod = _load_module_by_path(
    "algorithm.src.ingestion.enet_receiver",
    _os.path.join(_here, "enet_receiver.py"),
)

CameraCalibration = _models_mod.CameraCalibration
DroneFrame = _models_mod.DroneFrame
ENetReceiver = _receiver_mod.ENetReceiver
HEADER_FORMAT = _receiver_mod.HEADER_FORMAT
CALIBRATION_FORMAT = _receiver_mod.CALIBRATION_FORMAT
HEADER_SIZE = _receiver_mod.HEADER_SIZE
FIXED_SIZE = _receiver_mod.FIXED_SIZE


logger = logging.getLogger(__name__)


# =============================================================================
# Offline unit test (no server required)
# =============================================================================

def _build_synthetic_packet(
    drone_id: int = 3,
    frame_num: int = 42,
    timestamp_ns: int = 1_700_000_000_123_456_789,
) -> tuple[bytes, dict]:
    """
    Build a synthetic packet that matches the PacketBuilder format exactly.

    Returns:
        (packet_bytes, expected) where expected is a dict of decoded values.
    """
    # Calibration matrices
    K = np.array(
        [[800.0, 0.0, 320.0],
         [0.0,   800.0, 240.0],
         [0.0,   0.0,   1.0]],
        dtype=np.float32,
    )
    R = np.eye(3, dtype=np.float32)
    t = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    dist = np.array([0.1, 0.2, 0.01, 0.02, 0.05], dtype=np.float32)

    # Build a tiny synthetic JPEG
    img = np.zeros((60, 80, 3), dtype=np.uint8)
    img[:, :] = [100, 150, 200]  # BGR solid color
    success, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert success, "cv2.imencode failed in test"
    jpeg_bytes = buf.tobytes()

    # Pack header
    header = struct.pack(HEADER_FORMAT, drone_id, frame_num, timestamp_ns)

    # Pack calibration (row-major, little-endian)
    calibration = struct.pack(
        CALIBRATION_FORMAT,
        *K.flatten().tolist(),
        *R.flatten().tolist(),
        *t.flatten().tolist(),
        *dist.flatten().tolist(),
    )

    packet = header + calibration + jpeg_bytes

    expected = {
        "drone_id": drone_id,
        "frame_num": frame_num,
        "timestamp_ns": timestamp_ns,
        "timestamp_sec": timestamp_ns / 1e9,
        "K": K,
        "R": R,
        "t": t,
        "dist": dist,
        "jpeg_len": len(jpeg_bytes),
    }
    return packet, expected


def run_unit_test() -> bool:
    """
    Parse a round-trip synthetic packet and assert all fields match.

    Returns True on success, raises AssertionError on failure.
    """
    print("=" * 60)
    print("ENetReceiver Offline Unit Test")
    print("=" * 60)

    # 1. Verify format constants
    assert struct.calcsize(HEADER_FORMAT) == 13, (
        f"HEADER_FORMAT size is {struct.calcsize(HEADER_FORMAT)}, expected 13"
    )
    assert struct.calcsize(CALIBRATION_FORMAT) == 104, (
        f"CALIBRATION_FORMAT size is {struct.calcsize(CALIBRATION_FORMAT)}, expected 104"
    )
    assert FIXED_SIZE == 117, f"FIXED_SIZE is {FIXED_SIZE}, expected 117"
    print("  [PASS] Packet format constants: 13B header + 104B calibration = 117B fixed")

    # 2. Build a synthetic packet
    packet_bytes, expected = _build_synthetic_packet(
        drone_id=3, frame_num=42, timestamp_ns=1_700_000_000_123_456_789
    )
    print(f"  Built synthetic packet: {len(packet_bytes)} bytes total")

    # 3. Create a minimal ENetReceiver and call _parse_packet directly
    #    (no network connection needed for parsing)
    dummy_queue: queue.Queue = queue.Queue()
    receiver = ENetReceiver(drone_id=3, output_queue=dummy_queue)

    frame = receiver._parse_packet(packet_bytes)

    # 4. Assert fields match
    assert isinstance(frame, DroneFrame), f"Expected DroneFrame, got {type(frame)}"
    assert frame.drone_id == expected["drone_id"], (
        f"drone_id: got {frame.drone_id}, expected {expected['drone_id']}"
    )
    assert frame.frame_num == expected["frame_num"], (
        f"frame_num: got {frame.frame_num}, expected {expected['frame_num']}"
    )
    assert abs(frame.timestamp - expected["timestamp_sec"]) < 1e-6, (
        f"timestamp: got {frame.timestamp}, expected {expected['timestamp_sec']}"
    )
    print("  [PASS] Header fields: drone_id, frame_num, timestamp")

    assert frame.calibration.K.shape == (3, 3), "K shape mismatch"
    assert frame.calibration.R.shape == (3, 3), "R shape mismatch"
    assert frame.calibration.t.shape == (3, 1), "t shape mismatch"
    assert frame.calibration.dist.shape == (5,), "dist shape mismatch"
    np.testing.assert_allclose(frame.calibration.K, expected["K"], rtol=1e-5)
    np.testing.assert_allclose(frame.calibration.R, expected["R"], rtol=1e-5)
    np.testing.assert_allclose(frame.calibration.t, expected["t"], rtol=1e-5)
    np.testing.assert_allclose(frame.calibration.dist, expected["dist"], rtol=1e-5)
    print("  [PASS] Calibration matrices: K, R, t, dist (shapes and values)")

    assert frame.frame is not None, "Decoded frame is None"
    assert len(frame.frame.shape) == 3, "frame must be 3D (H,W,C)"
    assert frame.frame.shape[2] == 3, "frame must have 3 channels"
    print(f"  [PASS] JPEG decode: frame shape {frame.frame.shape}")

    assert receiver.frames_received == 1, (
        f"frames_received: got {receiver.frames_received}, expected 1"
    )
    assert receiver.bytes_received == len(packet_bytes), (
        f"bytes_received: got {receiver.bytes_received}, expected {len(packet_bytes)}"
    )
    print(f"  [PASS] Statistics: frames_received=1, bytes_received={len(packet_bytes)}")

    # 5. Camera center sanity check (C = -R.T @ t)
    C = frame.calibration.camera_center
    assert C.shape == (3,), f"camera_center shape: {C.shape}"
    expected_C = -(expected["R"].T @ expected["t"]).flatten()
    np.testing.assert_allclose(C, expected_C, rtol=1e-5)
    print(f"  [PASS] Camera center: {C}")

    print()
    print("ALL UNIT TESTS PASSED")
    return True


# =============================================================================
# Live demo (requires running enet_drone_streamer)
# =============================================================================

def _overlay_info(image: np.ndarray, frame: "DroneFrame", frame_idx: int, total: int) -> np.ndarray:
    """Draw metadata overlay on a copy of the frame."""
    vis = image.copy()
    C = frame.calibration.camera_center
    K = frame.calibration.K

    R = frame.calibration.R
    t = frame.calibration.t.flatten()
    dist = frame.calibration.dist

    lines = [
        f"Drone {frame.drone_id}  |  Frame {frame.frame_num}  ({frame_idx}/{total})  |  {frame.frame_width}x{frame.frame_height}  |  ts={frame.timestamp:.3f}s",
        f"K:  fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}",
        f"R:  [{R[0,0]:.4f}  {R[0,1]:.4f}  {R[0,2]:.4f}]",
        f"    [{R[1,0]:.4f}  {R[1,1]:.4f}  {R[1,2]:.4f}]",
        f"    [{R[2,0]:.4f}  {R[2,1]:.4f}  {R[2,2]:.4f}]",
        f"t:  [{t[0]:.4f}  {t[1]:.4f}  {t[2]:.4f}]",
        f"dist: k1={dist[0]:.6f}  k2={dist[1]:.6f}  p1={dist[2]:.6f}  p2={dist[3]:.6f}  k3={dist[4]:.6f}",
        f"Camera center: [{C[0]:.4f}  {C[1]:.4f}  {C[2]:.4f}]",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    pad = 10
    line_h = 28

    # Semi-transparent dark background behind text
    overlay_h = pad * 2 + line_h * len(lines)
    roi = vis[0:overlay_h, 0:vis.shape[1]]
    dark = np.zeros_like(roi)
    cv2.addWeighted(dark, 0.55, roi, 0.45, 0, roi)
    vis[0:overlay_h, 0:vis.shape[1]] = roi

    for idx, line in enumerate(lines):
        y = pad + idx * line_h + 24
        # Shadow
        cv2.putText(vis, line, (pad + 1, y + 1), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        # Text
        cv2.putText(vis, line, (pad, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return vis


def run_live_demo(drone_id: int = 1, num_frames: int = 5) -> None:
    """
    Connect to a running enet_drone_streamer and receive num_frames frames.

    Displays each frame in a window with metadata overlay.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print()
    print("=" * 60)
    print("ENetReceiver Live Demo")
    print("=" * 60)
    print()
    print("IMPORTANT: Start enet_drone_streamer first!")
    print(f"  python enet_drone_streamer/main.py --drone-id {drone_id}")
    print()

    frames_q: queue.Queue = queue.Queue()
    receiver = ENetReceiver(drone_id=drone_id, output_queue=frames_q)

    window = f"ENet Receiver — Drone {drone_id}"

    try:
        receiver.start()
        logger.info(
            "Waiting for frames from drone %d (expecting %d frames)...",
            drone_id,
            num_frames,
        )

        for i in range(num_frames):
            try:
                frame: DroneFrame = frames_q.get(timeout=10.0)
                C = frame.calibration.camera_center
                logger.info(
                    "Frame %d/%d: drone_id=%d frame_num=%d  size=%dx%d  "
                    "K[0,0]=%.1f  camera_center=[%.2f, %.2f, %.2f]",
                    i + 1,
                    num_frames,
                    frame.drone_id,
                    frame.frame_num,
                    frame.frame_width,
                    frame.frame_height,
                    frame.calibration.K[0, 0],
                    C[0],
                    C[1],
                    C[2],
                )

                # Show frame with overlay
                vis = _overlay_info(frame.frame, frame, i + 1, num_frames)
                # Scale down for display (1920x1080 is large)
                display = cv2.resize(vis, (960, 540))
                cv2.imshow(window, display)
                cv2.waitKey(1)

            except queue.Empty:
                logger.warning("No frame received within 10s timeout (frame %d/%d)", i + 1, num_frames)
                break

        # Hold last frame until keypress
        print("\nPress any key in the image window to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        cv2.destroyAllWindows()

    finally:
        receiver.stop()
        print()
        print("Stats:")
        print(f"  Frames received: {receiver.frames_received}")
        print(f"  Bytes received:  {receiver.bytes_received}")
        print(f"  Errors:          {receiver.errors}")
        print()
        print("Demo complete.")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ENetReceiver test/demo script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--unit-test",
        action="store_true",
        help="Run offline packet round-trip unit test (no server needed).",
    )
    parser.add_argument(
        "--drone-id",
        type=int,
        default=1,
        help="Drone ID to connect to in live demo mode.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Number of frames to receive in live demo mode.",
    )
    args = parser.parse_args()

    if args.unit_test:
        success = run_unit_test()
        sys.exit(0 if success else 1)
    else:
        run_live_demo(drone_id=args.drone_id, num_frames=args.frames)


if __name__ == "__main__":
    main()
