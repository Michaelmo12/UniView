"""
run_all_drones.py — Launch all 8 ENet drone streamer processes

Spawns one subprocess per drone (Drone 1-8), each running main.py with
the appropriate --drone-id.  On Ctrl+C, terminates all child processes
gracefully (SIGTERM + wait).

Usage:
    python run_all_drones.py --dataset /path/to/MATRIX
    python run_all_drones.py --dataset /path/to/MATRIX --fps 5 --host 0.0.0.0
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_PORT = 16000
NUM_DRONES = 8

# Default dataset path relative to project root
DEFAULT_DATASET = str(Path(__file__).parent.parent / "MATRIX_30x30" / "MATRIX_30x30")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch all 8 ENet drone streamer subprocesses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        metavar="PATH",
        help="Root path to MATRIX dataset directory.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        metavar="FPS",
        help="Target frame send rate for all drones.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        metavar="ADDR",
        help="ENet bind address for all drones.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        metavar="Q",
        help="JPEG compression quality (0-100).",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Stop each drone when its dataset frames are exhausted.",
    )
    parser.add_argument(
        "--drones",
        type=str,
        default="",
        metavar="IDS",
        help="Comma-separated drone IDs to launch, e.g. 3,4,6,7 (default: all 1-8).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    args = _parse_args()

    # Resolve path to main.py relative to this script
    script_dir = Path(__file__).parent.resolve()
    main_py = script_dir / "main.py"

    drone_ids = (
        sorted(int(x) for x in args.drones.split(",") if x.strip())
        if args.drones
        else list(range(1, NUM_DRONES + 1))
    )

    processes: list[subprocess.Popen] = []

    logger.info("Launching %d ENet drone streamers...", len(drone_ids))

    for drone_id in drone_ids:
        port = BASE_PORT + (drone_id - 1)

        cmd = [
            sys.executable,
            str(main_py),
            "--drone-id", str(drone_id),
            "--dataset", args.dataset,
            "--host", args.host,
            "--fps", str(args.fps),
            "--jpeg-quality", str(args.jpeg_quality),
        ]
        if args.no_loop:
            cmd.append("--no-loop")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(script_dir.parent)
        proc = subprocess.Popen(cmd, env=env)
        processes.append((drone_id, proc))
        print(f"Launched drone {drone_id} on port {port}  (PID {proc.pid})")

    # Synchronized start: wait 1 second after all processes are spawned so all
    # 8 drones initialize at roughly the same time before any begin streaming.
    # Mirrors the threading.Event barrier in mock_drone_streamer/server.py.
    print(f"\n{'='*60}")
    print(f"All {len(drone_ids)} drones launched. Waiting 1s for synchronized start...")
    print(f"{'='*60}\n")
    time.sleep(1.0)
    logger.info("All %d drones streaming. Press Ctrl+C to stop.", len(drone_ids))

    try:
        # Wait for any process to exit (unexpected)
        while True:
            for drone_id, proc in processes:
                ret = proc.poll()
                if ret is not None:
                    logger.warning(
                        "Drone %d process exited with code %d", drone_id, ret
                    )
            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("Received Ctrl+C — shutting down all drone processes...")

    finally:
        for drone_id, proc in processes:
            if proc.poll() is None:
                logger.info("Terminating drone %d (PID %d)...", drone_id, proc.pid)
                proc.terminate()

        # Wait for all to finish
        for drone_id, proc in processes:
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Drone %d did not terminate cleanly, killing...", drone_id
                )
                proc.kill()

        logger.info("All drone processes stopped.")


if __name__ == "__main__":
    main()
