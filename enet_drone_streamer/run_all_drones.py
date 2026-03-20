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
import subprocess
import sys
import time
from pathlib import Path

BASE_PORT = 16000
NUM_DRONES = 8


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch all 8 ENet drone streamer subprocesses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MATRIX",
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

    processes: list[subprocess.Popen] = []

    logger.info("Launching %d ENet drone streamers...", NUM_DRONES)

    for drone_id in range(1, NUM_DRONES + 1):
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

        proc = subprocess.Popen(cmd)
        processes.append(proc)
        print(f"Launched drone {drone_id} on port {port}  (PID {proc.pid})")

    logger.info("All %d drones launched. Press Ctrl+C to stop.", NUM_DRONES)

    try:
        # Wait for any process to exit (unexpected)
        while True:
            for i, proc in enumerate(processes):
                ret = proc.poll()
                if ret is not None:
                    drone_id = i + 1
                    logger.warning(
                        "Drone %d process exited with code %d", drone_id, ret
                    )
            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("Received Ctrl+C — shutting down all drone processes...")

    finally:
        for i, proc in enumerate(processes):
            if proc.poll() is None:
                drone_id = i + 1
                logger.info("Terminating drone %d (PID %d)...", drone_id, proc.pid)
                proc.terminate()

        # Wait for all to finish
        for i, proc in enumerate(processes):
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                drone_id = i + 1
                logger.warning(
                    "Drone %d did not terminate cleanly, killing...", drone_id
                )
                proc.kill()

        logger.info("All drone processes stopped.")


if __name__ == "__main__":
    main()
