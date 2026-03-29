"""
Shared helpers for live pipeline stage scripts.

All scripts in live_pipeline/ use these utilities for:
- ENet receiver startup/shutdown
- Connection waiting
- Per-stage timing bar printing
- Consistent argument parsing
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

# Must be set before OpenVINO / OpenMP loads.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENVINO_CPU_THREADS_NUM", "4")

# ---------------------------------------------------------------------------
# Path bootstrap (scripts may import this before src.* is on path)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_ALGO_ROOT  = _SCRIPT_DIR.parents[2]   # algorithm/
if str(_ALGO_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALGO_ROOT))

import cv2
cv2.setNumThreads(1)

from src.config.settings import settings
from src.ingestion.enet_receiver import (
    ENetReceiver,
    create_enet_receivers,
    start_all_enet_receivers,
    stop_all_enet_receivers,
)
from src.ingestion.synchronizer import FrameSynchronizer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def base_parser(description: str) -> argparse.ArgumentParser:
    """Return an ArgumentParser with the common flags all scripts share."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--drones", type=str, default="3,4,6,7",
                        help="Comma-separated drone IDs to connect to.")
    parser.add_argument("--frames", type=int, default=20,
                        help="Number of synchronized frame sets to process.")
    parser.add_argument("--timeout", type=float, default=15.0,
                        help="Max seconds to wait for each frame set.")
    return parser


def parse_drone_ids(drones_str: str) -> list[int]:
    return sorted({int(x) for x in drones_str.split(",") if x.strip()})


# ---------------------------------------------------------------------------
# Pipeline lifecycle
# ---------------------------------------------------------------------------

def start_pipeline(
    drone_ids: list[int],
    base_port: int = 16000,
    host: str = "127.0.0.1",
) -> Tuple[Dict[int, ENetReceiver], FrameSynchronizer, queue.Queue]:
    """
    Create and start ENet receivers + FrameSynchronizer.

    Returns (receivers_dict, synchronizer, sync_queue).
    sync_queue produces SynchronizedFrameSet objects.
    """
    # Override settings so synchronizer picks up the right drone list
    settings.ingestion.drone_ids = drone_ids

    receiver_queue: queue.Queue = queue.Queue()
    sync_queue:     queue.Queue = queue.Queue()

    receivers = create_enet_receivers(
        drone_ids,
        output_queue=receiver_queue,
        host=host,
        base_port=base_port,
    )
    synchronizer = FrameSynchronizer(
        input_queue=receiver_queue,
        output_queue=sync_queue,
    )

    synchronizer.start()
    start_all_enet_receivers(receivers)
    return receivers, synchronizer, sync_queue


def stop_pipeline(
    receivers: Dict[int, ENetReceiver],
    synchronizer: FrameSynchronizer,
    timeout: float = 5.0,
) -> None:
    """Stop all receivers and synchronizer cleanly."""
    stop_all_enet_receivers(receivers, timeout=timeout)
    synchronizer.stop(timeout=timeout)


def wait_for_connections(
    receivers: Dict[int, ENetReceiver],
    n_expected: int,
    timeout: float = 10.0,
    label: str = "ENet connections",
) -> int:
    """
    Busy-wait until all (or any) drones connect, up to timeout seconds.
    Prints a progress line. Returns number of connected drones.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        connected = sum(1 for r in receivers.values() if r.is_connected())
        print(
            f"\r  Waiting for {label} — {connected}/{n_expected} connected  ",
            end="",
            flush=True,
        )
        if connected == n_expected:
            break
        time.sleep(0.2)

    connected = sum(1 for r in receivers.values() if r.is_connected())
    print(f"\r  Connected: {connected}/{n_expected} drones                      ")
    return connected


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def bar(ms: float, total_ms: float, width: int = 30) -> str:
    """Return a filled ASCII progress bar string."""
    frac = min(ms / total_ms, 1.0) if total_ms > 0 else 0.0
    filled = int(frac * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {ms:>8.1f} ms  ({frac*100:.1f}%)"


def print_stage_bars(stage_ms: dict[str, float]) -> None:
    """Print a table of stage name → timing bar."""
    total = stage_ms.get("total") or sum(v for k, v in stage_ms.items() if k != "total") or 1.0
    for name, ms in stage_ms.items():
        if name == "total":
            continue
        label = f"  {name:<18}"
        print(f"{label} {bar(ms, total)}")
    print(f"  {'─'*64}")
    print(f"  {'TOTAL':<18} {'':>32}   {total:>8.1f} ms")


def print_header(title: str, drone_ids: list[int]) -> None:
    ports = [16000 + d - 1 for d in drone_ids]
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")
    print(f"  Drones : {drone_ids}")
    print(f"  Ports  : {ports}")
    print(f"{'='*64}")
    print(f"\n  Make sure ENet streamers are running:")
    print(f"    python enet_drone_streamer/run_all_drones.py --dataset MATRIX_30x30/MATRIX_30x30 --drones {','.join(str(d) for d in drone_ids)}")
    print(f"\n  Connecting...\n")