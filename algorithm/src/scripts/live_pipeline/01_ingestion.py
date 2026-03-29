"""
Stage 1: Ingestion Only

Connects ENet receivers → FrameSynchronizer, shows per-set stats.
No processing beyond receiving and synchronizing frames.

Usage (from UniView/ root):
    python algorithm/src/scripts/live_pipeline/01_ingestion.py
    python algorithm/src/scripts/live_pipeline/01_ingestion.py --drones 3,4,6,7 --frames 30
"""

from __future__ import annotations

import queue
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_ALGO_ROOT  = _SCRIPT_DIR.parents[2]
if str(_ALGO_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALGO_ROOT))

import logging
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

from _helpers import (
    base_parser, parse_drone_ids,
    start_pipeline, stop_pipeline, wait_for_connections,
    print_header,
)


def main() -> None:
    args = base_parser("Stage 1 — ENet ingestion + frame synchronization").parse_args()
    drone_ids = parse_drone_ids(args.drones)

    print_header("Stage 1: Ingestion Only", drone_ids)

    receivers, synchronizer, sync_queue = start_pipeline(drone_ids)

    connected = wait_for_connections(receivers, len(drone_ids))
    if connected == 0:
        print("\n  ERROR: no drones connected. Is the ENet streamer running?")
        stop_pipeline(receivers, synchronizer)
        sys.exit(1)

    print(f"  {'Set':>4}  {'FrameNum':>8}  {'Drones':>9}  {'Status':>12}  {'Gap ms':>7}  {'Size':>10}  Present")
    print(f"  {'-'*80}")

    sets_received = 0
    sets_complete = 0
    last_t: float | None = None
    t_start = time.monotonic()

    try:
        while sets_received < args.frames:
            try:
                sync_set = sync_queue.get(timeout=args.timeout)
            except queue.Empty:
                print(f"\n  TIMEOUT: no frame in {args.timeout}s. Received {sets_received}/{args.frames}.")
                break

            t_now = time.monotonic()
            gap_ms = (t_now - last_t) * 1000 if last_t else 0.0
            last_t = t_now
            sets_received += 1

            is_complete = sync_set.is_complete
            if is_complete:
                sets_complete += 1

            first_frame = next(iter(sync_set.frames.values()))
            h, w = first_frame.frame.shape[:2]
            present = sorted(sync_set.frames.keys())
            status = "COMPLETE" if is_complete else f"PARTIAL {len(present)}/{len(drone_ids)}"

            print(
                f"  {sets_received:>4}  {sync_set.frame_num:>8}  "
                f"{sync_set.num_drones_present:>3}/{len(drone_ids):<5}  "
                f"{status:>12}  "
                f"{gap_ms:>7.0f}  "
                f"{w}x{h:>4}  "
                f"{present}"
            )

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        elapsed = time.monotonic() - t_start
        fps = sets_received / elapsed if elapsed > 0 else 0
        print(f"\n  Sets received : {sets_received}")
        print(f"  Complete sets : {sets_complete}  ({sets_complete/sets_received*100:.0f}%)" if sets_received else "")
        print(f"  Elapsed       : {elapsed:.1f}s  ({fps:.2f} sets/s)")
        total_rx = sum(r.frames_received for r in receivers.values())
        print(f"  Total frames  : {total_rx}  ({sum(r.errors for r in receivers.values())} errors)")
        stop_pipeline(receivers, synchronizer)


if __name__ == "__main__":
    main()
