"""
Stage 2: Ingestion + Detection

Adds YOLO BatchDetector on top of stage 1.
Shows per-drone detection counts and detection time per frame.

Usage (from UniView/ root):
    python algorithm/src/scripts/live_pipeline/02_detection.py
    python algorithm/src/scripts/live_pipeline/02_detection.py --drones 3,4,6,7 --frames 20
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
from src.detection.batch_detector import BatchDetector


def main() -> None:
    args = base_parser("Stage 2 — Ingestion + YOLO Detection").parse_args()
    drone_ids = parse_drone_ids(args.drones)

    print_header("Stage 2: Ingestion + Detection", drone_ids)

    print("  Initializing YOLO detector (includes warmup)...")
    t_init = time.monotonic()
    batch_detector = BatchDetector()
    print(f"  Detector ready in {(time.monotonic()-t_init)*1000:.0f} ms\n")

    receivers, synchronizer, sync_queue = start_pipeline(drone_ids)
    connected = wait_for_connections(receivers, len(drone_ids))
    if connected == 0:
        print("\n  ERROR: no drones connected.")
        stop_pipeline(receivers, synchronizer)
        sys.exit(1)

    print(f"\n  {'Frame':>6}  {'Det ms':>7}  {'Gap ms':>7}  Per-drone detections")
    print(f"  {'-'*70}")

    sets_received = 0
    det_times: list[float] = []
    last_t: float | None = None
    t_start = time.monotonic()

    try:
        while sets_received < args.frames:
            try:
                sync_set = sync_queue.get(timeout=args.timeout)
            except queue.Empty:
                print(f"\n  TIMEOUT: no frame in {args.timeout}s.")
                break

            t_now = time.monotonic()
            gap_ms = (t_now - last_t) * 1000 if last_t else 0.0
            last_t = t_now
            sets_received += 1

            t0 = time.monotonic()
            detection_sets = batch_detector.process(sync_set)
            det_ms = (time.monotonic() - t0) * 1000
            det_times.append(det_ms)

            total_dets = sum(ds.num_detections for ds in detection_sets.values())
            per_drone = "  ".join(
                f"D{d}:{detection_sets[d].num_detections if d in detection_sets else '?'}"
                for d in drone_ids
            )

            print(
                f"  {sync_set.frame_num:>6}  {det_ms:>7.1f}  {gap_ms:>7.0f}  "
                f"{per_drone}  (total={total_dets})"
            )

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        elapsed = time.monotonic() - t_start
        if det_times:
            avg = sum(det_times) / len(det_times)
            mn  = min(det_times)
            mx  = max(det_times)
            print(f"\n  Detection timing over {len(det_times)} frames:")
            print(f"    avg={avg:.1f}ms  min={mn:.1f}ms  max={mx:.1f}ms")
        print(f"  Elapsed: {elapsed:.1f}s")
        stop_pipeline(receivers, synchronizer)


if __name__ == "__main__":
    main()
