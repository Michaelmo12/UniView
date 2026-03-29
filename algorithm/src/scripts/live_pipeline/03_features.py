"""
Stage 3: Ingestion + Detection + Feature Extraction (WCH)

Adds WCHExtractor and ProjectionMatrixCalculator on top of stage 2.
This is the KEY script to diagnose the WCH slowdown: in isolation (no ENet
threads decoding JPEGs simultaneously), WCH should be ~16ms per drone.
If it is still slow here, the bottleneck is NOT thread contention.

Usage (from UniView/ root):
    python algorithm/src/scripts/live_pipeline/03_features.py
    python algorithm/src/scripts/live_pipeline/03_features.py --drones 3,4,6,7 --frames 10
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
    print_header, print_stage_bars,
)
from src.config.settings import settings
from src.detection.batch_detector import BatchDetector
from src.features.wch_extractor import WCHExtractor
from src.features.projection_matrix import ProjectionMatrixCalculator


def main() -> None:
    args = base_parser("Stage 3 — Ingestion + Detection + WCH Features").parse_args()
    drone_ids = parse_drone_ids(args.drones)

    print_header("Stage 3: Ingestion + Detection + WCH Features", drone_ids)

    print("  Initializing components...")
    t_init = time.monotonic()
    batch_detector  = BatchDetector()
    wch_extractor   = WCHExtractor(settings.features)
    projection_calc = ProjectionMatrixCalculator()
    print(f"  Ready in {(time.monotonic()-t_init)*1000:.0f} ms\n")

    receivers, synchronizer, sync_queue = start_pipeline(drone_ids)
    connected = wait_for_connections(receivers, len(drone_ids))
    if connected == 0:
        print("\n  ERROR: no drones connected.")
        stop_pipeline(receivers, synchronizer)
        sys.exit(1)

    print(f"\n  {'Frame':>6}  {'Det ms':>7}  {'Feat ms':>8}  {'Gap ms':>7}  Per-drone WCH (ms / features)")
    print(f"  {'-'*80}")

    sets_received  = 0
    det_times:  list[float] = []
    feat_times: list[float] = []
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

            # Stage 1: Detection
            t0 = time.monotonic()
            detection_sets = batch_detector.process(sync_set)
            det_ms = (time.monotonic() - t0) * 1000
            det_times.append(det_ms)

            # Stage 2: WCH + projection
            t1 = time.monotonic()
            calibrations       = sync_set.get_all_calibrations()
            projection_matrices = projection_calc.compute_batch(calibrations)
            features_dict: dict = {}
            wch_per_drone: dict[int, tuple[float, int]] = {}

            for drone_id, drone_frame in sync_set.frames.items():
                det_set = detection_sets.get(drone_id)
                tw = time.monotonic()
                if det_set and not det_set.is_empty:
                    ff = wch_extractor.extract_frame(frame=drone_frame, detectionSet=det_set)
                    features_dict[drone_id] = ff.features
                else:
                    features_dict[drone_id] = []
                wch_ms = (time.monotonic() - tw) * 1000
                wch_per_drone[drone_id] = (wch_ms, len(features_dict[drone_id]))

            feat_ms = (time.monotonic() - t1) * 1000
            feat_times.append(feat_ms)

            drone_detail = "  ".join(
                f"D{d}:{wch_per_drone[d][0]:.0f}ms/{wch_per_drone[d][1]}f"
                for d in sorted(wch_per_drone)
            )

            print(
                f"  {sync_set.frame_num:>6}  {det_ms:>7.1f}  {feat_ms:>8.1f}  "
                f"{gap_ms:>7.0f}  {drone_detail}"
            )

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        elapsed = time.monotonic() - t_start
        if det_times:
            print(f"\n  Summary over {len(det_times)} frames:")
            print(f"    Detection : avg={sum(det_times)/len(det_times):.1f}ms  max={max(det_times):.1f}ms")
            print(f"    WCH total : avg={sum(feat_times)/len(feat_times):.1f}ms  max={max(feat_times):.1f}ms")
        print(f"  Elapsed: {elapsed:.1f}s")
        stop_pipeline(receivers, synchronizer)


if __name__ == "__main__":
    main()
