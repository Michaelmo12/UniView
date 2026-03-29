"""
Stage 4: Ingestion + Detection + Features + Cross-Camera Fusion

Adds CrossCameraMatcher (epipolar + WCH Hungarian matching) on top of stage 3.
Shows match groups, pairwise matches, and fusion timing.

Usage (from UniView/ root):
    python algorithm/src/scripts/live_pipeline/04_fusion.py
    python algorithm/src/scripts/live_pipeline/04_fusion.py --drones 3,4,6,7 --frames 10
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
from src.config.settings import settings
from src.detection.batch_detector import BatchDetector
from src.features.wch_extractor import WCHExtractor
from src.features.projection_matrix import ProjectionMatrixCalculator
from src.fusion.cross_camera_matcher import CrossCameraMatcher


def main() -> None:
    args = base_parser("Stage 4 — Ingestion + Detection + Features + Fusion").parse_args()
    drone_ids = parse_drone_ids(args.drones)

    print_header("Stage 4: Ingestion + Detection + Features + Fusion", drone_ids)

    print("  Initializing components...")
    t_init = time.monotonic()
    batch_detector       = BatchDetector()
    wch_extractor        = WCHExtractor(settings.features)
    projection_calc      = ProjectionMatrixCalculator()
    cross_camera_matcher = CrossCameraMatcher(settings.fusion)
    print(f"  Ready in {(time.monotonic()-t_init)*1000:.0f} ms\n")

    receivers, synchronizer, sync_queue = start_pipeline(drone_ids)
    connected = wait_for_connections(receivers, len(drone_ids))
    if connected == 0:
        print("\n  ERROR: no drones connected.")
        stop_pipeline(receivers, synchronizer)
        sys.exit(1)

    print(f"\n  {'Frame':>6}  {'Det ms':>7}  {'Feat ms':>8}  {'Fus ms':>7}  {'Groups':>6}  {'Pairs':>6}  {'Unmatched':>9}")
    print(f"  {'-'*75}")

    sets_received = 0
    stage_times: dict[str, list[float]] = {"det": [], "feat": [], "fus": []}
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
            last_t = t_now
            sets_received += 1

            # Stage 1: Detection
            t0 = time.monotonic()
            detection_sets = batch_detector.process(sync_set)
            det_ms = (time.monotonic() - t0) * 1000
            stage_times["det"].append(det_ms)

            # Stage 2: Features
            t1 = time.monotonic()
            calibrations        = sync_set.get_all_calibrations()
            projection_matrices = projection_calc.compute_batch(calibrations)
            features_dict: dict = {}
            for drone_id, drone_frame in sync_set.frames.items():
                det_set = detection_sets.get(drone_id)
                if det_set and not det_set.is_empty:
                    ff = wch_extractor.extract_frame(frame=drone_frame, detectionSet=det_set)
                    features_dict[drone_id] = ff.features
                else:
                    features_dict[drone_id] = []
            feat_ms = (time.monotonic() - t1) * 1000
            stage_times["feat"].append(feat_ms)

            # Stage 3: Fusion
            t2 = time.monotonic()
            fusion_result = cross_camera_matcher.match_frame(
                detection_sets=detection_sets,
                projection_matrices=projection_matrices,
                features_dict=features_dict,
            )
            fus_ms = (time.monotonic() - t2) * 1000
            stage_times["fus"].append(fus_ms)

            n_groups    = len(fusion_result.match_groups)
            n_pairs     = fusion_result.total_matches
            n_unmatched = len(fusion_result.unmatched_detections)

            print(
                f"  {sync_set.frame_num:>6}  {det_ms:>7.1f}  {feat_ms:>8.1f}  "
                f"{fus_ms:>7.1f}  {n_groups:>6}  {n_pairs:>6}  {n_unmatched:>9}"
            )

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        elapsed = time.monotonic() - t_start
        n = len(stage_times["det"])
        if n:
            print(f"\n  Summary over {n} frames:")
            for label, key in [("Detection", "det"), ("Features ", "feat"), ("Fusion   ", "fus")]:
                vals = stage_times[key]
                print(f"    {label}: avg={sum(vals)/len(vals):.1f}ms  max={max(vals):.1f}ms")
        print(f"  Elapsed: {elapsed:.1f}s")
        stop_pipeline(receivers, synchronizer)


if __name__ == "__main__":
    main()
