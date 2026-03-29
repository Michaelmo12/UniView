"""
Stage 6: Full Pipeline (all stages including Tracking)

The complete live pipeline with per-stage timing bars printed every frame.
Equivalent to inference_pipeline.py but runs standalone (no FastAPI, no HTTP POST).

This is the definitive benchmark for the LIVE pipeline — compare against
benchmark_single_frame.py to see the overhead from ENet receiver threads.

Usage (from UniView/ root):
    python algorithm/src/scripts/live_pipeline/06_tracking.py
    python algorithm/src/scripts/live_pipeline/06_tracking.py --drones 3,4,6,7 --frames 10
    python algorithm/src/scripts/live_pipeline/06_tracking.py --frames 0   # run forever
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
from src.fusion.cross_camera_matcher import CrossCameraMatcher
from src.reconstruction.scene_reconstructor import SceneReconstructor
from src.tracking.tracker import PersonTracker


def main() -> None:
    parser = base_parser("Stage 6 — Full Pipeline (all stages)")
    parser.add_argument("--bars", action="store_true", default=True,
                        help="Print timing bars after each frame (default: on).")
    parser.add_argument("--no-bars", dest="bars", action="store_false",
                        help="Print only the summary line per frame.")
    args = parser.parse_args()
    # --frames 0 means run forever
    run_forever = args.frames == 0
    drone_ids = parse_drone_ids(args.drones)

    print_header("Stage 6: Full Pipeline", drone_ids)

    print("  Initializing all pipeline components...")
    t_init = time.monotonic()
    batch_detector       = BatchDetector()
    wch_extractor        = WCHExtractor(settings.features)
    projection_calc      = ProjectionMatrixCalculator()
    cross_camera_matcher = CrossCameraMatcher(settings.fusion)
    scene_reconstructor  = SceneReconstructor(settings.reconstruction)
    person_tracker       = PersonTracker()
    init_ms = (time.monotonic() - t_init) * 1000
    print(f"  All components ready in {init_ms:.0f} ms\n")

    receivers, synchronizer, sync_queue = start_pipeline(drone_ids)
    connected = wait_for_connections(receivers, len(drone_ids))
    if connected == 0:
        print("\n  ERROR: no drones connected.")
        stop_pipeline(receivers, synchronizer)
        sys.exit(1)

    print(f"\n  Running full pipeline... (press Ctrl+C to stop)\n")

    sets_received = 0
    all_stage_times: list[dict[str, float]] = []
    last_frame_t: float | None = None
    t_start = time.monotonic()

    try:
        while run_forever or sets_received < args.frames:
            try:
                sync_set = sync_queue.get(timeout=args.timeout)
            except queue.Empty:
                if not run_forever:
                    print(f"\n  TIMEOUT: no frame in {args.timeout}s.")
                    break
                continue

            t_now = time.monotonic()
            gap_ms = (t_now - last_frame_t) * 1000 if last_frame_t else 0.0
            last_frame_t = t_now
            sets_received += 1

            try:
                t0 = time.monotonic()

                # Stage 1: Detection
                detection_sets = batch_detector.process(sync_set)
                t1 = time.monotonic()

                # Stage 2: Features (with per-drone sub-timing)
                calibrations        = sync_set.get_all_calibrations()
                projection_matrices = projection_calc.compute_batch(calibrations)
                features_dict: dict = {}
                wch_drone_ms: dict[int, float] = {}
                for drone_id, drone_frame in sync_set.frames.items():
                    det_set = detection_sets.get(drone_id)
                    tw = time.monotonic()
                    if det_set and not det_set.is_empty:
                        ff = wch_extractor.extract_frame(frame=drone_frame, detectionSet=det_set)
                        features_dict[drone_id] = ff.features
                    else:
                        features_dict[drone_id] = []
                    wch_drone_ms[drone_id] = (time.monotonic() - tw) * 1000
                t2 = time.monotonic()

                # Stage 3: Fusion
                fusion_result = cross_camera_matcher.match_frame(
                    detection_sets=detection_sets,
                    projection_matrices=projection_matrices,
                    features_dict=features_dict,
                )
                t3 = time.monotonic()

                # Stage 4: Reconstruction
                reconstruction_result = scene_reconstructor.reconstruct(
                    fusion_result=fusion_result,
                    detection_sets=detection_sets,
                    sync_set=sync_set,
                )
                t4 = time.monotonic()

                # Stage 5: Tracking
                tracking_result = person_tracker.update(reconstruction_result)
                t5 = time.monotonic()

                stage_ms = {
                    "Detection":      round((t1 - t0) * 1000, 1),
                    "Features (WCH)": round((t2 - t1) * 1000, 1),
                    "Fusion":         round((t3 - t2) * 1000, 1),
                    "Reconstruction": round((t4 - t3) * 1000, 1),
                    "Tracking":       round((t5 - t4) * 1000, 1),
                    "total":          round((t5 - t0) * 1000, 1),
                }
                all_stage_times.append(stage_ms)

                n_confirmed = len(tracking_result.tracked_persons)
                n_tentative = tracking_result.num_tentative

                wch_detail = "  ".join(
                    f"D{d}:{wch_drone_ms[d]:.0f}ms" for d in sorted(wch_drone_ms)
                )

                print(f"\n  ── Frame {sync_set.frame_num}  gap={gap_ms:.0f}ms  "
                      f"drones={sync_set.num_drones_present}/{len(drone_ids)}  "
                      f"tracks={n_confirmed} confirmed / {n_tentative} tentative")
                print(f"     WCH per drone: {wch_detail}")

                if args.bars:
                    print_stage_bars(stage_ms)
                else:
                    print(f"     det={stage_ms['Detection']}ms  "
                          f"feat={stage_ms['Features (WCH)']}ms  "
                          f"fus={stage_ms['Fusion']}ms  "
                          f"rec={stage_ms['Reconstruction']}ms  "
                          f"trk={stage_ms['Tracking']}ms  "
                          f"TOTAL={stage_ms['total']}ms")

            except Exception as exc:
                print(f"  ERROR on frame {sync_set.frame_num}: {exc}")

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        elapsed = time.monotonic() - t_start
        if all_stage_times:
            n = len(all_stage_times)
            print(f"\n  ── Summary over {n} frames ──")
            for key in ["Detection", "Features (WCH)", "Fusion", "Reconstruction", "Tracking", "total"]:
                vals = [s[key] for s in all_stage_times if key in s]
                if vals:
                    label = "TOTAL" if key == "total" else key
                    print(f"    {label:<18} avg={sum(vals)/len(vals):.1f}ms  "
                          f"min={min(vals):.1f}ms  max={max(vals):.1f}ms")
        print(f"\n  Elapsed: {elapsed:.1f}s  ({sets_received} frames processed)")
        stop_pipeline(receivers, synchronizer)


if __name__ == "__main__":
    main()
