"""
Stage 5: Ingestion + Detection + Features + Fusion + 3D Reconstruction

Adds SceneReconstructor on top of stage 4.
Shows triangulated persons, single-view persons, and reconstruction timing.

Usage (from UniView/ root):
    python algorithm/src/scripts/live_pipeline/05_reconstruction.py
    python algorithm/src/scripts/live_pipeline/05_reconstruction.py --drones 3,4,6,7 --frames 10
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
from src.reconstruction.scene_reconstructor import SceneReconstructor


def main() -> None:
    args = base_parser("Stage 5 — Ingestion + Detection + Features + Fusion + Reconstruction").parse_args()
    drone_ids = parse_drone_ids(args.drones)

    print_header("Stage 5: + 3D Reconstruction", drone_ids)

    print("  Initializing components...")
    t_init = time.monotonic()
    batch_detector       = BatchDetector()
    wch_extractor        = WCHExtractor(settings.features)
    projection_calc      = ProjectionMatrixCalculator()
    cross_camera_matcher = CrossCameraMatcher(settings.fusion)
    scene_reconstructor  = SceneReconstructor(settings.reconstruction)
    print(f"  Ready in {(time.monotonic()-t_init)*1000:.0f} ms\n")

    receivers, synchronizer, sync_queue = start_pipeline(drone_ids)
    connected = wait_for_connections(receivers, len(drone_ids))
    if connected == 0:
        print("\n  ERROR: no drones connected.")
        stop_pipeline(receivers, synchronizer)
        sys.exit(1)

    print(f"\n  {'Frame':>6}  {'Det':>6}  {'Feat':>6}  {'Fus':>6}  {'Rec':>6}  {'Total':>7}  {'Tri':>4}  {'SV':>4}  {'Persons':>7}")
    print(f"  {'-'*75}")

    sets_received = 0
    all_totals: list[float] = []
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

            t0 = time.monotonic()
            detection_sets = batch_detector.process(sync_set)
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
            t2 = time.monotonic()

            fusion_result = cross_camera_matcher.match_frame(
                detection_sets=detection_sets,
                projection_matrices=projection_matrices,
                features_dict=features_dict,
            )
            t3 = time.monotonic()

            reconstruction_result = scene_reconstructor.reconstruct(
                fusion_result=fusion_result,
                detection_sets=detection_sets,
                sync_set=sync_set,
            )
            t4 = time.monotonic()

            det_ms  = (t1 - t0) * 1000
            feat_ms = (t2 - t1) * 1000
            fus_ms  = (t3 - t2) * 1000
            rec_ms  = (t4 - t3) * 1000
            total   = (t4 - t0) * 1000
            all_totals.append(total)

            n_tri = len(reconstruction_result.triangulated_persons)
            n_sv  = len(reconstruction_result.single_view_persons)
            n_tot = reconstruction_result.num_persons

            print(
                f"  {sync_set.frame_num:>6}  {det_ms:>6.0f}  {feat_ms:>6.0f}  "
                f"{fus_ms:>6.0f}  {rec_ms:>6.0f}  {total:>7.0f}  "
                f"{n_tri:>4}  {n_sv:>4}  {n_tot:>7}"
            )

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        elapsed = time.monotonic() - t_start
        if all_totals:
            avg = sum(all_totals) / len(all_totals)
            print(f"\n  Total pipeline avg={avg:.1f}ms over {len(all_totals)} frames")
        print(f"  Elapsed: {elapsed:.1f}s")
        stop_pipeline(receivers, synchronizer)


if __name__ == "__main__":
    main()
