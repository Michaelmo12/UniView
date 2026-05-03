"""
Automatic Batch Static BBox Remover

Automatically removes static/uniform bboxes from all frames in all drone folders
without user interaction. Use this after verifying the logic with the interactive script.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse


def calculate_bbox_variance(image, bbox):
    """Calculate pixel variance within a bbox."""
    xmin = bbox["xmin"]
    ymin = bbox["ymin"]
    xmax = bbox["xmax"]
    ymax = bbox["ymax"]

    # Validate bbox coordinates
    img_height, img_width = image.shape[:2]

    # Check if bbox is valid
    if xmin >= xmax or ymin >= ymax:
        return 0  # Invalid bbox, treat as uniform

    # Clip to image bounds
    xmin = max(0, min(xmin, img_width - 1))
    xmax = max(0, min(xmax, img_width))
    ymin = max(0, min(ymin, img_height - 1))
    ymax = max(0, min(ymax, img_height))

    # Check again after clipping
    if xmin >= xmax or ymin >= ymax:
        return 0

    # Extract ROI
    roi = image[ymin:ymax, xmin:xmax]

    # Check if ROI is empty
    if roi.size == 0:
        return 0

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray_roi)

    return variance


def process_frame(
    drone_id, frame_num, variance_threshold, images_dir, annotations_dir, labels_dir
):
    """
    Process a single frame and save filtered labels.

    Now reads from YOLO labels in image_subsets if they exist,
    otherwise reads from original JSON annotations.

    Returns:
        dict: Statistics about the frame processing
    """
    img_path = images_dir / f"{frame_num:04d}.png"
    label_path = labels_dir / f"{frame_num:04d}.txt"
    json_path = annotations_dir / f"{frame_num:04d}.json"

    # Check if image exists
    if not img_path.exists():
        return None

    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    img_height, img_width = img.shape[:2]
    view_num = drone_id - 1

    # Try to load from YOLO labels first (already filtered)
    current_bboxes = []

    if label_path.exists():
        # Read YOLO labels
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # class_id, x_center, y_center, width, height (normalized)
                    x_center = float(parts[1]) * img_width
                    y_center = float(parts[2]) * img_height
                    w = float(parts[3]) * img_width
                    h = float(parts[4]) * img_height

                    xmin = int(x_center - w / 2)
                    ymin = int(y_center - h / 2)
                    xmax = int(x_center + w / 2)
                    ymax = int(y_center + h / 2)

                    current_bboxes.append(
                        {
                            "xmin": xmin,
                            "ymin": ymin,
                            "xmax": xmax,
                            "ymax": ymax,
                            "person_id": len(current_bboxes) + 1,  # Dummy ID
                        }
                    )

    else:
        # Fallback to JSON annotations
        if not json_path.exists():
            return None

        with open(json_path, "r") as f:
            annotations = json.load(f)

        for person in annotations:
            person_id = person["personID"]

            for view in person["views"]:
                if view["viewNum"] == view_num:
                    xmin = view["xmin"]
                    ymin = view["ymin"]
                    xmax = view["xmax"]
                    ymax = view["ymax"]

                    if xmin == -1:
                        break

                    current_bboxes.append(
                        {
                            "xmin": xmin,
                            "ymin": ymin,
                            "xmax": xmax,
                            "ymax": ymax,
                            "person_id": person_id,
                        }
                    )

                    break

    # Filter bboxes based on variance
    filtered_bboxes = []
    stats = {
        "frame_num": frame_num,
        "total_bboxes": 0,
        "removed_bboxes": 0,
        "kept_bboxes": 0,
    }

    for bbox_info in current_bboxes:
        stats["total_bboxes"] += 1

        bbox = {
            "xmin": bbox_info["xmin"],
            "ymin": bbox_info["ymin"],
            "xmax": bbox_info["xmax"],
            "ymax": bbox_info["ymax"],
        }

        variance = calculate_bbox_variance(img, bbox)
        is_uniform = variance < variance_threshold

        if is_uniform:
            stats["removed_bboxes"] += 1
        else:
            stats["kept_bboxes"] += 1
            filtered_bboxes.append(bbox)

    # Save labels if any bboxes were removed
    if stats["removed_bboxes"] > 0:
        save_filtered_labels_yolo(drone_id, frame_num, filtered_bboxes, labels_dir, img)

    return stats


def _bbox_from_view(view: dict) -> dict:
    return {
        "xmin": int(view["xmin"]),
        "ymin": int(view["ymin"]),
        "xmax": int(view["xmax"]),
        "ymax": int(view["ymax"]),
    }


def process_real_matrix_frame(
    frame_num: int,
    drone_ids: list[int],
    variance_threshold: float,
    real_dataset_root: Path,
    output_annotations_dir: Path,
) -> dict:
    """Filter static bboxes in original MATRIX JSON annotations for one frame.

    Writes a new JSON file to output_annotations_dir and never modifies source JSON.
    """
    annotations_dir = real_dataset_root / "annotations_positions"
    images_root = real_dataset_root / "image_subsets"

    in_json = annotations_dir / f"{frame_num:04d}.json"
    if not in_json.exists():
        return {
            "frame_num": frame_num,
            "exists": False,
            "total_bboxes": 0,
            "removed_bboxes": 0,
            "kept_bboxes": 0,
        }

    with open(in_json, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    images_by_drone: dict[int, np.ndarray] = {}
    for drone_id in drone_ids:
        img_path = images_root / f"D{drone_id}" / f"{frame_num:04d}.png"
        img = cv2.imread(str(img_path))
        if img is not None:
            images_by_drone[drone_id] = img

    stats = {
        "frame_num": frame_num,
        "exists": True,
        "total_bboxes": 0,
        "removed_bboxes": 0,
        "kept_bboxes": 0,
    }

    drone_set = set(drone_ids)

    # Filter per-view bbox. If a view is static/uniform, mark it as absent (-1s)
    # while preserving the person and all other views.
    for person in annotations:
        for view in person.get("views", []):
            view_num = int(view.get("viewNum", -1))
            drone_id = view_num + 1
            if drone_id not in drone_set:
                continue
            if drone_id not in images_by_drone:
                continue

            xmin = int(view.get("xmin", -1))
            ymin = int(view.get("ymin", -1))
            xmax = int(view.get("xmax", -1))
            ymax = int(view.get("ymax", -1))

            if xmin == -1 or ymin == -1 or xmax == -1 or ymax == -1:
                continue

            stats["total_bboxes"] += 1
            bbox = _bbox_from_view(view)
            variance = calculate_bbox_variance(images_by_drone[drone_id], bbox)

            if variance < variance_threshold:
                view["xmin"] = -1
                view["ymin"] = -1
                view["xmax"] = -1
                view["ymax"] = -1
                stats["removed_bboxes"] += 1
            else:
                stats["kept_bboxes"] += 1

    output_annotations_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_annotations_dir / f"{frame_num:04d}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)

    return stats


def batch_process_real_matrix(
    variance_threshold: float = 80,
    start_drone: int = 1,
    end_drone: int = 8,
    drone_ids_override: list[int] | None = None,
    start_frame: int = 0,
    num_frames: int = 200,
    real_dataset_root: Path | None = None,
    output_annotations_dir: Path | None = None,
) -> None:
    """Create a cleaned annotation folder from original MATRIX JSON annotations."""
    script_root = Path(__file__).parent
    if real_dataset_root is None:
        real_dataset_root = script_root.parent.parent.parent / "MATRIX_30x30" / "MATRIX_30x30"

    if output_annotations_dir is None:
        output_annotations_dir = real_dataset_root / f"annotations_positions_static_filtered_v{int(variance_threshold)}"

    if drone_ids_override:
        drone_ids = sorted(set(drone_ids_override))
    else:
        drone_ids = list(range(start_drone, end_drone + 1))
    frame_nums = list(range(start_frame, start_frame + num_frames))

    print("=" * 80)
    print("Real MATRIX Static BBox Filter (non-destructive)")
    print("=" * 80)
    print(f"Dataset root: {real_dataset_root}")
    print(f"Output annotations: {output_annotations_dir}")
    print(f"Drones: {start_drone}-{end_drone}")
    print(f"Frames: {start_frame}-{start_frame + num_frames - 1} ({num_frames} total)")
    print(f"Variance threshold: {variance_threshold}")
    print("=" * 80)

    total = {
        "frames_written": 0,
        "frames_missing": 0,
        "total_bboxes": 0,
        "removed_bboxes": 0,
        "kept_bboxes": 0,
    }
    per_drone_removed = {d: 0 for d in drone_ids}

    for frame_num in tqdm(frame_nums, desc="Filtering real MATRIX frames"):
        stats = process_real_matrix_frame(
            frame_num=frame_num,
            drone_ids=drone_ids,
            variance_threshold=variance_threshold,
            real_dataset_root=real_dataset_root,
            output_annotations_dir=output_annotations_dir,
        )

        if not stats["exists"]:
            total["frames_missing"] += 1
            continue

        total["frames_written"] += 1
        total["total_bboxes"] += stats["total_bboxes"]
        total["removed_bboxes"] += stats["removed_bboxes"]
        total["kept_bboxes"] += stats["kept_bboxes"]

        # Recount removals per drone for this frame using saved JSON + source JSON
        # to keep this utility simple and robust even when view order varies.
        src_json = real_dataset_root / "annotations_positions" / f"{frame_num:04d}.json"
        dst_json = output_annotations_dir / f"{frame_num:04d}.json"
        with open(src_json, "r", encoding="utf-8") as f:
            src = json.load(f)
        with open(dst_json, "r", encoding="utf-8") as f:
            dst = json.load(f)

        for p_src, p_dst in zip(src, dst):
            for v_src, v_dst in zip(p_src.get("views", []), p_dst.get("views", [])):
                d = int(v_src.get("viewNum", -1)) + 1
                if d not in per_drone_removed:
                    continue
                src_vis = int(v_src.get("xmin", -1)) != -1
                dst_vis = int(v_dst.get("xmin", -1)) != -1
                if src_vis and (not dst_vis):
                    per_drone_removed[d] += 1

    summary_path = output_annotations_dir.parent / f"{output_annotations_dir.name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_root": str(real_dataset_root),
                "output_annotations": str(output_annotations_dir),
                "variance_threshold": variance_threshold,
                "start_drone": start_drone,
                "end_drone": end_drone,
                "start_frame": start_frame,
                "num_frames": num_frames,
                "totals": total,
                "removed_by_drone": per_drone_removed,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("Completed")
    print("=" * 80)
    print(f"Frames written: {total['frames_written']}")
    print(f"Frames missing: {total['frames_missing']}")
    print(f"Total bboxes checked: {total['total_bboxes']}")
    print(f"Removed static bboxes: {total['removed_bboxes']}")
    print(f"Kept bboxes: {total['kept_bboxes']}")
    print("Removed by drone:")
    for d in drone_ids:
        print(f"  D{d}: {per_drone_removed[d]}")
    print(f"Output annotations folder: {output_annotations_dir}")
    print(f"Summary file: {summary_path}")


def save_filtered_labels_yolo(drone_id, frame_num, filtered_bboxes, labels_dir, img):
    """
    Save filtered bboxes to labels directory in YOLO format.

    Args:
        drone_id: Drone ID
        frame_num: Frame number
        filtered_bboxes: List of bbox dicts with xmin, ymin, xmax, ymax
        labels_dir: Path to labels directory
        img: Image for getting dimensions
    """
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    label_path = labels_dir / f"{frame_num:04d}.txt"

    img_height, img_width = img.shape[:2]

    # Write YOLO format labels
    with open(label_path, "w") as f:
        for bbox in filtered_bboxes:
            xmin = bbox["xmin"]
            ymin = bbox["ymin"]
            xmax = bbox["xmax"]
            ymax = bbox["ymax"]

            # Convert to YOLO format (normalized x_center, y_center, width, height)
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            # Class 0 for person
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def save_filtered_labels(drone_id, frame_num, filtered_annotations, labels_dir, img):
    """
    DEPRECATED: Old function for JSON annotations. Kept for compatibility.
    Use save_filtered_labels_yolo instead.
    """
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    label_path = labels_dir / f"{frame_num:04d}.txt"

    img_height, img_width = img.shape[:2]
    view_num = drone_id - 1

    # Write YOLO format labels
    with open(label_path, "w") as f:
        for person in filtered_annotations:
            for view in person["views"]:
                if view["viewNum"] == view_num:
                    xmin = view["xmin"]
                    ymin = view["ymin"]
                    xmax = view["xmax"]
                    ymax = view["ymax"]

                    if xmin == -1:
                        continue

                    # Convert to YOLO format (normalized x_center, y_center, width, height)
                    x_center = ((xmin + xmax) / 2) / img_width
                    y_center = ((ymin + ymax) / 2) / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height

                    # Class 0 for person
                    f.write(
                        f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                    )

                    break


def process_drone(drone_id, variance_threshold, dataset_root, annotations_dir):
    """
    Process all frames for a single drone.

    Returns:
        dict: Statistics for the drone
    """
    images_dir = dataset_root / f"D{drone_id}"
    if not images_dir.exists():
        return None

    labels_dir = dataset_root / f"D{drone_id}" / "labels"

    frame_files = sorted(images_dir.glob("*.png"))

    drone_stats = {
        "drone_id": drone_id,
        "frames_processed": 0,
        "frames_modified": 0,
        "total_bboxes_removed": 0,
    }

    for img_file in tqdm(frame_files, desc=f"Drone {drone_id}", position=drone_id - 1):
        frame_num = int(img_file.stem)

        stats = process_frame(
            drone_id,
            frame_num,
            variance_threshold,
            images_dir,
            annotations_dir,
            labels_dir,
        )

        if stats:
            drone_stats["frames_processed"] += 1
            if stats["removed_bboxes"] > 0:
                drone_stats["frames_modified"] += 1
                drone_stats["total_bboxes_removed"] += stats["removed_bboxes"]

    return drone_stats


def batch_process_automatic(
    variance_threshold=100, start_drone=1, end_drone=8, use_threads=True
):
    """
    Automatically process all frames for all drones without user interaction.

    Args:
        variance_threshold: Variance threshold for filtering
        start_drone: Starting drone ID
        end_drone: Ending drone ID
        use_threads: Use multi-threading for parallel processing
    """
    dataset_root = Path(__file__).parent
    annotations_dir = (
        dataset_root.parent / "MATRIX_30x30" / "MATRIX_30x30" / "annotations_positions"
    )

    print("=" * 70)
    print("Automatic Batch Static BBox Remover")
    print("=" * 70)
    print(f"Variance Threshold: {variance_threshold}")
    print(f"Processing Drones: {start_drone}-{end_drone}")
    print(f"Multi-threading: {use_threads}")
    print("=" * 70)
    print()

    total_stats = {
        "total_frames_processed": 0,
        "total_frames_modified": 0,
        "total_bboxes_removed": 0,
    }

    drone_ids = range(start_drone, end_drone + 1)

    if use_threads:
        # Multi-threaded processing
        print("Starting multi-threaded processing...\n")

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(
                    process_drone,
                    drone_id,
                    variance_threshold,
                    dataset_root,
                    annotations_dir,
                ): drone_id
                for drone_id in drone_ids
            }

            for future in as_completed(futures):
                drone_id = futures[future]
                try:
                    drone_stats = future.result()

                    if drone_stats:
                        total_stats["total_frames_processed"] += drone_stats[
                            "frames_processed"
                        ]
                        total_stats["total_frames_modified"] += drone_stats[
                            "frames_modified"
                        ]
                        total_stats["total_bboxes_removed"] += drone_stats[
                            "total_bboxes_removed"
                        ]

                        print(f"\n✓ Drone {drone_id} complete:")
                        print(f"  Frames processed: {drone_stats['frames_processed']}")
                        print(f"  Frames modified: {drone_stats['frames_modified']}")
                        print(
                            f"  BBoxes removed: {drone_stats['total_bboxes_removed']}"
                        )

                except Exception as e:
                    print(f"\n✗ Error processing Drone {drone_id}: {e}")

    else:
        # Sequential processing
        for drone_id in drone_ids:
            print(f"\nProcessing Drone {drone_id}...")

            drone_stats = process_drone(
                drone_id, variance_threshold, dataset_root, annotations_dir
            )

            if drone_stats:
                total_stats["total_frames_processed"] += drone_stats["frames_processed"]
                total_stats["total_frames_modified"] += drone_stats["frames_modified"]
                total_stats["total_bboxes_removed"] += drone_stats[
                    "total_bboxes_removed"
                ]

                print(f"✓ Drone {drone_id} complete:")
                print(f"  Frames processed: {drone_stats['frames_processed']}")
                print(f"  Frames modified: {drone_stats['frames_modified']}")
                print(f"  BBoxes removed: {drone_stats['total_bboxes_removed']}")

    print(f"\n{'='*70}")
    print("Automatic Batch Processing Complete!")
    print(f"{'='*70}")
    print(f"Total frames processed: {total_stats['total_frames_processed']}")
    print(f"Total frames modified: {total_stats['total_frames_modified']}")
    print(f"Total bboxes removed: {total_stats['total_bboxes_removed']}")
    print(f"{'='*70}")

    # Show where labels were saved
    print(f"\nFiltered labels saved to:")
    for drone_id in drone_ids:
        labels_dir = dataset_root / f"D{drone_id}" / "labels"
        if labels_dir.exists():
            print(f"  D{drone_id}/labels/")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove static bboxes automatically from YOLO labels or real MATRIX JSON annotations"
    )
    parser.add_argument("--mode", choices=["yolo", "real-json"], default="real-json")
    parser.add_argument("--variance-threshold", type=float, default=80)
    parser.add_argument("--start-drone", type=int, default=1)
    parser.add_argument("--end-drone", type=int, default=8)

    # YOLO mode options
    parser.add_argument("--use-threads", action="store_true")

    # Real JSON mode options
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--real-dataset-root", type=str, default="")
    parser.add_argument("--output-annotations-dir", type=str, default="")
    parser.add_argument("--drone-ids", type=str, default="",
                        help="Comma-separated drone IDs to process, e.g. 3,4,6,7 (overrides --start-drone/--end-drone)")

    args = parser.parse_args()

    if args.mode == "yolo":
        batch_process_automatic(
            variance_threshold=args.variance_threshold,
            start_drone=args.start_drone,
            end_drone=args.end_drone,
            use_threads=args.use_threads,
        )
    else:
        real_root = Path(args.real_dataset_root) if args.real_dataset_root else None
        out_dir = Path(args.output_annotations_dir) if args.output_annotations_dir else None
        drone_ids_override = [int(x) for x in args.drone_ids.split(",") if x.strip()] if args.drone_ids else None
        batch_process_real_matrix(
            variance_threshold=args.variance_threshold,
            start_drone=args.start_drone,
            end_drone=args.end_drone,
            drone_ids_override=drone_ids_override,
            start_frame=args.start_frame,
            num_frames=args.num_frames,
            real_dataset_root=real_root,
            output_annotations_dir=out_dir,
        )
