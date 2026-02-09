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
    import sys

    print()
    print("=" * 70)
    print("Automatic Static BBox Remover")
    print("=" * 70)
    print()
    print("⚠️  WARNING: This will automatically process ALL frames")
    print("   Make sure you've tested with the interactive script first!")
    print()

    # Get parameters
    if len(sys.argv) >= 2:
        variance_threshold = int(sys.argv[1])
        start_drone = int(sys.argv[2]) if len(sys.argv) >= 3 else 1
        end_drone = int(sys.argv[3]) if len(sys.argv) >= 4 else 8
        use_threads = sys.argv[4].lower() != "false" if len(sys.argv) >= 5 else True
    else:
        variance_threshold = int(input("Variance Threshold (default 80): ") or "80")
        start_drone = int(input("Start Drone (1-8, default 1): ") or "1")
        end_drone = int(input("End Drone (1-8, default 8): ") or "8")
        use_threads_input = (
            input("Use multi-threading? (y/n, default y): ").strip().lower()
        )
        use_threads = use_threads_input != "n"

    print()
    print(f"Configuration:")
    print(f"  Variance Threshold: {variance_threshold}")
    print(f"  Drones: {start_drone}-{end_drone}")
    print(f"  Multi-threading: {use_threads}")
    print()

    confirm = input("Continue? (yes/no): ").strip().lower()

    if confirm == "yes":
        print()
        batch_process_automatic(variance_threshold, start_drone, end_drone, use_threads)
    else:
        print("\nCancelled - no files were modified.")
