"""
Batch Interactive Static BBox Remover

Goes through all frames in all drone folders and lets you approve/reject
bbox filtering one frame at a time.
"""

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


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


def is_bbox_uniform(image, bbox, variance_threshold=100):
    """Check if bbox is uniform/static."""
    variance = calculate_bbox_variance(image, bbox)
    return variance < variance_threshold


def analyze_and_visualize_frame(drone_id, frame_num, variance_threshold=100):
    """
    Analyze frame and show before/after visualization.

    Now reads from YOLO labels in image_subsets if they exist,
    otherwise reads from original JSON annotations.

    Returns:
        tuple: (stats, filtered_annotations, fig) or (None, None, None) if error
    """
    # Paths - working in image_subsets directory
    dataset_root = Path(__file__).parent
    images_dir = dataset_root / f"D{drone_id}"
    labels_dir = dataset_root / f"D{drone_id}" / "labels"
    label_path = labels_dir / f"{frame_num:04d}.txt"

    # Annotations are in MATRIX_30x30 folder (fallback)
    annotations_dir = (
        dataset_root.parent / "MATRIX_30x30" / "MATRIX_30x30" / "annotations_positions"
    )

    img_path = images_dir / f"{frame_num:04d}.png"
    json_path = annotations_dir / f"{frame_num:04d}.json"

    # Check if image exists
    if not img_path.exists():
        return None, None, None

    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None, None

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
            return None, None, None

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

    # Analyze current bboxes
    stats = {
        "total_bboxes": 0,
        "removed_bboxes": 0,
        "kept_bboxes": 0,
        "removed_persons": [],
        "kept_persons": [],
    }

    filtered_bboxes = []

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

        person_info = {
            "person_id": bbox_info["person_id"],
            "bbox": bbox,
            "variance": variance,
            "is_uniform": is_uniform,
        }

        if is_uniform:
            stats["removed_bboxes"] += 1
            stats["removed_persons"].append(person_info)
        else:
            stats["kept_bboxes"] += 1
            stats["kept_persons"].append(person_info)
            filtered_bboxes.append(bbox)

    # Skip if no bboxes to remove
    if stats["removed_bboxes"] == 0:
        return None, None, None

    # Create visualization
    img_before = img.copy()
    img_after = img.copy()

    # Draw all bboxes on "before"
    for person_info in stats["removed_persons"] + stats["kept_persons"]:
        bbox = person_info["bbox"]
        person_id = person_info["person_id"]

        cv2.rectangle(
            img_before,
            (bbox["xmin"], bbox["ymin"]),
            (bbox["xmax"], bbox["ymax"]),
            (0, 255, 0),
            2,
        )

        label = f"P{person_id}"
        cv2.putText(
            img_before,
            label,
            (bbox["xmin"], bbox["ymin"] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Draw kept (green) and removed (red) on "after"
    for person_info in stats["kept_persons"]:
        bbox = person_info["bbox"]
        person_id = person_info["person_id"]

        cv2.rectangle(
            img_after,
            (bbox["xmin"], bbox["ymin"]),
            (bbox["xmax"], bbox["ymax"]),
            (0, 255, 0),
            2,
        )

        label = f"P{person_id}"
        cv2.putText(
            img_after,
            label,
            (bbox["xmin"], bbox["ymin"] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    for person_info in stats["removed_persons"]:
        bbox = person_info["bbox"]
        person_id = person_info["person_id"]
        variance = person_info["variance"]

        cv2.rectangle(
            img_after,
            (bbox["xmin"], bbox["ymin"]),
            (bbox["xmax"], bbox["ymax"]),
            (0, 0, 255),
            2,
        )

        label = f"P{person_id} (v={variance:.0f})"
        cv2.putText(
            img_after,
            label,
            (bbox["xmin"], bbox["ymin"] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    # Show visualization
    img_before_rgb = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)
    img_after_rgb = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    axes[0].imshow(img_before_rgb)
    axes[0].set_title(
        f'BEFORE - All BBoxes (Total: {stats["total_bboxes"]})',
        fontsize=14,
        fontweight="bold",
    )
    axes[0].axis("off")

    axes[1].imshow(img_after_rgb)
    axes[1].set_title(
        f"AFTER - Removed Uniform BBoxes\n"
        f'Green: Kept ({stats["kept_bboxes"]}) | Red: Removed ({stats["removed_bboxes"]})',
        fontsize=14,
        fontweight="bold",
    )
    axes[1].axis("off")

    plt.suptitle(
        f"Drone {drone_id} - Frame {frame_num:04d}\n"
        f"Press 'Y' to save, 'N' to skip, 'Q' to quit",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    # Don't show yet - return the figure for keyboard handling
    return stats, filtered_bboxes, fig


def save_filtered_labels(drone_id, frame_num, filtered_bboxes, labels_dir):
    """
    Save filtered bboxes to labels directory in YOLO format.

    Args:
        drone_id: Drone ID
        frame_num: Frame number
        filtered_bboxes: List of bbox dicts with xmin, ymin, xmax, ymax
        labels_dir: Path to labels directory
    """
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    label_path = labels_dir / f"{frame_num:04d}.txt"

    # Get image dimensions for YOLO normalization
    dataset_root = Path(__file__).parent
    img_path = dataset_root / f"D{drone_id}" / f"{frame_num:04d}.png"
    img = cv2.imread(str(img_path))
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


def batch_process_interactive(variance_threshold=100, start_drone=1, end_drone=8):
    """
    Process all frames for all drones interactively.

    Args:
        variance_threshold: Variance threshold for filtering
        start_drone: Starting drone ID
        end_drone: Ending drone ID
    """
    dataset_root = Path(__file__).parent

    print("=" * 70)
    print("Batch Interactive Static BBox Remover")
    print("=" * 70)
    print(f"Variance Threshold: {variance_threshold}")
    print(f"Processing Drones: {start_drone}-{end_drone}")
    print("=" * 70)
    print()

    stats_total = {
        "frames_processed": 0,
        "frames_saved": 0,
        "frames_skipped": 0,
        "bboxes_removed": 0,
    }

    for drone_id in range(start_drone, end_drone + 1):
        print(f"\n{'='*70}")
        print(f"Processing Drone {drone_id}")
        print(f"{'='*70}\n")

        # Get all frames for this drone
        images_dir = dataset_root / f"D{drone_id}"
        if not images_dir.exists():
            print(f"Skipping Drone {drone_id} - directory not found")
            continue

        frame_files = sorted(images_dir.glob("*.png"))
        labels_dir = dataset_root / f"D{drone_id}" / "labels"

        for img_file in tqdm(frame_files, desc=f"Drone {drone_id}"):
            frame_num = int(img_file.stem)

            # Analyze and visualize
            result = analyze_and_visualize_frame(
                drone_id, frame_num, variance_threshold
            )

            # Skip if no changes or error
            if result[0] is None:
                stats_total["frames_skipped"] += 1
                plt.close("all")
                continue

            stats, filtered_annotations, fig = result
            stats_total["frames_processed"] += 1

            # Show stats
            print(f"\nDrone {drone_id} - Frame {frame_num:04d}:")
            print(
                f"  Total: {stats['total_bboxes']} | Kept: {stats['kept_bboxes']} | Removed: {stats['removed_bboxes']}"
            )
            print("Press Y (sabboxeip), or Q (quit) on the visualization window...")

            # Handle keyboard input on figure
            user_choice = {"response": None}

            def on_key(event):
                if event.key in ["y", "Y", "n", "N", "q", "Q"]:
                    user_choice["response"] = event.key.lower()
                    plt.close(fig)

            fig.canvas.mpl_connect("key_press_event", on_key)
            plt.show()

            response = user_choice["response"]

            if response == "q":
                print("\n✗ Quitting...")
                print(f"\nTotal Stats:")
                print(f"  Frames processed: {stats_total['frames_processed']}")
                print(f"  Frames saved: {stats_total['frames_saved']}")
                print(f"  Frames skipped: {stats_total['frames_skipped']}")
                print(f"  Total bboxes removed: {stats_total['bboxes_removed']}")
                return

            elif response == "y":
                # Save filtered labels
                save_filtered_labels(
                    drone_id, frame_num, filtered_annotations, labels_dir
                )
                stats_total["frames_saved"] += 1
                stats_total["bboxes_removed"] += stats["removed_bboxes"]
                print(f"✓ Saved to {labels_dir / f'{frame_num:04d}.txt'}")

            elif response == "n":
                print("⊘ Skipped - no changes saved")

            else:
                # No key pressed, treat as skip
                print("⊘ No input - skipped")

            plt.close("all")

    print(f"\n{'='*70}")
    print("Batch Processing Complete!")
    print(f"{'='*70}")
    print(f"Frames processed: {stats_total['frames_processed']}")
    print(f"Frames saved: {stats_total['frames_saved']}")
    print(f"Frames skipped: {stats_total['frames_skipped']}")
    print(f"Total bboxes removed: {stats_total['bboxes_removed']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import sys

    print()
    print("=" * 70)
    print("Interactive Batch Static BBox Remover")
    print("=" * 70)
    print()
    print("This will go through all frames and let you approve/reject filtering.")
    print()

    # Get parameters
    if len(sys.argv) >= 2:
        variance_threshold = int(sys.argv[1])
        start_drone = int(sys.argv[2]) if len(sys.argv) >= 3 else 1
        end_drone = int(sys.argv[3]) if len(sys.argv) >= 4 else 8
    else:
        variance_threshold = int(input("Variance Threshold (default 100): ") or "100")
        start_drone = int(input("Start Drone (1-8, default 1): ") or "1")
        end_drone = int(input("End Drone (1-8, default 8): ") or "8")

    print()
    print(f"Configuration:")
    print(f"  Variance Threshold: {variance_threshold}")
    print(f"  Drones: {start_drone}-{end_drone}")
    print()
    input("Press Enter to start...")
    print()

    batch_process_interactive(variance_threshold, start_drone, end_drone)
