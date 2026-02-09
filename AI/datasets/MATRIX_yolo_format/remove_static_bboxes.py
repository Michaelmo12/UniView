"""
Remove Static/Uniform BBoxes from MATRIX Annotations

Removes bounding boxes that contain uniform/static content (like poles, pillars)
by analyzing pixel variance within each bbox. Low variance = likely not a person.
"""

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def calculate_bbox_variance(image, bbox):
    """
    Calculate the variance of pixels within a bounding box.
    Low variance indicates uniform/static content (like a pole).

    Args:
        image: OpenCV image (BGR)
        bbox: Dict with xmin, ymin, xmax, ymax

    Returns:
        float: Pixel variance (lower = more uniform)
    """
    xmin = bbox["xmin"]
    ymin = bbox["ymin"]
    xmax = bbox["xmax"]
    ymax = bbox["ymax"]

    # Extract bbox region
    roi = image[ymin:ymax, xmin:xmax]

    # Convert to grayscale for variance calculation
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Calculate variance
    variance = np.var(gray_roi)

    return variance


def is_bbox_uniform(image, bbox, variance_threshold=100):
    """
    Check if a bbox contains uniform/static content.

    Args:
        image: OpenCV image (BGR)
        bbox: Dict with xmin, ymin, xmax, ymax
        variance_threshold: Maximum variance for "uniform" (default: 100)

    Returns:
        bool: True if bbox is uniform (should be removed)
    """
    variance = calculate_bbox_variance(image, bbox)
    return variance < variance_threshold


def analyze_frame_bboxes(drone_id, frame_num, variance_threshold=100, visualize=True):
    """
    Analyze all bboxes in a frame and identify which are uniform/static.

    Args:
        drone_id: Drone ID (1-8)
        frame_num: Frame number
        variance_threshold: Variance threshold for filtering
        visualize: Show before/after comparison

    Returns:
        dict: Statistics about removed bboxes
    """
    # Paths
    dataset_root = Path(__file__).parent
    images_dir = dataset_root / "MATRIX_30x30" / "image_subsets" / f"D{drone_id}"
    annotations_dir = dataset_root / "MATRIX_30x30" / "annotations_positions"

    img_path = images_dir / f"{frame_num:04d}.png"
    json_path = annotations_dir / f"{frame_num:04d}.json"

    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Could not load image {img_path}")
        return None

    # Load annotations
    with open(json_path, "r") as f:
        annotations = json.load(f)

    # Get view number for this drone
    view_num = drone_id - 1

    # Analyze each person's bbox
    stats = {
        "total_bboxes": 0,
        "removed_bboxes": 0,
        "kept_bboxes": 0,
        "removed_persons": [],
        "kept_persons": [],
    }

    print(f"\n{'='*70}")
    print(f"Analyzing Frame {frame_num:04d} - Drone {drone_id}")
    print(f"Variance Threshold: {variance_threshold}")
    print(f"{'='*70}\n")

    for person in annotations:
        person_id = person["personID"]

        # Find view for this drone
        for view in person["views"]:
            if view["viewNum"] == view_num:
                xmin = view["xmin"]
                ymin = view["ymin"]
                xmax = view["xmax"]
                ymax = view["ymax"]

                # Skip if not visible
                if xmin == -1:
                    continue

                stats["total_bboxes"] += 1

                bbox = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}

                variance = calculate_bbox_variance(img, bbox)
                is_uniform = variance < variance_threshold

                person_info = {
                    "person_id": person_id,
                    "bbox": bbox,
                    "variance": variance,
                    "is_uniform": is_uniform,
                }

                if is_uniform:
                    stats["removed_bboxes"] += 1
                    stats["removed_persons"].append(person_info)
                    print(
                        f"❌ Person {person_id:3d} - Variance: {variance:8.2f} - REMOVED (uniform)"
                    )
                else:
                    stats["kept_bboxes"] += 1
                    stats["kept_persons"].append(person_info)
                    print(
                        f"✓  Person {person_id:3d} - Variance: {variance:8.2f} - KEPT"
                    )

                break

    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Total bboxes: {stats['total_bboxes']}")
    print(f"  Removed (uniform): {stats['removed_bboxes']}")
    print(f"  Kept (varied): {stats['kept_bboxes']}")
    print(f"{'='*70}\n")

    # Visualization
    if visualize:
        visualize_before_after(img, stats, frame_num, drone_id)

    return stats


def visualize_before_after(img, stats, frame_num, drone_id):
    """
    Show before/after comparison of bbox filtering.
    """
    # Create two copies for visualization
    img_before = img.copy()
    img_after = img.copy()

    # Draw all bboxes on "before" image
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

    # Draw only kept bboxes on "after" image (removed ones in red)
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

    # Draw removed bboxes in red
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

    # Convert to RGB for matplotlib
    img_before_rgb = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)
    img_after_rgb = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)

    # Display side-by-side
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
        f"Drone {drone_id} - Frame {frame_num:04d}", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


def save_filtered_annotations(
    drone_id, frame_num, variance_threshold=100, output_dir=None
):
    """
    Save filtered annotations (without uniform bboxes) to a new file.

    Args:
        drone_id: Drone ID (1-8)
        frame_num: Frame number
        variance_threshold: Variance threshold
        output_dir: Directory to save filtered annotations (None = create backup)
    """
    # Paths
    dataset_root = Path(__file__).parent
    images_dir = dataset_root / "MATRIX_30x30" / "image_subsets" / f"D{drone_id}"
    annotations_dir = dataset_root / "MATRIX_30x30" / "annotations_positions"

    img_path = images_dir / f"{frame_num:04d}.png"
    json_path = annotations_dir / f"{frame_num:04d}.json"

    # Load image and annotations
    img = cv2.imread(str(img_path))
    with open(json_path, "r") as f:
        annotations = json.load(f)

    view_num = drone_id - 1

    # Filter annotations
    filtered_annotations = []

    for person in annotations:
        # Find view for this drone
        for view in person["views"]:
            if view["viewNum"] == view_num:
                xmin = view["xmin"]
                ymin = view["ymin"]
                xmax = view["xmax"]
                ymax = view["ymax"]

                # Skip if not visible
                if xmin == -1:
                    filtered_annotations.append(person)
                    break

                bbox = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}

                # Keep only if variance is high enough
                if not is_bbox_uniform(img, bbox, variance_threshold):
                    filtered_annotations.append(person)

                break

    # Save filtered annotations
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{frame_num:04d}.json"
    else:
        # Create backup and overwrite
        backup_path = json_path.with_suffix(".json.backup")
        import shutil

        shutil.copy(json_path, backup_path)
        output_path = json_path
        print(f"Backup created: {backup_path}")

    with open(output_path, "w") as f:
        json.dump(filtered_annotations, f, indent=2)

    print(f"Filtered annotations saved to: {output_path}")
    print(f"Original: {len(annotations)} persons")
    print(f"Filtered: {len(filtered_annotations)} persons")
    print(f"Removed: {len(annotations) - len(filtered_annotations)} persons")


if __name__ == "__main__":
    import sys

    print()
    print("=" * 70)
    print("MATRIX Static/Uniform BBox Remover")
    print("=" * 70)
    print()
    print("This script removes bboxes with uniform/static content (like poles)")
    print("by analyzing pixel variance within each bbox.")
    print()

    # Get parameters
    if len(sys.argv) >= 3:
        drone_id = int(sys.argv[1])
        frame_num = int(sys.argv[2])
        variance_threshold = int(sys.argv[3]) if len(sys.argv) >= 4 else 100
    else:
        drone_id = int(input("Enter Drone ID (1-8): "))
        frame_num = int(input("Enter Frame Number: "))
        variance_threshold = int(
            input("Enter Variance Threshold (default 100, lower=stricter): ") or "100"
        )

    print()
    print(f"Configuration:")
    print(f"  Drone ID: {drone_id}")
    print(f"  Frame: {frame_num:04d}")
    print(f"  Variance Threshold: {variance_threshold}")
    print()

    # Analyze frame
    stats = analyze_frame_bboxes(
        drone_id, frame_num, variance_threshold, visualize=True
    )

    if stats:
        print()
        save = input("Save filtered annotations? (y/n): ").strip().lower()

        if save == "y":
            output_dir = input(
                "Output directory (press Enter to overwrite original with backup): "
            ).strip()
            output_dir = output_dir if output_dir else None

            save_filtered_annotations(
                drone_id, frame_num, variance_threshold, output_dir
            )
            print("\n✓ Filtering complete!")
        else:
            print("\nFiltering cancelled. No files were modified.")
