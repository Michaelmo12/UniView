"""
prepare_dataset.py - Reorganize MATRIX dataset to standard YOLO format

Converts:
    D1/0000.png + D1/labels/0000.txt
To:
    images/train/D1_0000.png + labels/train/D1_0000.txt

Usage:
    python prepare_dataset.py
"""

import shutil
from pathlib import Path

# Config
DATASET_ROOT = Path(__file__).parent.parent / "datasets" / "MATRIX_yolo_format"

TRAIN_FOLDERS = ["D1", "D2", "D3", "D4", "D5", "D6"]
VAL_FOLDERS = ["D7"]
TEST_FOLDERS = ["D8"]


def prepare_split(folders: list, split_name: str):
    """Copy images and labels from source folders to YOLO structure."""
    images_dir = DATASET_ROOT / "images" / split_name
    labels_dir = DATASET_ROOT / "labels" / split_name

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for folder in folders:
        src_folder = DATASET_ROOT / folder
        src_labels = src_folder / "labels"

        if not src_folder.exists():
            print(f"  WARNING: {folder} not found, skipping")
            continue

        images = list(src_folder.glob("*.png")) + list(src_folder.glob("*.jpg"))

        for img_path in images:
            base = img_path.stem
            new_name = f"{folder}_{base}"

            # Copy image
            dst_img = images_dir / f"{new_name}{img_path.suffix}"
            shutil.copy2(img_path, dst_img)

            # Copy label
            label_path = src_labels / f"{base}.txt"
            if label_path.exists():
                dst_label = labels_dir / f"{new_name}.txt"
                shutil.copy2(label_path, dst_label)

            total += 1

        print(f"  {folder}: {len(images)} images copied")

    return total


def main():
    print("="*50)
    print("MATRIX Dataset Preparation")
    print("="*50)
    print(f"Source: {DATASET_ROOT}")
    print()

    # Clean up previous runs
    for split in ["train", "val", "test"]:
        img_dir = DATASET_ROOT / "images" / split
        lbl_dir = DATASET_ROOT / "labels" / split
        if img_dir.exists():
            shutil.rmtree(img_dir)
        if lbl_dir.exists():
            shutil.rmtree(lbl_dir)

    print("Preparing TRAIN split...")
    train_count = prepare_split(TRAIN_FOLDERS, "train")

    print("\nPreparing VAL split...")
    val_count = prepare_split(VAL_FOLDERS, "val")

    print("\nPreparing TEST split...")
    test_count = prepare_split(TEST_FOLDERS, "test")

    print()
    print("="*50)
    print("DONE!")
    print(f"  Train: {train_count} images")
    print(f"  Val:   {val_count} images")
    print(f"  Test:  {test_count} images")
    print("="*50)


if __name__ == "__main__":
    main()