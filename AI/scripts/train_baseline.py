"""
train_baseline.py - Baseline Training Script for MATRIX Dataset

Purpose: Initial verification that everything works before running full training.
- Short training (10 epochs)
- Default parameters
- Validates dataset is properly formatted

Usage:
    python train_baseline.py

Output: models/trained/baseline/
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import yaml
from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Paths (relative to AI folder)
SCRIPT_DIR = Path(__file__).parent
AI_DIR = SCRIPT_DIR.parent
CONFIG_PATH = AI_DIR / "configs" / "training_config.yaml"
DATASET_YAML = AI_DIR / "datasets" / "MATRIX_yolo_format" / "MATRIX.yaml"


def load_config():
    """Load training configuration from YAML."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def verify_dataset(dataset_yaml: Path) -> dict:
    """Verify dataset exists and return stats."""
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    with open(dataset_yaml, "r", encoding="utf-8") as f:
        ds_config = yaml.safe_load(f)

    dataset_root = AI_DIR / ds_config["path"]

    stats = {
        "train_images": 0,
        "train_labels": 0,
        "val_images": 0,
        "val_labels": 0,
        "test_images": 0,
        "test_labels": 0,
    }

    # Standard YOLO structure: images/train, labels/train, etc.
    for split in ["train", "val", "test"]:
        split_path = ds_config.get(split, "")
        if not split_path:
            continue

        images_path = dataset_root / split_path
        labels_path = dataset_root / split_path.replace("images", "labels")

        if images_path.exists():
            img_count = len(list(images_path.glob("*.png"))) + len(
                list(images_path.glob("*.jpg"))
            )
            stats[f"{split}_images"] = img_count

        if labels_path.exists():
            lbl_count = len(list(labels_path.glob("*.txt")))
            stats[f"{split}_labels"] = lbl_count

    print(f"Dataset root: {dataset_root}")
    print(f"Classes: {ds_config.get('nc', 'N/A')} - {ds_config.get('names', {})}")
    print("\nSplit statistics:")
    print(f"  Train: {stats['train_images']} images, {stats['train_labels']} labels")
    print(f"  Val:   {stats['val_images']} images, {stats['val_labels']} labels")
    print(f"  Test:  {stats['test_images']} images, {stats['test_labels']} labels")

    # Validate
    if stats["train_images"] == 0:
        raise ValueError("No training images found! Run prepare_dataset.py first.")
    if stats["val_images"] == 0:
        raise ValueError("No validation images found! Run prepare_dataset.py first.")
    if stats["train_images"] != stats["train_labels"]:
        print("  WARNING: Train image/label mismatch!")
    if stats["val_images"] != stats["val_labels"]:
        print("  WARNING: Val image/label mismatch!")

    print("\n✓ Dataset verification passed!")
    return stats


def run_baseline_training():
    """Run baseline training."""
    print("\n" + "=" * 60)
    print("BASELINE TRAINING - MATRIX DATASET")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load config
    config = load_config()
    baseline_cfg = config["baseline"]
    hardware_cfg = config["hardware"]
    aug_cfg = config["augmentation"]

    print("\nConfiguration:")
    print(f"  Epochs: {baseline_cfg['epochs']}")
    print(f"  Batch size: {baseline_cfg['batch']}")
    print(f"  Image size: {baseline_cfg['imgsz']}")
    print(f"  Learning rate: {baseline_cfg['lr0']}")
    print(f"  Freeze layers: {baseline_cfg['freeze']}")
    print(f"  Device: {hardware_cfg['device']}")

    # Verify dataset
    verify_dataset(DATASET_YAML)

    # Initialize model
    print("\n" + "-" * 60)
    print("Loading model...")

    model_size = config["model"]["size"]
    pretrained_dir = AI_DIR / "models" / "pretrained"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    pretrained_path = pretrained_dir / f"{model_size}.pt"

    if pretrained_path.exists():
        print(f"Using pretrained: {pretrained_path}")
        model = YOLO(str(pretrained_path))
    else:
        print(f"Downloading {model_size}.pt to {pretrained_dir}...")
        model = YOLO(f"{model_size}.pt")
        # Move downloaded model to pretrained folder
        default_path = Path(f"{model_size}.pt")
        if default_path.exists():
            default_path.rename(pretrained_path)
            print(f"Saved to: {pretrained_path}")

    # Run training
    print("\n" + "-" * 60)
    print("Starting training...")
    print("-" * 60 + "\n")

    # Change to AI directory so relative paths in YAML work
    os.chdir(AI_DIR)

    results = model.train(
        # Dataset
        data=str(DATASET_YAML),
        # Training params
        epochs=baseline_cfg["epochs"],
        batch=baseline_cfg["batch"],
        imgsz=baseline_cfg["imgsz"],
        patience=baseline_cfg["patience"],
        # Learning rate
        lr0=baseline_cfg["lr0"],
        # Freeze backbone layers
        freeze=baseline_cfg["freeze"],
        # Hardware
        device=hardware_cfg["device"],
        workers=hardware_cfg["workers"],
        amp=hardware_cfg["amp"],
        # Augmentation (from config)
        hsv_h=aug_cfg["hsv_h"],
        hsv_s=aug_cfg["hsv_s"],
        hsv_v=aug_cfg["hsv_v"],
        degrees=aug_cfg["degrees"],
        translate=aug_cfg["translate"],
        scale=aug_cfg["scale"],
        shear=aug_cfg["shear"],
        perspective=aug_cfg["perspective"],
        flipud=aug_cfg["flipud"],
        fliplr=aug_cfg["fliplr"],
        mosaic=aug_cfg["mosaic"],
        mixup=aug_cfg["mixup"],
        # Output
        project=str(AI_DIR / "models" / "trained"),
        name="baseline",
        exist_ok=True,
        plots=True,
        verbose=True,
        # Reproducibility
        seed=config.get("seed", 42),
    )

    # Print results
    print("\n" + "=" * 60)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Display metrics
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        print("\nFinal Metrics:")
        print(
            f"  mAP50:      {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}"
            if isinstance(metrics.get("metrics/mAP50(B)"), float)
            else f"  mAP50:      {metrics.get('metrics/mAP50(B)', 'N/A')}"
        )
        print(
            f"  mAP50-95:   {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}"
            if isinstance(metrics.get("metrics/mAP50-95(B)"), float)
            else f"  mAP50-95:   {metrics.get('metrics/mAP50-95(B)', 'N/A')}"
        )
        print(
            f"  Precision:  {metrics.get('metrics/precision(B)', 'N/A'):.4f}"
            if isinstance(metrics.get("metrics/precision(B)"), float)
            else f"  Precision:  {metrics.get('metrics/precision(B)', 'N/A')}"
        )
        print(
            f"  Recall:     {metrics.get('metrics/recall(B)', 'N/A'):.4f}"
            if isinstance(metrics.get("metrics/recall(B)"), float)
            else f"  Recall:     {metrics.get('metrics/recall(B)', 'N/A')}"
        )

    print(f"\nOutput saved to: {AI_DIR / 'models' / 'trained' / 'baseline'}")
    print(
        f"Best model: {AI_DIR / 'models' / 'trained' / 'baseline' / 'weights' / 'best.pt'}"
    )

    return results


if __name__ == "__main__":
    try:
        run_baseline_training()
        print("\n✓ Baseline training completed successfully!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
