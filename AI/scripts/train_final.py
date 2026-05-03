"""
train_final.py - Final Production Training

Purpose: Train the final model using best hyperparameters from Optuna.
- Uses trial 1's params (best mAP50-95: 0.6576)
- Longer training (100 epochs)
- Saves production-ready model

Usage:
    python train_final.py

Output: models/trained/final/weights/best.pt
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import yaml
from ultralytics import YOLO

# MLflow tracking
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "yolo-matrix"


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).parent
AI_DIR = SCRIPT_DIR.parent
CONFIG_PATH = AI_DIR / "configs" / "training_config.yaml"
DATASET_YAML = AI_DIR / "datasets" / "MATRIX_yolo_format" / "MATRIX.yaml"

# Best params from Optuna trial 1
BEST_PARAMS = {
    "lr0": 1.0715146078908925e-05,
    "lrf": 0.013530745243410571,
    "freeze": 7,
    "batch": 16,
}


def load_config():
    """Load training configuration from YAML."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_final_training():
    """Run final production training."""
    print("\n" + "=" * 60)
    print("FINAL TRAINING - MATRIX DATASET")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load config
    config = load_config()
    final_cfg = config["final"]
    hardware_cfg = config["hardware"]
    aug_cfg = config["augmentation"]

    print("\nUsing Optuna trial 1 params:")
    print(f"  lr0: {BEST_PARAMS['lr0']:.10f}")
    print(f"  lrf: {BEST_PARAMS['lrf']:.4f}")
    print(f"  freeze: {BEST_PARAMS['freeze']}")
    print(f"  batch: {BEST_PARAMS['batch']}")
    print(f"\nTraining config:")
    print(f"  Epochs: {final_cfg['epochs']}")
    print(f"  Patience: {final_cfg['patience']}")
    print(f"  Device: {hardware_cfg['device']}")

    # Load model
    model_size = config["model"]["size"]
    pretrained_path = AI_DIR / "models" / "pretrained" / f"{model_size}.pt"

    if pretrained_path.exists():
        print(f"\nLoading model: {pretrained_path}")
        model = YOLO(str(pretrained_path))
    else:
        print(f"\nDownloading {model_size}.pt...")
        model = YOLO(f"{model_size}.pt")

    # Change to AI directory
    os.chdir(AI_DIR)

    # Run training
    print("\n" + "-" * 60)
    print("Starting training...")
    print("-" * 60 + "\n")

    results = model.train(
        # Dataset
        data=str(DATASET_YAML),
        # Training params (from Optuna)
        epochs=final_cfg["epochs"],
        batch=BEST_PARAMS["batch"],
        imgsz=640,
        patience=final_cfg["patience"],
        lr0=BEST_PARAMS["lr0"],
        lrf=BEST_PARAMS["lrf"],
        freeze=BEST_PARAMS["freeze"],
        # Hardware
        device=hardware_cfg["device"],
        workers=hardware_cfg["workers"],
        amp=hardware_cfg["amp"],
        # Augmentation
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
        name="final",
        exist_ok=True,
        plots=True,
        verbose=True,
        seed=config.get("seed", 42),
    )

    # Print results
    print("\n" + "=" * 60)
    print("FINAL TRAINING COMPLETE")
    print("=" * 60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        print("\nFinal Metrics:")
        for key in ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"]:
            val = metrics.get(key, "N/A")
            name = key.split("/")[1].replace("(B)", "")
            if isinstance(val, float):
                print(f"  {name}: {val:.4f}")
            else:
                print(f"  {name}: {val}")

    print(f"\nOutput saved to: {AI_DIR / 'models' / 'trained' / 'final'}")
    print(f"Best model: {AI_DIR / 'models' / 'trained' / 'final' / 'weights' / 'best.pt'}")

    return results


if __name__ == "__main__":
    try:
        run_final_training()
        print("\n✓ Final training completed successfully!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)