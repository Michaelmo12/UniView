"""
train_final_nano.py - Final Production Training (YOLOv11n)

Purpose: Train final nano model using best hyperparameters from Optuna nano search.
- Reads best params from configs/best_params_nano.yaml (written by train_optuna_nano.py)
- 100 epochs with early stopping
- Saves production-ready model

Usage:
    python train_final_nano.py

Output: models/trained/final_nano/weights/best.pt
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import yaml
from ultralytics import YOLO
from ultralytics.utils import callbacks as ultralytics_callbacks

# Remove MLflow callback
for event in list(ultralytics_callbacks.default_callbacks.keys()):
    ultralytics_callbacks.default_callbacks[event] = [
        cb for cb in ultralytics_callbacks.default_callbacks[event]
        if "mlflow" not in getattr(cb, "__module__", "")
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).parent
AI_DIR = SCRIPT_DIR.parent
CONFIG_PATH = AI_DIR / "configs" / "training_config.yaml"
DATASET_YAML = AI_DIR / "datasets" / "MATRIX_yolo_format" / "MATRIX.yaml"
BEST_PARAMS_PATH = AI_DIR / "configs" / "best_params_nano.yaml"

MODEL_SIZE = "yolo11n"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_best_params():
    if not BEST_PARAMS_PATH.exists():
        print(f"[ERROR] {BEST_PARAMS_PATH} not found.")
        print("  Run train_optuna_nano.py first, or check configs/trial_results_nano.yaml")
        sys.exit(1)
    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["params"], data


def run_final_training():
    print("\n" + "=" * 60)
    print(f"FINAL TRAINING — {MODEL_SIZE.upper()}")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config = load_config()
    final_cfg = config["final"]
    hardware_cfg = config["hardware"]
    aug_cfg = config["augmentation"]

    best_params, meta = load_best_params()

    print(f"\nBest params from Optuna (trial #{meta['best_trial']}, mAP50-95={meta['best_map50_95']:.4f}):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"\nTraining config:")
    print(f"  Epochs:   {final_cfg['epochs']}")
    print(f"  Patience: {final_cfg['patience']}")
    print(f"  Device:   {hardware_cfg['device']}")

    # Load model
    pretrained_path = AI_DIR / "models" / "pretrained" / f"{MODEL_SIZE}.pt"
    if pretrained_path.exists():
        print(f"\nLoading: {pretrained_path}")
        model = YOLO(str(pretrained_path))
    else:
        print(f"\nDownloading {MODEL_SIZE}.pt...")
        model = YOLO(f"{MODEL_SIZE}.pt")

    os.chdir(AI_DIR)

    print("\n" + "-" * 60)
    print("Starting final training...")
    print("-" * 60 + "\n")

    results = model.train(
        data=str(DATASET_YAML),
        epochs=final_cfg["epochs"],
        batch=best_params["batch"],
        imgsz=640,
        patience=final_cfg["patience"],
        lr0=best_params["lr0"],
        lrf=best_params["lrf"],
        freeze=best_params["freeze"],
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
        name="final_nano",
        exist_ok=True,
        plots=True,
        verbose=True,
        seed=config.get("seed", 42),
    )

    print("\n" + "=" * 60)
    print("FINAL TRAINING COMPLETE")
    print("=" * 60)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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

    out_dir = AI_DIR / "models" / "trained" / "final_nano"
    print(f"\nOutput: {out_dir}")
    print(f"Best model: {out_dir / 'weights' / 'best.pt'}")

    return results


if __name__ == "__main__":
    try:
        run_final_training()
        print("\n[OK] Final nano training completed!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
