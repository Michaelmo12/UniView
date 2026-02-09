"""
train_optuna.py - Hyperparameter Search with Optuna

Purpose: Find optimal hyperparameters for YOLO training on MATRIX dataset.
- Uses Optuna for Bayesian optimization
- Searches: learning rate, freeze layers, batch size, final lr
- Saves best parameters to config file

Usage:
    python train_optuna.py

Output:
    - Optuna study results
    - Best hyperparameters saved to configs/best_params.yaml
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import yaml
import optuna
from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).parent
AI_DIR = SCRIPT_DIR.parent
CONFIG_PATH = AI_DIR / "configs" / "training_config.yaml"
DATASET_YAML = AI_DIR / "datasets" / "MATRIX_yolo_format" / "MATRIX.yaml"
BEST_PARAMS_PATH = AI_DIR / "configs" / "best_params.yaml"


def load_config():
    """Load training configuration from YAML."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def objective(trial, config):
    """Optuna objective function - train model and return mAP."""

    optuna_cfg = config["optuna"]
    hardware_cfg = config["hardware"]
    aug_cfg = config["augmentation"]
    search_space = optuna_cfg["search_space"]

    # Sample hyperparameters
    lr0 = trial.suggest_float("lr0", search_space["lr0"][0], search_space["lr0"][1], log=True)
    freeze = trial.suggest_int("freeze", search_space["freeze"][0], search_space["freeze"][1])
    batch = trial.suggest_categorical("batch", search_space["batch"])
    lrf = trial.suggest_float("lrf", search_space["lrf"][0], search_space["lrf"][1], log=True)

    print(f"\n{'='*60}")
    print(f"TRIAL {trial.number}")
    print(f"{'='*60}")
    print(f"  lr0: {lr0:.6f}")
    print(f"  freeze: {freeze}")
    print(f"  batch: {batch}")
    print(f"  lrf: {lrf:.4f}")

    # Load model
    model_size = config["model"]["size"]
    pretrained_path = AI_DIR / "models" / "pretrained" / f"{model_size}.pt"

    if pretrained_path.exists():
        model = YOLO(str(pretrained_path))
    else:
        model = YOLO(f"{model_size}.pt")

    # Train
    os.chdir(AI_DIR)

    try:
        results = model.train(
            data=str(DATASET_YAML),
            epochs=optuna_cfg["epochs_per_trial"],
            batch=batch,
            imgsz=640,
            patience=10,
            lr0=lr0,
            lrf=lrf,
            freeze=freeze,
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
            project=str(AI_DIR / "models" / "trained" / "optuna"),
            name=f"trial_{trial.number}",
            exist_ok=True,
            plots=False,
            verbose=False,
            seed=config.get("seed", 42),
        )

        # Get mAP50-95 as the metric to optimize
        metrics = results.results_dict
        map50_95 = metrics.get("metrics/mAP50-95(B)", 0)

        print(f"  Result: mAP50-95 = {map50_95:.4f}")

        return map50_95

    except Exception as e:
        print(f"  Trial failed: {e}")
        return 0.0


def run_optuna_search():
    """Run Optuna hyperparameter search."""
    print("\n" + "=" * 60)
    print("OPTUNA HYPERPARAMETER SEARCH - MATRIX DATASET")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load config
    config = load_config()
    optuna_cfg = config["optuna"]

    print("\nSearch configuration:")
    print(f"  Trials: {optuna_cfg['n_trials']}")
    print(f"  Epochs per trial: {optuna_cfg['epochs_per_trial']}")
    print(f"  Timeout: {optuna_cfg['timeout']}s")
    print("\nSearch space:")
    for param, values in optuna_cfg["search_space"].items():
        print(f"  {param}: {values}")

    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name="yolo_matrix_optimization",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, config),
        n_trials=optuna_cfg["n_trials"],
        timeout=optuna_cfg["timeout"],
        show_progress_bar=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best mAP50-95: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save best params
    best_params = {
        "best_trial": study.best_trial.number,
        "best_map50_95": float(study.best_value),
        "params": study.best_params,
        "timestamp": datetime.now().isoformat(),
    }

    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        yaml.dump(best_params, f, default_flow_style=False)

    print(f"\nBest parameters saved to: {BEST_PARAMS_PATH}")

    return study


if __name__ == "__main__":
    try:
        run_optuna_search()
        print("\n✓ Optuna search completed successfully!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nSearch interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)