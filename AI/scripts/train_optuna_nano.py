"""
train_optuna_nano.py - Hyperparameter Search with Optuna (YOLOv11n)

Purpose: Find optimal hyperparameters for YOLOv11 NANO on MATRIX dataset.
- Uses Optuna with SQLite storage so it RESUMES if interrupted
- Saves trial_results.yaml after every completed trial (safe to stop anytime)
- Searches: learning rate, freeze layers, batch size, final lr

Usage:
    python train_optuna_nano.py

Output:
    - models/trained/optuna_nano/trial_N/     (each trial's weights + logs)
    - configs/trial_results_nano.yaml         (updated after every trial)
    - configs/best_params_nano.yaml           (best params at the end)
    - optuna_nano.db                          (SQLite study — enables resume)
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Neutralize MLflow env vars that may leak from other projects
os.environ.pop("MLFLOW_TRACKING_URI", None)
os.environ.pop("MLFLOW_EXPERIMENT_NAME", None)
os.environ.pop("MLFLOW_RUN_ID", None)

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
BEST_PARAMS_PATH = AI_DIR / "configs" / "best_params_nano.yaml"
TRIAL_RESULTS_PATH = AI_DIR / "configs" / "trial_results_nano.yaml"
OUTPUT_DIR = AI_DIR / "models" / "trained" / "optuna_nano"
DB_PATH = AI_DIR / "optuna_nano.db"

MODEL_SIZE = "yolo11n"
STUDY_NAME = "yolo_matrix_nano_optimization"

# yolo11n has fewer layers than yolo11s — keep freeze range smaller
FREEZE_MAX = 10


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_trial_results(study):
    """Save all completed trial results to YAML. Called after every trial."""
    trials_data = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            trials_data.append({
                "trial": t.number,
                "map50_95": round(float(t.value), 6),
                "params": {k: (round(v, 8) if isinstance(v, float) else v) for k, v in t.params.items()},
                "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
            })
        elif t.state == optuna.trial.TrialState.PRUNED:
            trials_data.append({
                "trial": t.number,
                "map50_95": None,
                "state": "PRUNED",
                "params": t.params,
            })

    # Sort by map50_95 descending
    complete = [t for t in trials_data if t.get("map50_95") is not None]
    complete.sort(key=lambda x: x["map50_95"], reverse=True)

    best = complete[0] if complete else None

    output = {
        "model": MODEL_SIZE,
        "study_name": STUDY_NAME,
        "last_updated": datetime.now().isoformat(),
        "n_complete": len(complete),
        "best_trial": best,
        "all_trials": trials_data,
    }

    with open(TRIAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)

    print(f"  [saved] trial_results_nano.yaml — {len(complete)} complete trial(s)")


def objective(trial, config):
    """Optuna objective — train model, save results immediately, return mAP."""

    optuna_cfg = config["optuna"]
    hardware_cfg = config["hardware"]
    aug_cfg = config["augmentation"]
    search_space = optuna_cfg["search_space"]

    # Sample hyperparameters
    lr0 = trial.suggest_float("lr0", search_space["lr0"][0], search_space["lr0"][1], log=True)
    freeze = trial.suggest_int("freeze", 0, FREEZE_MAX)
    batch = trial.suggest_categorical("batch", search_space["batch"])
    lrf = trial.suggest_float("lrf", search_space["lrf"][0], search_space["lrf"][1], log=True)

    print(f"\n{'='*60}")
    print(f"TRIAL {trial.number}  [{MODEL_SIZE}]")
    print(f"{'='*60}")
    print(f"  lr0:    {lr0:.2e}")
    print(f"  lrf:    {lrf:.4f}")
    print(f"  freeze: {freeze}")
    print(f"  batch:  {batch}")

    # Load nano model
    pretrained_path = AI_DIR / "models" / "pretrained" / f"{MODEL_SIZE}.pt"
    if pretrained_path.exists():
        model = YOLO(str(pretrained_path))
    else:
        print(f"  Downloading {MODEL_SIZE}.pt from Ultralytics...")
        model = YOLO(f"{MODEL_SIZE}.pt")

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
            project=str(OUTPUT_DIR),
            name=f"trial_{trial.number}",
            exist_ok=True,
            plots=False,
            verbose=False,
            seed=config.get("seed", 42),
        )

        metrics = results.results_dict
        map50_95 = metrics.get("metrics/mAP50-95(B)", 0)
        map50 = metrics.get("metrics/mAP50(B)", 0)

        print(f"  Result: mAP50-95 = {map50_95:.4f}  |  mAP50 = {map50:.4f}")
        return map50_95

    except Exception as e:
        print(f"  Trial {trial.number} failed: {e}")
        return 0.0


def run_optuna_search():
    print("\n" + "=" * 60)
    print(f"OPTUNA HYPERPARAMETER SEARCH — {MODEL_SIZE.upper()}")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config = load_config()
    optuna_cfg = config["optuna"]

    print(f"\n  Trials:           {optuna_cfg['n_trials']}")
    print(f"  Epochs per trial: {optuna_cfg['epochs_per_trial']}")
    print(f"  Freeze range:     [0, {FREEZE_MAX}]")
    print(f"  Study DB:         {DB_PATH}")
    print(f"  Output dir:       {OUTPUT_DIR}")
    print(f"\n  Study will RESUME automatically if it already exists.\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # SQLite storage — study persists across restarts
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{DB_PATH}",
        engine_kwargs={"connect_args": {"timeout": 30}},
    )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
        load_if_exists=True,  # <-- resume if interrupted
    )

    already_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if already_done:
        print(f"  Resuming: {already_done} trial(s) already complete.")

    def after_trial_callback(study, trial):
        """Save results to YAML immediately after each trial finishes."""
        save_trial_results(study)

    study.optimize(
        lambda trial: objective(trial, config),
        n_trials=optuna_cfg["n_trials"],
        timeout=optuna_cfg["timeout"],
        show_progress_bar=True,
        callbacks=[after_trial_callback],
    )

    # Final summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\nCompleted trials: {len(complete_trials)}")

    if complete_trials:
        print(f"Best trial:       #{study.best_trial.number}")
        print(f"Best mAP50-95:    {study.best_value:.4f}")
        print("\nBest hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        best_params = {
            "model": MODEL_SIZE,
            "best_trial": study.best_trial.number,
            "best_map50_95": float(study.best_value),
            "params": study.best_params,
            "timestamp": datetime.now().isoformat(),
        }

        with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
            yaml.dump(best_params, f, default_flow_style=False)

        print(f"\nBest params saved to: {BEST_PARAMS_PATH}")
        print(f"All results saved to: {TRIAL_RESULTS_PATH}")
        print(f"\nRun final training with:")
        print(f"  python train_final_nano.py")

    return study


if __name__ == "__main__":
    try:
        run_optuna_search()
        print("\n[OK] Optuna nano search completed!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nSearch interrupted. Results saved. Resume by running the script again.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
