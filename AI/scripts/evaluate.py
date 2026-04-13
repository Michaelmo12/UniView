"""
evaluate.py - GT vs Detection evaluation on test set

Runs YOLO on the test set, compares predictions against ground truth labels,
and produces per-image and overall metrics + a visual comparison grid.

Usage:
    python evaluate.py                          # uses trial_5 best.pt by default
    python evaluate.py --weights path/to/best.pt
    python evaluate.py --weights final_nano     # shortcut for final_nano/weights/best.pt

Output:
    models/eval/
        results.csv         per-image TP/FP/FN/precision/recall
        summary.txt         overall metrics
        visuals/            side-by-side GT vs detection images (sample)
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.utils import callbacks as ultralytics_callbacks

# Disable MLflow
for event in list(ultralytics_callbacks.default_callbacks.keys()):
    ultralytics_callbacks.default_callbacks[event] = [
        cb for cb in ultralytics_callbacks.default_callbacks[event]
        if "mlflow" not in getattr(cb, "__module__", "")
    ]

SCRIPT_DIR = Path(__file__).parent
AI_DIR = SCRIPT_DIR.parent
TEST_IMAGES = AI_DIR / "datasets" / "MATRIX_yolo_format" / "images" / "test"
TEST_LABELS = AI_DIR / "datasets" / "MATRIX_yolo_format" / "labels" / "test"
DATASET_YAML = AI_DIR / "datasets" / "MATRIX_yolo_format" / "MATRIX.yaml"
EVAL_DIR = AI_DIR / "models" / "eval"

IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25


def iou(box_a, box_b):
    """Compute IoU between two boxes in xyxy format."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def load_gt(label_path, img_w, img_h):
    """Load YOLO format labels and convert to xyxy pixel coords."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes


def match_predictions(gt_boxes, pred_boxes, iou_thresh=IOU_THRESHOLD):
    """Match predictions to GT. Returns TP, FP, FN counts."""
    matched_gt = set()
    tp = 0
    fp = 0
    for pred in pred_boxes:
        best_iou = 0
        best_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            v = iou(pred, gt)
            if v > best_iou:
                best_iou = v
                best_idx = i
        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def draw_comparison(img, gt_boxes, pred_boxes, pred_confs):
    """Draw GT (green) and predictions (red) on image side by side."""
    gt_img = img.copy()
    pred_img = img.copy()

    for box in gt_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for box, conf in zip(pred_boxes, pred_confs):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(pred_img, f"{conf:.2f}", (x1, max(y1-4, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.putText(gt_img, "GT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(pred_img, "Pred", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return np.hstack([gt_img, pred_img])


def resolve_weights(arg):
    shortcuts = {
        "trial_5": AI_DIR / "models/trained/optuna_nano/trial_5/weights/best.pt",
        "final_nano": AI_DIR / "models/trained/final_nano/weights/best.pt",
        "final": AI_DIR / "models/trained/final/weights/best.pt",
        "small": Path("C:/Projects_H.W/FINAL-PROJECT/UniView/algorithm/weights/best.pt"),
    }
    if arg in shortcuts:
        return shortcuts[arg]
    return Path(arg)


def run_evaluation(weights_path, max_visuals=20):
    print("\n" + "=" * 60)
    print("EVALUATION — GT vs DETECTIONS")
    print("=" * 60)
    print(f"Weights: {weights_path}")
    print(f"Test images: {TEST_IMAGES}")
    print(f"IoU threshold: {IOU_THRESHOLD}")
    print(f"Conf threshold: {CONF_THRESHOLD}")

    model = YOLO(str(weights_path))

    images = sorted(TEST_IMAGES.glob("*.png")) + sorted(TEST_IMAGES.glob("*.jpg"))
    print(f"\nTest images found: {len(images)}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    visuals_dir = EVAL_DIR / "visuals"
    visuals_dir.mkdir(exist_ok=True)

    total_tp = total_fp = total_fn = 0
    rows = []
    visual_count = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        label_path = TEST_LABELS / (img_path.stem + ".txt")
        gt_boxes = load_gt(label_path, w, h)

        results = model.predict(str(img_path), conf=CONF_THRESHOLD, verbose=False)
        preds = results[0].boxes
        pred_boxes = []
        pred_confs = []
        if preds is not None and len(preds):
            pred_boxes = preds.xyxy.cpu().numpy().tolist()
            pred_confs = preds.conf.cpu().numpy().tolist()

        tp, fp, fn = match_predictions(gt_boxes, pred_boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

        rows.append({
            "image": img_path.name,
            "gt": len(gt_boxes),
            "pred": len(pred_boxes),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        })

        if visual_count < max_visuals:
            comp = draw_comparison(img, gt_boxes, pred_boxes, pred_confs)
            cv2.imwrite(str(visuals_dir / img_path.name), comp)
            visual_count += 1

    # Overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Save CSV
    csv_path = EVAL_DIR / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Save summary
    summary = f"""
EVALUATION SUMMARY
==================
Weights:   {weights_path}
Images:    {len(images)}
IoU@{IOU_THRESHOLD}

Total GT boxes:   {total_tp + total_fn}
Total Pred boxes: {total_tp + total_fp}

TP: {total_tp}
FP: {total_fp}
FN: {total_fn}

Precision: {precision:.4f}
Recall:    {recall:.4f}
F1:        {f1:.4f}

Per-image CSV: {csv_path}
Visuals:       {visuals_dir} ({visual_count} images)
"""
    print(summary)
    with open(EVAL_DIR / "summary.txt", "w") as f:
        f.write(summary)

    print(f"Results saved to: {EVAL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default="trial_5",
        help="Path to weights or shortcut: trial_5 | final_nano | final"
    )
    parser.add_argument("--visuals", type=int, default=20, help="Number of visual comparisons to save")
    args = parser.parse_args()

    weights = resolve_weights(args.weights)
    if not weights.exists():
        print(f"[ERROR] Weights not found: {weights}")
        sys.exit(1)

    run_evaluation(weights, max_visuals=args.visuals)
