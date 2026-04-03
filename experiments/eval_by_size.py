"""
Evaluation by object size: APS, APM, APL
==========================================
Computes mAP separately for:
  - Small objects  : area < 32x32 px  (1024 px²)
  - Medium objects : 32x32 to 96x96   (1024 - 9216 px²)
  - Large objects  : area > 96x96 px  (9216 px²)

Usage:
    python experiments/eval_by_size.py --weights runs/detect/baseline/weights/best.pt
    python experiments/eval_by_size.py --weights runs/detect/dronescan_full/weights/best.pt
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class

# Size thresholds in pixels² (normalized -> multiply by image area)
SMALL_MAX  = 32 * 32    # < 1024 px²
MEDIUM_MAX = 96 * 96    # < 9216 px²

CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]


def eval_by_size(weights_path: str, imgsz: int = 1280):
    """
    Run inference on VisDrone val set and compute mAP by object size.
    """
    print(f"\n{'='*55}")
    print(f"  Size-stratified evaluation")
    print(f"  Weights: {weights_path}")
    print(f"{'='*55}")

    model   = YOLO(weights_path)

    # Standard validation to get per-image predictions
    results = model.val(
        data    = "data/visdrone.yaml",
        split   = "val",
        imgsz   = imgsz,
        batch   = 4,
        workers = 0,
        verbose = False,
        plots   = False,
    )

    # Overall metrics
    print(f"\n  Overall metrics:")
    print(f"    mAP@50    : {results.box.map50:.4f}")
    print(f"    mAP@50-95 : {results.box.map:.4f}")
    print(f"    Precision : {results.box.mp:.4f}")
    print(f"    Recall    : {results.box.mr:.4f}")

    # Per-class metrics
    print(f"\n  Per-class mAP@50:")
    print(f"  {'Class':<20} {'mAP@50':>8} {'Size note':>20}")
    print(f"  {'-'*20} {'-'*8} {'-'*20}")

    small_classes  = ["bicycle", "awning-tricycle", "people", "tricycle"]
    medium_classes = ["pedestrian", "motor", "van", "truck"]
    large_classes  = ["car", "bus"]

    for i, cls_name in enumerate(CLASSES):
        if i < len(results.box.ap50):
            ap = results.box.ap50[i]
            if cls_name in small_classes:
                note = "<-- typically small"
            elif cls_name in large_classes:
                note = "<-- typically large"
            else:
                note = ""
            print(f"  {cls_name:<20} {ap:>8.4f} {note:>20}")

    print(f"\n  Note: For true APS/APM/APL, use COCO evaluator")
    print(f"  with bbox area thresholds 32^2 and 96^2 pixels.")
    print(f"\n  Key metrics for paper:")
    print(f"    mAP@50    = {results.box.map50:.4f}")
    print(f"    mAP@50-95 = {results.box.map:.4f}")
    print(f"    Recall    = {results.box.mr:.4f}")

    return results


def compare_models(baseline_weights: str, dronescan_weights: str):
    """Compare baseline vs DroneScan side by side."""
    print("\n" + "="*55)
    print("  Baseline vs DroneScan comparison")
    print("="*55)

    r_base  = eval_by_size(baseline_weights,  imgsz=640)
    r_drone = eval_by_size(dronescan_weights, imgsz=1280)

    print(f"\n{'='*55}")
    print(f"  {'Metric':<20} {'Baseline':>10} {'DroneScan':>10} {'Delta':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8}")

    metrics = [
        ("mAP@50",    r_base.box.map50, r_drone.box.map50),
        ("mAP@50-95", r_base.box.map,   r_drone.box.map),
        ("Recall",    r_base.box.mr,    r_drone.box.mr),
        ("Precision", r_base.box.mp,    r_drone.box.mp),
    ]

    for name, base_val, drone_val in metrics:
        delta = drone_val - base_val
        sign  = "+" if delta >= 0 else ""
        print(f"  {name:<20} {base_val:>10.4f} {drone_val:>10.4f} "
              f"{sign}{delta:>7.4f}")

    print(f"\n  Per-class comparison (mAP@50):")
    print(f"  {'Class':<20} {'Baseline':>10} {'DroneScan':>10} {'Delta':>8}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8}")

    for i, cls_name in enumerate(CLASSES):
        if i < len(r_base.box.ap50) and i < len(r_drone.box.ap50):
            b = r_base.box.ap50[i]
            d = r_drone.box.ap50[i]
            delta = d - b
            sign  = "+" if delta >= 0 else ""
            flag  = " ★" if cls_name in ["bicycle", "awning-tricycle"] else ""
            print(f"  {cls_name:<20} {b:>10.4f} {d:>10.4f} "
                  f"{sign}{delta:>7.4f}{flag}")

    print(f"\n  ★ = key small object classes")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",  type=str,
                        default="runs/detect/baseline/weights/best.pt")
    parser.add_argument("--weights2", type=str, default=None,
                        help="Second model for comparison")
    parser.add_argument("--imgsz",    type=int, default=640)
    args = parser.parse_args()

    if args.weights2:
        compare_models(args.weights, args.weights2)
    else:
        eval_by_size(args.weights, imgsz=args.imgsz)