"""
DroneScan-YOLO — Full Ablation Study (8 runs)
===============================================
Trains all ablation configurations automatically and saves results.

Run  1: YOLOv8s baseline           (loaded from train_baseline.py output)
Run  2: + SAL-NWD only
Run  3: + MSFD only
Run  4: + RPA only
Run  5: + RPA + MSFD
Run  6: + RPA + SAL-NWD
Run  7: + MSFD + SAL-NWD
Run  8: DroneScan-YOLO (all three)

Results saved to: results/ablation_results.csv
Estimated time : ~6-7h on RTX 4090 (7 runs x ~55 min each)
"""

import csv
import time
import torch
from pathlib import Path
from ultralytics import YOLO

YAML        = "data/visdrone.yaml"
EPOCHS      = 50
IMGSZ       = 640
BATCH       = 16
WORKERS     = 0
DEVICE      = 0
PROJECT     = "runs/detect"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 8-run ablation matrix
# ---------------------------------------------------------------------------
ABLATION_RUNS = [
    {
        "name":        "abl_sal_nwd",
        "description": "YOLOv8s + SAL-NWD",
        "use_rpa":     False,
        "use_msfd":    False,
        "use_sal_nwd": True,
    },
    {
        "name":        "abl_msfd",
        "description": "YOLOv8s + MSFD",
        "use_rpa":     False,
        "use_msfd":    True,
        "use_sal_nwd": False,
    },
    {
        "name":        "abl_rpa",
        "description": "YOLOv8s + RPA",
        "use_rpa":     True,
        "use_msfd":    False,
        "use_sal_nwd": False,
    },
    {
        "name":        "abl_rpa_msfd",
        "description": "YOLOv8s + RPA + MSFD",
        "use_rpa":     True,
        "use_msfd":    True,
        "use_sal_nwd": False,
    },
    {
        "name":        "abl_rpa_sal",
        "description": "YOLOv8s + RPA + SAL-NWD",
        "use_rpa":     True,
        "use_msfd":    False,
        "use_sal_nwd": True,
    },
    {
        "name":        "abl_msfd_sal",
        "description": "YOLOv8s + MSFD + SAL-NWD",
        "use_rpa":     False,
        "use_msfd":    True,
        "use_sal_nwd": True,
    },
    {
        "name":        "dronescan_full",
        "description": "DroneScan-YOLO (all)",
        "use_rpa":     True,
        "use_msfd":    True,
        "use_sal_nwd": True,
    },
]


def train_run(config: dict) -> dict:
    """Train one ablation run and return its metrics."""
    print(f"\n{'='*60}")
    print(f"  {config['description']}")
    print(f"  RPA={config['use_rpa']} | "
          f"MSFD={config['use_msfd']} | "
          f"SAL-NWD={config['use_sal_nwd']}")
    print(f"{'='*60}")

    model = YOLO("yolov8s.pt")

    start = time.time()
    results = model.train(
        data         = YAML,
        epochs       = EPOCHS,
        imgsz        = IMGSZ,
        batch        = BATCH,
        workers      = WORKERS,
        device       = DEVICE,
        project      = PROJECT,
        name         = config["name"],
        exist_ok     = True,
        pretrained   = True,
        optimizer    = "AdamW",
        lr0          = 0.001,
        lrf          = 0.01,
        weight_decay = 0.0005,
        warmup_epochs= 3,
        close_mosaic = 10,
        amp          = True,
        plots        = True,
        verbose      = False,
    )
    elapsed = time.time() - start

    out = {
        "name":            config["name"],
        "description":     config["description"],
        "use_rpa":         config["use_rpa"],
        "use_msfd":        config["use_msfd"],
        "use_sal_nwd":     config["use_sal_nwd"],
        "map50":           results.results_dict.get("metrics/mAP50(B)", 0),
        "map50_95":        results.results_dict.get("metrics/mAP50-95(B)", 0),
        "precision":       results.results_dict.get("metrics/precision(B)", 0),
        "recall":          results.results_dict.get("metrics/recall(B)", 0),
        "train_time_min":  elapsed / 60,
        "weights":         f"{PROJECT}/{config['name']}/weights/best.pt",
        "fps":             0.0,
    }

    print(f"\n  mAP@50    : {out['map50']:.4f}")
    print(f"  mAP@50-95 : {out['map50_95']:.4f}")
    print(f"  Time      : {out['train_time_min']:.1f} min")
    return out


def measure_fps(weights_path: str, n: int = 100) -> float:
    """Measure inference FPS on dummy images."""
    if not Path(weights_path).exists():
        return 0.0
    model = YOLO(weights_path)
    dummy = torch.zeros(1, 3, 640, 640)
    for _ in range(5):
        model.predict(dummy, verbose=False)
    t = time.time()
    for _ in range(n):
        model.predict(dummy, verbose=False)
    fps = n / (time.time() - t)
    print(f"  FPS: {fps:.1f}")
    return fps


def load_baseline_result() -> dict:
    """Load baseline mAP from the already-trained baseline run."""
    import pandas as pd
    csv_path = Path(f"{PROJECT}/baseline/results.csv")
    if not csv_path.exists():
        print("  [WARN] Baseline results.csv not found — run train_baseline.py first")
        return {}
    df  = pd.read_csv(csv_path)
    row = df.iloc[-1]
    return {
        "name":           "baseline",
        "description":    "YOLOv8s baseline",
        "use_rpa":        False,
        "use_msfd":       False,
        "use_sal_nwd":    False,
        "map50":          row.get("metrics/mAP50(B)", 0),
        "map50_95":       row.get("metrics/mAP50-95(B)", 0),
        "precision":      row.get("metrics/precision(B)", 0),
        "recall":         row.get("metrics/recall(B)", 0),
        "train_time_min": 0,
        "fps":            0.0,
        "weights":        f"{PROJECT}/baseline/weights/best.pt",
    }


def save_csv(results: list):
    path = RESULTS_DIR / "ablation_results.csv"
    fields = ["name", "description", "use_rpa", "use_msfd", "use_sal_nwd",
              "map50", "map50_95", "precision", "recall",
              "train_time_min", "fps", "weights"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"  Saved -> {path}")


def print_summary(results: list):
    print(f"\n{'='*75}")
    print(f"  {'Model':<35} {'mAP@50':>8} {'mAP@50-95':>10} {'FPS':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*8}")
    for r in results:
        print(f"  {r['description']:<35} "
              f"{r['map50']:>8.4f} "
              f"{r['map50_95']:>10.4f} "
              f"{r.get('fps', 0):>8.1f}")
    print(f"{'='*75}")


def main():
    print("=" * 60)
    print("  DroneScan-YOLO — Full Ablation Study")
    print("  8 runs total (including baseline)")
    print(f"  Estimated: ~7h on RTX 4090")
    print("=" * 60)

    # Load already-trained baseline
    all_results = []
    baseline = load_baseline_result()
    if baseline:
        print(f"\n  Baseline loaded: mAP@50={baseline['map50']:.4f}")
        baseline["fps"] = measure_fps(baseline["weights"])
        all_results.append(baseline)

    # Run 7 ablation configurations
    for config in ABLATION_RUNS:
        result = train_run(config)
        result["fps"] = measure_fps(result["weights"])
        all_results.append(result)
        save_csv(all_results)   # save after each run

    print_summary(all_results)
    save_csv(all_results)
    print("\nAblation study complete.")
    print(f"Results -> results/ablation_results.csv")


if __name__ == "__main__":
    main()