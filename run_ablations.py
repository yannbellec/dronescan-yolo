"""
DroneScan-YOLO — Ablation Study
=================================
Uses standard YOLO API with callbacks to inject custom modules.
Avoids Windows multiprocessing issues with DetectionTrainer subclassing.

Run single:  python run_ablations.py dronescan_full
Run all:     python run_ablations.py
Skip done:   python run_ablations.py --skip abl_sal_nwd,abl_msfd
"""

import csv
import sys
import time
import torch
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from models.modules.rpa_block import RPAModule
from models.modules.msfd      import MSFDNeck
from dronescan_loss           import DroneScanDetectionLoss

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_ARGS = dict(
    data            = "data/visdrone.yaml",
    epochs          = 100,
    imgsz           = 1280,
    batch           = 4,
    workers         = 0,
    device          = 0,
    optimizer       = "AdamW",
    lr0             = 0.001,
    lrf             = 0.01,
    weight_decay    = 0.0005,
    warmup_epochs   = 5,
    close_mosaic    = 15,
    amp             = True,
    plots           = True,
    verbose         = True,
    copy_paste      = 0.3,
    copy_paste_mode = "flip",
    mosaic          = 1.0,
    mixup           = 0.1,
    scale           = 0.9,
    translate       = 0.1,
    fliplr          = 0.5,
    pretrained      = True,
    exist_ok        = True,
)

CONFIGS = {
    "abl_sal_nwd":   {"description": "YOLOv8s + SAL-NWD",
                      "use_rpa": False, "use_msfd": False, "use_sal_nwd": True},
    "abl_msfd":      {"description": "YOLOv8s + MSFD",
                      "use_rpa": False, "use_msfd": True,  "use_sal_nwd": False},
    "abl_rpa":       {"description": "YOLOv8s + RPA",
                      "use_rpa": True,  "use_msfd": False, "use_sal_nwd": False},
    "abl_rpa_msfd":  {"description": "YOLOv8s + RPA + MSFD",
                      "use_rpa": True,  "use_msfd": True,  "use_sal_nwd": False},
    "abl_rpa_sal":   {"description": "YOLOv8s + RPA + SAL-NWD",
                      "use_rpa": True,  "use_msfd": False, "use_sal_nwd": True},
    "abl_msfd_sal":  {"description": "YOLOv8s + MSFD + SAL-NWD",
                      "use_rpa": False, "use_msfd": True,  "use_sal_nwd": True},
    "dronescan_full":{"description": "DroneScan-YOLO (all)",
                      "use_rpa": True,  "use_msfd": True,  "use_sal_nwd": True},
}


def inject_modules(yolo_model, config):
    """Inject RPA, MSFD, SAL-NWD into a loaded YOLO model via hooks."""
    rpa_modules = []
    hooks       = []
    device      = next(yolo_model.model.parameters()).device

    # RPA
    if config["use_rpa"]:
        for idx, (in_ch, out_ch) in [(2, (64, 64)), (4, (128, 128))]:
            rpa = RPAModule(
                in_channels=in_ch, out_channels=out_ch,
                warmup_epochs=10, update_interval=5, threshold=0.85,
            ).to(device)

            def make_hook(rpa_mod, ch):
                def hook(module, input, output):
                    if not isinstance(output, torch.Tensor):
                        return output
                    if output.shape[1] != ch:
                        return output
                    try:
                        orig_dtype = output.dtype
                        x   = output.detach().float()
                        out = rpa_mod(x)
                        return (out + output.float() -
                                output.float().detach()).to(orig_dtype)
                    except Exception:
                        return output
                return hook

            h = yolo_model.model.model[idx].register_forward_hook(
                make_hook(rpa, out_ch))
            hooks.append(h)
            rpa_modules.append(rpa)

        print("  [RPA] Injected at layers 2 (64ch) and 4 (128ch)")

    # MSFD
    if config["use_msfd"]:
        msfd    = MSFDNeck(64, 128, 64).to(device)
        p2_buf  = [None]

        def hook_p2(module, input, output):
            if isinstance(output, torch.Tensor):
                p2_buf[0] = output.detach().float()

        def hook_p3(module, input, output):
            if isinstance(output, torch.Tensor) and p2_buf[0] is not None:
                try:
                    msfd(p2_buf[0], output.detach().float())
                except Exception:
                    pass

        hooks.append(yolo_model.model.model[2].register_forward_hook(hook_p2))
        hooks.append(yolo_model.model.model[4].register_forward_hook(hook_p3))
        print("  [MSFD] P2 branch active")

    # SAL-NWD
    if config["use_sal_nwd"]:
        try:
            from ultralytics.utils.torch_utils import de_parallel
            inner = de_parallel(yolo_model.model)
            inner.criterion = DroneScanDetectionLoss(inner, lambda_nwd=0.5)
            print("  [SAL-NWD] Criterion replaced")
        except Exception as e:
            print(f"  [SAL-NWD] Warning: {e}")

    return rpa_modules, hooks


def train_run(run_name: str, resume: bool = False) -> dict:
    config = CONFIGS[run_name]
    print(f"\n{'='*60}")
    print(f"  {config['description']}")
    print(f"  RPA={config['use_rpa']} | "
          f"MSFD={config['use_msfd']} | "
          f"SAL-NWD={config['use_sal_nwd']}")
    print(f"{'='*60}")

    weights = "yolov8s.pt"
    if resume:
        last = Path(f"runs/detect/{run_name}/weights/last.pt")
        if last.exists():
            weights = str(last)
            print(f"  Resuming from {last}")

    model = YOLO(weights)
    rpa_modules, hooks = inject_modules(model, config)

    def on_epoch_end(trainer):
        for rpa in rpa_modules:
            rpa.step_epoch()
        if rpa_modules and trainer.epoch % 10 == 0:
            sp = sum(
                m.get_stats()["rpa1_sparsity"] + m.get_stats()["rpa2_sparsity"]
                for m in rpa_modules
            ) / (2 * len(rpa_modules))
            print(f"\n  [RPA] Epoch {trainer.epoch} sparsity={sp:.2%}")

    model.add_callback("on_fit_epoch_end", on_epoch_end)

    t_start = time.time()
    model.train(
        **TRAIN_ARGS,
        project = "runs/detect",
        name    = run_name,
        resume  = resume,
    )
    elapsed = (time.time() - t_start) / 60

    for h in hooks:
        h.remove()

    metrics = load_metrics(run_name)
    fps     = measure_fps(run_name)

    print(f"\n  mAP@50={metrics.get('map50',0):.4f} | "
          f"time={elapsed:.1f}min | fps={fps:.1f}")

    return {
        "name":           run_name,
        "description":    config["description"],
        "train_time_min": elapsed,
        "fps":            fps,
        **metrics,
    }


def load_metrics(run_name: str) -> dict:
    p = Path(f"runs/detect/{run_name}/results.csv")
    if not p.exists():
        return {"map50": 0, "map50_95": 0, "precision": 0, "recall": 0}
    row = pd.read_csv(p).iloc[-1]
    return {
        "map50":     float(row.get("metrics/mAP50(B)",    0)),
        "map50_95":  float(row.get("metrics/mAP50-95(B)", 0)),
        "precision": float(row.get("metrics/precision(B)",0)),
        "recall":    float(row.get("metrics/recall(B)",   0)),
    }


def measure_fps(run_name: str, n: int = 100) -> float:
    w = Path(f"runs/detect/{run_name}/weights/best.pt")
    if not w.exists():
        return 0.0
    m = YOLO(str(w))
    d = torch.zeros(1, 3, 640, 640)
    for _ in range(5):
        m.predict(d, verbose=False)
    t = time.time()
    for _ in range(n):
        m.predict(d, verbose=False)
    return n / (time.time() - t)


def load_baseline() -> dict:
    p = Path("runs/detect/baseline/results.csv")
    if not p.exists():
        return {}
    row = pd.read_csv(p).iloc[-1]
    return {
        "name": "baseline", "description": "YOLOv8s baseline",
        "map50":     float(row.get("metrics/mAP50(B)",    0)),
        "map50_95":  float(row.get("metrics/mAP50-95(B)", 0)),
        "precision": float(row.get("metrics/precision(B)",0)),
        "recall":    float(row.get("metrics/recall(B)",   0)),
        "fps": measure_fps("baseline"), "train_time_min": 0,
    }


def save_csv(results: list):
    p = RESULTS_DIR / "ablation_results.csv"
    f = ["name","description","map50","map50_95","precision","recall",
         "fps","train_time_min"]
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=f, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"  Saved -> {p}")


def print_summary(results: list):
    best = max(r.get("map50", 0) for r in results)
    print(f"\n{'='*70}")
    print(f"  {'Model':<30} {'mAP@50':>8} {'mAP50-95':>9} "
          f"{'Recall':>8} {'FPS':>6}")
    print(f"  {'-'*30} {'-'*8} {'-'*9} {'-'*8} {'-'*6}")
    for r in results:
        s = " ★" if r.get("map50",0) == best else ""
        print(f"  {r['description']:<30} "
              f"{r.get('map50',0):>8.4f} "
              f"{r.get('map50_95',0):>9.4f} "
              f"{r.get('recall',0):>8.4f} "
              f"{r.get('fps',0):>6.1f}{s}")
    print(f"{'='*70}")


def main():
    argv       = sys.argv[1:]
    skip_runs  = []
    resume_run = None

    if "--skip"   in argv:
        skip_runs  = argv[argv.index("--skip")   + 1].split(",")
    if "--resume" in argv:
        resume_run = argv[argv.index("--resume") + 1]

    if argv and not argv[0].startswith("--"):
        run_name = argv[0]
        assert run_name in CONFIGS, \
            f"Unknown '{run_name}'. Choose: {list(CONFIGS.keys())}"
        train_run(run_name, resume=(run_name == resume_run))
        return

    print("="*60)
    print("  DroneScan-YOLO Full Ablation (7 runs)")
    print("  100ep | 1280px | copy-paste=0.3")
    print("="*60)

    all_results = []
    b = load_baseline()
    if b:
        print(f"\n  Baseline mAP@50={b['map50']:.4f} | FPS={b['fps']:.1f}")
        all_results.append(b)

    for run_name in CONFIGS:
        if run_name in skip_runs:
            m = load_metrics(run_name)
            if m["map50"] > 0:
                all_results.append({"name": run_name,
                    "description": CONFIGS[run_name]["description"],
                    "fps": measure_fps(run_name), **m})
            continue
        r = train_run(run_name, resume=(run_name == resume_run))
        all_results.append(r)
        save_csv(all_results)

    print_summary(all_results)
    save_csv(all_results)
    print("\nDone.")


if __name__ == "__main__":
    main()