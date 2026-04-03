"""
Debug script — finds exactly where DroneScan training fails.
"""
import sys
import torch
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("="*55)
print("  DroneScan Debug")
print("="*55)

# Step 1: imports
print("\n[1] Importing modules...")
try:
    from dronescan_model import DroneScanModel
    from dronescan_loss import DroneScanDetectionLoss
    from run_ablations import DroneScanTrainer, CONFIGS, TRAIN_CFG
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 2: create trainer
print("\n[2] Creating trainer...")
try:
    config = CONFIGS["dronescan_full"]
    args = {
        "model":    "yolov8s.pt",
        "data":     "data/visdrone.yaml",
        "epochs":   1,           # just 1 epoch for debug
        "imgsz":    640,         # smaller for speed
        "batch":    4,
        "workers":  0,
        "device":   0,
        "project":  "runs/detect",
        "name":     "debug_run",
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "AdamW",
        "lr0":      0.001,
        "warmup_epochs": 1,
        "amp":      True,
        "plots":    False,
        "verbose":  True,
        "copy_paste": 0.0,  # disable for speed
        "close_mosaic": 0,
    }
    trainer = DroneScanTrainer(run_config=config, overrides=args)
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 3: setup train internals
print("\n[3] Setting up training internals...")
try:
    trainer._setup_train(world_size=1)
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 4: try one batch
print("\n[4] Testing one forward pass...")
try:
    trainer.model.train()
    batch = next(iter(trainer.train_loader))
    batch = trainer.preprocess_batch(batch)
    loss, loss_items = trainer.model(batch["img"], batch)
    print(f"  OK — loss={loss.item():.4f}")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*55)
print("  All steps passed — training should work")
print("="*55)