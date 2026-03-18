"""
DroneScan-YOLO — One-click setup verification
Run this after cloning the repo to verify everything is ready.
"""
import subprocess
import sys

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

print("=" * 55)
print("  DroneScan-YOLO Setup Check")
print("=" * 55)

# Check Python
print(f"\n  Python: {sys.version.split()[0]}")

# Check packages
packages = [
    ("torch",        "PyTorch"),
    ("ultralytics",  "Ultralytics"),
    ("cv2",          "OpenCV"),
    ("numpy",        "NumPy"),
    ("pandas",       "Pandas"),
]

all_ok = True
for pkg, name in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "?")
        print(f"  {name:<15}: {ver} ✓")
    except ImportError:
        print(f"  {name:<15}: NOT FOUND ✗")
        all_ok = False

# Check CUDA
try:
    import torch
    if torch.cuda.is_available():
        print(f"\n  GPU   : {torch.cuda.get_device_name(0)} ✓")
        print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        print(f"  CUDA  : {torch.version.cuda}")
    else:
        print("\n  GPU   : NOT AVAILABLE ✗")
        all_ok = False
except Exception as e:
    print(f"\n  GPU check failed: {e}")

# Check project files
import os
from pathlib import Path

required = [
    "models/modules/rpa_block.py",
    "models/modules/msfd.py",
    "models/modules/sal_nwd.py",
    "dronescan_loss.py",
    "dronescan_model.py",
    "run_ablations.py",
    "train_baseline.py",
    "data/visdrone.yaml",
]

print("\n  Project files:")
for f in required:
    exists = Path(f).exists()
    print(f"  {'✓' if exists else '✗'} {f}")
    if not exists:
        all_ok = False

print("\n" + "=" * 55)
if all_ok:
    print("  All checks passed. Ready to train.")
    print("  Run: python train_baseline.py")
else:
    print("  Some checks failed. See above.")
print("=" * 55)