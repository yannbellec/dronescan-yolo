"""
Evaluate DroneScan-YOLO with Test-Time Augmentation (TTA).
TTA runs inference on the image + flipped + scaled versions,
then merges predictions for a better final result.
No retraining required.
"""
from ultralytics import YOLO

model = YOLO("runs/detect/dronescan_full/weights/best.pt")

print("=" * 50)
print("  Evaluation WITHOUT TTA (baseline)")
print("=" * 50)
results_normal = model.val(
    data="data/visdrone.yaml",
    imgsz=1280,
    batch=4,
    workers=0,
    augment=False,
    verbose=True,
)
print(f"  mAP@50    : {results_normal.box.map50:.4f}")
print(f"  mAP@50-95 : {results_normal.box.map:.4f}")
print(f"  Recall    : {results_normal.box.mr:.4f}")

print()
print("=" * 50)
print("  Evaluation WITH TTA")
print("=" * 50)
results_tta = model.val(
    data="data/visdrone.yaml",
    imgsz=1280,
    batch=4,
    workers=0,
    augment=True,   # <-- this enables TTA
    verbose=True,
)
print(f"  mAP@50    : {results_tta.box.map50:.4f}")
print(f"  mAP@50-95 : {results_tta.box.map:.4f}")
print(f"  Recall    : {results_tta.box.mr:.4f}")

print()
print("=" * 50)
print("  TTA GAIN")
print("=" * 50)
print(f"  mAP@50    : +{results_tta.box.map50 - results_normal.box.map50:.4f}")
print(f"  mAP@50-95 : +{results_tta.box.map - results_normal.box.map:.4f}")
print(f"  Recall    : +{results_tta.box.mr - results_normal.box.mr:.4f}")