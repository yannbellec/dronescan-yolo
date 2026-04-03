"""
Final evaluation on the official VisDrone TEST set.
All previous results were measured on the VAL set.
For a rigorous paper, final numbers must come from the TEST set
which the model never saw during training or checkpoint selection.
"""
from ultralytics import YOLO

model = YOLO("runs/detect/dronescan_full/weights/best.pt")

print("=" * 50)
print("  FINAL EVALUATION ON TEST SET")
print("  (official VisDrone2019-DET-test-dev)")
print("=" * 50)

results = model.val(
    data="data/visdrone.yaml",
    split="test",        # <-- test set, not val
    imgsz=1280,
    batch=4,
    workers=0,
    verbose=True,
)

print()
print("=" * 50)
print("  TEST SET RESULTS — use these in your paper")
print("=" * 50)
print(f"  mAP@50    : {results.box.map50:.4f}")
print(f"  mAP@50-95 : {results.box.map:.4f}")
print(f"  Precision : {results.box.mp:.4f}")
print(f"  Recall    : {results.box.mr:.4f}")
print()
print("  Per-class AP@50:")
names = results.names
for i, ap in enumerate(results.box.ap50):
    print(f"    {names[i]:20s} {ap:.4f}")