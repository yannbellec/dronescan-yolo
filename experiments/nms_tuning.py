"""
NMS Tuning — find optimal conf and iou thresholds for VisDrone.
Tests all combinations and prints the best configuration.
No retraining required.
"""
from ultralytics import YOLO
import pandas as pd

model = YOLO("runs/detect/dronescan_full/weights/best.pt")

conf_values = [0.001, 0.005, 0.01, 0.05]
iou_values  = [0.4, 0.5, 0.6, 0.7]

results_list = []
best_map = 0
best_conf = 0
best_iou  = 0

print("Testing all conf/iou combinations...")
print(f"{'conf':>8} {'iou':>6} {'mAP@50':>10} {'mAP@50-95':>12} {'Recall':>8}")
print("-" * 50)

for conf in conf_values:
    for iou in iou_values:
        r = model.val(
            data="data/visdrone.yaml",
            imgsz=1280,
            batch=4,
            workers=0,
            conf=conf,
            iou=iou,
            verbose=False,
        )
        map50    = r.box.map50
        map5095  = r.box.map
        recall   = r.box.mr

        print(f"{conf:>8.3f} {iou:>6.1f} {map50:>10.4f} {map5095:>12.4f} {recall:>8.4f}")

        results_list.append({
            "conf": conf, "iou": iou,
            "mAP50": map50, "mAP50-95": map5095, "Recall": recall
        })

        if map50 > best_map:
            best_map  = map50
            best_conf = conf
            best_iou  = iou

# Save results
df = pd.DataFrame(results_list)
df.to_csv("results/nms_tuning.csv", index=False)

print()
print("=" * 50)
print("  BEST CONFIGURATION")
print("=" * 50)
print(f"  conf      : {best_conf}")
print(f"  iou       : {best_iou}")
print(f"  mAP@50    : {best_map:.4f}")
print(f"  Saved to  : results/nms_tuning.csv")