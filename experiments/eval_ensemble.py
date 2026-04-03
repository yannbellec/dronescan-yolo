"""
Ensemble — average the weights of dronescan_full and dronescan_native.
Combining two models trained differently often gives better results
than either model alone. No retraining required.
"""
import os
import torch
from ultralytics import YOLO

model1_path = "runs/detect/dronescan_full/weights/best.pt"
model2_path = "runs/detect/dronescan_native/weights/best.pt"

# Check both models exist
if not os.path.exists(model1_path):
    print(f"ERROR: {model1_path} not found")
    exit(1)
if not os.path.exists(model2_path):
    print(f"WARNING: {model2_path} not found — using dronescan_full only")
    model2_path = None

print("Loading models...")
model1 = YOLO(model1_path)

print("Evaluating model 1 (dronescan_full)...")
r1 = model1.val(data="data/visdrone.yaml", imgsz=1280,
                batch=4, workers=0, verbose=False)
print(f"  dronescan_full  mAP@50 = {r1.box.map50:.4f}")

if model2_path:
    model2 = YOLO(model2_path)
    print("Evaluating model 2 (dronescan_native)...")
    r2 = model2.val(data="data/visdrone.yaml", imgsz=1280,
                    batch=4, workers=0, verbose=False)
    print(f"  dronescan_native mAP@50 = {r2.box.map50:.4f}")

    # Average weights
    print("Averaging weights...")
    sd1 = model1.model.state_dict()
    sd2 = model2.model.state_dict()

    # Only average if architectures match
    if sd1.keys() == sd2.keys():
        sd_ensemble = {}
        for k in sd1:
            if sd1[k].dtype.is_floating_point:
                sd_ensemble[k] = (sd1[k] + sd2[k]) / 2.0
            else:
                sd_ensemble[k] = sd1[k]  # keep integers as-is

        model1.model.load_state_dict(sd_ensemble)
        os.makedirs("runs/detect/ensemble/weights", exist_ok=True)
        model1.save("runs/detect/ensemble/weights/best.pt")
        print("Ensemble saved to runs/detect/ensemble/weights/best.pt")

        print("Evaluating ensemble...")
        r_ens = model1.val(data="data/visdrone.yaml", imgsz=1280,
                           batch=4, workers=0, verbose=False)
        print()
        print("=" * 50)
        print("  ENSEMBLE RESULTS")
        print("=" * 50)
        print(f"  dronescan_full   mAP@50 = {r1.box.map50:.4f}")
        print(f"  dronescan_native mAP@50 = {r2.box.map50:.4f}")
        print(f"  Ensemble         mAP@50 = {r_ens.box.map50:.4f}")
        print(f"  Gain vs best     mAP@50 = +{r_ens.box.map50 - max(r1.box.map50, r2.box.map50):.4f}")
    else:
        print("ERROR: Model architectures don't match — ensemble not possible")
        print("This is expected if dronescan_full and dronescan_native use different YAMLs")
else:
    print("Only one model available — ensemble skipped")