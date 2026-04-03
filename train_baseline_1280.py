"""
Baseline YOLOv8s @ 1280px / 100 epochs
Resolution-matched baseline for fair comparison with DroneScan-YOLO.
Same config as DroneScan but WITHOUT custom modules.
copy_paste disabled to avoid RAM crashes at 1280px.
"""

from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data            = "data/visdrone.yaml",
    epochs          = 100,
    imgsz           = 1280,
    batch           = 4,
    workers         = 0,
    device          = 0,
    project         = "runs/detect",
    name            = "baseline_1280",
    exist_ok        = True,
    pretrained      = True,
    optimizer       = "AdamW",
    lr0             = 0.001,
    lrf             = 0.01,
    weight_decay    = 0.0005,
    warmup_epochs   = 5,
    close_mosaic    = 15,
    amp             = True,
    plots           = True,
    verbose         = True,
    copy_paste      = 0.0,   # disabled — RAM crash at 1280px
    mixup           = 0.0,   # disabled — RAM crash at 1280px
    mosaic          = 1.0,
    scale           = 0.5,   # reduced scale jitter for stability
    translate       = 0.1,
    fliplr          = 0.5,
)

print("\nBaseline 1280px done.")
print("Results -> runs/detect/baseline_1280/")