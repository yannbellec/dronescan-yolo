import os
from ultralytics import YOLO

runs = ["abl_rpa", "abl_rpa_msfd", "abl_rpa_sal", "abl_msfd_sal", "dronescan_full"]

for name in runs:
    last = f"runs/detect/{name}/weights/last.pt"
    results = f"runs/detect/{name}/results.csv"
    
    # Skip if already fully completed
    if os.path.exists(results) and os.path.exists(last):
        model = YOLO(last)
        try:
            model.train(resume=True)
        except AssertionError:
            print(f"Skipping {name} - already finished")
            continue
    else:
        YOLO('yolov8s.pt').train(
            data='data/visdrone.yaml',
            epochs=50, imgsz=1280, batch=4,
            workers=0, device=0,
            project='runs/detect', name=name,
            optimizer='AdamW', lr0=0.001,
            weight_decay=0.0005, warmup_epochs=5,
            amp=True, mosaic=1.0,
            copy_paste=0.0, mixup=0.0,
            exist_ok=True,
        )