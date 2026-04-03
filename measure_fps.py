"""Measure inference FPS for baseline YOLOv8s model."""
import torch
import time
from ultralytics import YOLO

model = YOLO("runs/detect/dronescan_full/weights/best.pt")
dummy = torch.zeros(1, 3, 640, 640)

# Warm up
print("Warming up...")
for _ in range(10):
    model.predict(dummy, verbose=False)

# Measure
print("Measuring FPS (100 runs)...")
t = time.time()
for _ in range(100):
    model.predict(dummy, verbose=False)
elapsed = time.time() - t

fps     = 100 / elapsed
latency = elapsed / 100 * 1000

print(f"\n  FPS     : {fps:.1f}")
print(f"  Latency : {latency:.2f} ms per image")
print(f"\n  Note this in your paper as baseline inference speed.")