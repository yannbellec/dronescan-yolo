"""
Diagnostic: find the correct layer indices and output shapes
in YOLOv8s backbone to identify P2 and P3 feature maps.
"""

import torch
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = YOLO("yolov8s.pt").model.to(device)

x      = torch.randn(1, 3, 640, 640).to(device)
shapes = {}

def make_hook(idx):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            shapes[idx] = output.shape
    return hook

hooks = []
for idx, layer in enumerate(model.model):
    h = layer.register_forward_hook(make_hook(idx))
    hooks.append(h)

with torch.no_grad():
    model(x)

for h in hooks:
    h.remove()

print("Layer index -> output shape (looking for stride 4=160x160, stride 8=80x80)")
for idx, shape in shapes.items():
    marker = ""
    if len(shape) == 4:
        if shape[2] == 160: marker = "  <-- P2 (stride 4)"
        if shape[2] == 80:  marker = "  <-- P3 (stride 8)"
        if shape[2] == 40:  marker = "  <-- P4 (stride 16)"
        if shape[2] == 20:  marker = "  <-- P5 (stride 32)"
    print(f"  Layer {idx:2d}: {str(shape):<35}{marker}")