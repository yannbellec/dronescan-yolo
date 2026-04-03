"""Test DroneScan v2 YAML with DySample and SimAM."""
import sys
sys.path.insert(0, '.')

import dronescan_registry
import yaml
from ultralytics.nn.tasks import DetectionModel

print("Loading dronescan_v2.yaml with scale='s'...")
with open("dronescan_v2.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["scale"] = "s"

model = DetectionModel(cfg, ch=3, nc=10, verbose=True)
n = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {n:,}")
print(f"Expected: ~10-12M for scale s with DySample+SimAM")

# Check DySample and SimAM are in the model
has_dysample = any("DySample" in str(type(m)) for m in model.modules())
has_simam    = any("SimAM"    in str(type(m)) for m in model.modules())
print(f"DySample in model: {has_dysample}")
print(f"SimAM    in model: {has_simam}")

if has_dysample and has_simam:
    print("\nv2 YAML OK - ready to train")
else:
    print("\nWARNING: some modules not found")