"""
DroneScan-YOLO: Efficient Real-Time Aerial Object Detection
via Adaptive Pruning and Wasserstein-Guided Loss
=============================================================
Main architecture file -- assembles RPA-Block, MSFD, and SAL-NWD
on top of YOLOv8s for tiny object detection on VisDrone2019-DET.

YOLOv8s actual feature map channels (verified empirically):
    Layer 2  -> P2: 64ch,  160x160  (stride 4)
    Layer 4  -> P3: 128ch,  80x80   (stride 8)
    Layer 6  -> P4: 256ch,  40x40   (stride 16)
    Layer 8  -> P5: 512ch,  20x20   (stride 32)
"""

import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.modules.rpa_block import RPAModule
from models.modules.msfd      import MSFDNeck
from models.modules.sal_nwd   import SALNWDLoss


class DroneScanYOLO(nn.Module):
    """
    DroneScan-YOLO: YOLOv8s + RPA-Block + MSFD + SAL-NWD.

    Verified channel dimensions for YOLOv8s:
        P2: layer 2,  64 channels, 160x160
        P3: layer 4, 128 channels,  80x80
    """

    def __init__(
        self,
        num_classes:     int   = 10,
        pretrained:      bool  = True,
        warmup_epochs:   int   = 10,
        update_interval: int   = 5,
        rpa_threshold:   float = 0.85,
        lambda_nwd:      float = 0.5,
    ):
        super().__init__()

        self.num_classes = num_classes

        # --- 1. YOLOv8s backbone ---
        weights = "yolov8s.pt" if pretrained else "yolov8s.yaml"
        self.backbone = YOLO(weights)

        # --- 2. RPA modules ---
        # Applied on P2 (64ch) and P3 (128ch) — verified via find_layers.py
        self.rpa_p2 = RPAModule(
            in_channels=64, out_channels=64,
            warmup_epochs=warmup_epochs,
            update_interval=update_interval,
            threshold=rpa_threshold,
        )
        self.rpa_p3 = RPAModule(
            in_channels=128, out_channels=128,
            warmup_epochs=warmup_epochs,
            update_interval=update_interval,
            threshold=rpa_threshold,
        )

        # --- 3. MSFD neck ---
        self.msfd = MSFDNeck(
            p2_channels=64,
            p3_channels=128,
            out_channels=64,
        )

        # --- 4. SAL-NWD loss ---
        self.sal_nwd_loss = SALNWDLoss(lambda_nwd=lambda_nwd)

        self._current_epoch = 0

    def step_epoch(self):
        """Call after each training epoch to update RPA masks."""
        self._current_epoch += 1
        self.rpa_p2.step_epoch()
        self.rpa_p3.step_epoch()

    def get_pruning_stats(self) -> dict:
        """Returns current sparsity stats for all RPA modules."""
        p2 = self.rpa_p2.get_stats()
        p3 = self.rpa_p3.get_stats()
        return {
            "epoch":          self._current_epoch,
            "p2_rpa1":        p2["rpa1_sparsity"],
            "p2_rpa2":        p2["rpa2_sparsity"],
            "p3_rpa1":        p3["rpa1_sparsity"],
            "p3_rpa2":        p3["rpa2_sparsity"],
            "mean_sparsity":  (p2["rpa1_sparsity"] + p2["rpa2_sparsity"] +
                               p3["rpa1_sparsity"] + p3["rpa2_sparsity"]) / 4,
        }

    def count_parameters(self) -> dict:
        """Returns parameter counts for each custom component."""
        rpa  = sum(p.numel() for p in self.rpa_p2.parameters()) + \
               sum(p.numel() for p in self.rpa_p3.parameters())
        msfd = sum(p.numel() for p in self.msfd.parameters())
        return {
            "rpa_params":  rpa,
            "msfd_params": msfd,
            "total_extra": rpa + msfd,
        }

    def _extract_features(self, x: torch.Tensor) -> dict:
        """
        Extract P2 and P3 feature maps from YOLOv8s backbone
        using forward hooks on the correct layer indices.

        Verified indices:
            Layer 2 -> P2 (64ch,  160x160)
            Layer 4 -> P3 (128ch,  80x80)
        """
        features = {}
        hooks    = []
        layer_map = {2: "p2", 4: "p3"}

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    features[name] = output
            return hook

        for idx, name in layer_map.items():
            h = self.backbone.model.model[idx].register_forward_hook(
                make_hook(name)
            )
            hooks.append(h)

        with torch.no_grad():
            self.backbone.model(x)

        for h in hooks:
            h.remove()

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: backbone -> RPA -> MSFD -> P2 enriched features.
        Returns enriched P2 feature map (64ch, H/4, W/4).
        """
        features = self._extract_features(x)

        p2 = self.rpa_p2(features["p2"])
        p3 = self.rpa_p3(features["p3"])
        p2_out = self.msfd(p2, p3)

        return p2_out


if __name__ == "__main__":
    print("Testing DroneScan-YOLO assembly...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = DroneScanYOLO(
        num_classes=10, pretrained=True,
        warmup_epochs=10, update_interval=5,
        rpa_threshold=0.85, lambda_nwd=0.5,
    ).to(device)

    params = model.count_parameters()
    print(f"\n  Parameter overhead:")
    print(f"    RPA modules : {params['rpa_params']:,}")
    print(f"    MSFD module : {params['msfd_params']:,}")
    print(f"    Total extra : {params['total_extra']:,} "
          f"(+{params['total_extra']/9_842_830*100:.1f}% over YOLOv8s)")

    x   = torch.randn(2, 3, 640, 640).to(device)
    out = model(x)
    print(f"\n  Input  shape : {x.shape}")
    print(f"  Output shape : {out.shape}")
    assert out.shape == (2, 64, 160, 160), \
        f"Expected (2,64,160,160), got {out.shape}"
    print(f"  Shape check  : OK (P2 at 160x160 confirmed)")

    print("\n  RPA epoch simulation:")
    for target_epoch in [1, 10, 15, 20, 25]:
        while model._current_epoch < target_epoch:
            model.step_epoch()
        stats = model.get_pruning_stats()
        print(f"    Epoch {target_epoch:2d} | "
              f"mean sparsity={stats['mean_sparsity']:.2%}")

    print("\n  SAL-NWD loss test:")
    pred   = torch.rand(32, 4).to(device) * 0.1
    target = pred + torch.randn(32, 4).to(device) * 0.01
    loss   = model.sal_nwd_loss(pred, target)
    print(f"    Loss value: {loss.item():.4f}")

    print("\nDroneScan-YOLO assembly OK")