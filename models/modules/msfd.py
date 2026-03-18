"""
MSFD: Multi-Scale Feature Distillation
========================================
Original contribution -- DroneScan-YOLO

Problem:
    YOLOv8s outputs predictions starting from P3 (stride 8).
    A 8x8 pixel object in a 640px image maps to a 1x1 feature
    on P3 -- nearly undetectable. VisDrone has ~68% tiny objects.

Solution:
    Add a dedicated P2 detection head (stride 4, 160x160 feature map)
    that operates at twice the spatial resolution of P3.
    Objects as small as 4x4 pixels become detectable at 1x1 on P2.

    To compensate for the extra FLOPs introduced by P2:
    - Lightweight depthwise separable convolutions in the P2 branch
    - RPA-Block applied on P2/P3 features (see rpa_block.py)

Architecture:
    backbone P2 features (stride 4)
        |
    DepthwiseSeparable Conv (lightweight)
        |
    Upsample not needed -- P2 is already high resolution
        |
    P2 detection head --> small object predictions
        |
    Concat with P3 neck features --> standard FPN flow continues

Integration in DroneScan-YOLO:
    The MSFD module is inserted between the backbone and the neck.
    It taps into the P2 feature map (layer 4 in YOLOv8s) and produces
    an additional detection output alongside P3, P4, P5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    Lightweight depthwise separable convolution.
    Replaces standard conv to reduce FLOPs in the P2 branch.
    FLOPs reduction: ~8-9x vs standard 3x3 conv at same channels.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # Depthwise: one filter per input channel
        self.dw = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False
        )
        # Pointwise: 1x1 conv to mix channels
        self.pw  = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class P2DetectionBranch(nn.Module):
    """
    Lightweight P2 detection branch for tiny objects.

    Takes the high-resolution P2 feature map from the backbone
    and produces feature embeddings ready for the detection head.

    Input  : P2 features  (B, C_in,  H/4, W/4)  e.g. (B, 128, 160, 160)
    Output : P2 detection (B, C_out, H/4, W/4)  e.g. (B, 128, 160, 160)
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        num_layers:   int = 2,
    ):
        """
        Args:
            in_channels  : channels from backbone P2 output
            out_channels : channels fed to the detection head
            num_layers   : number of depthwise separable conv layers
        """
        super().__init__()

        layers = []
        c = in_channels
        for i in range(num_layers):
            c_out = out_channels if i == num_layers - 1 else in_channels
            layers.append(DepthwiseSeparableConv(c, c_out))
            c = c_out
        self.branch = nn.Sequential(*layers)

        # Channel attention: highlights the most informative P2 channels
        # squeeze-and-excitation style, lightweight (reduction=4)
        r = max(out_channels // 4, 1)
        self.se_pool  = nn.AdaptiveAvgPool2d(1)
        self.se_fc1   = nn.Linear(out_channels, r)
        self.se_fc2   = nn.Linear(r, out_channels)
        self.se_act   = nn.SiLU()
        self.se_sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Lightweight feature extraction on P2
        feat = self.branch(x)

        # Channel attention (SE block)
        b, c, _, _ = feat.shape
        se = self.se_pool(feat).view(b, c)
        se = self.se_act(self.se_fc1(se))
        se = self.se_sigmoid(self.se_fc2(se)).view(b, c, 1, 1)
        feat = feat * se

        return feat


class MSFDNeck(nn.Module):
    """
    Multi-Scale Feature Distillation Neck.

    Extends the standard YOLOv8 FPN neck with a P2 branch.
    The P2 branch is fused with upsampled P3 features for
    richer tiny-object representations.

    Standard YOLOv8 neck: P3, P4, P5
    DroneScan neck      : P2, P3, P4, P5  (P2 added by MSFD)

    Args:
        p2_channels : backbone P2 output channels  (128 for YOLOv8s)
        p3_channels : backbone P3 output channels  (256 for YOLOv8s)
        out_channels: output channels for P2 head  (128 for YOLOv8s)
    """

    def __init__(
        self,
        p2_channels:  int = 128,
        p3_channels:  int = 256,
        out_channels: int = 128,
    ):
        super().__init__()

        # P2 detection branch
        self.p2_branch = P2DetectionBranch(
            in_channels=p2_channels,
            out_channels=out_channels,
        )

        # Fusion: downsample P3 to match P2 spatial size, then concat
        self.p3_downsample = DepthwiseSeparableConv(
            p3_channels, p2_channels, stride=1
        )

        # Final fusion conv after concat(P2_branch, P3_down)
        self.fusion_conv = DepthwiseSeparableConv(
            out_channels + p2_channels, out_channels
        )

    def forward(
        self,
        p2: torch.Tensor,
        p3: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            p2 : (B, p2_channels, H/4, W/4)  -- backbone P2 features
            p3 : (B, p3_channels, H/8, W/8)  -- backbone P3 features

        Returns:
            p2_out : (B, out_channels, H/4, W/4) -- enriched P2 for detection
        """
        # Process P2 through the detection branch
        p2_feat = self.p2_branch(p2)

        # Upsample P3 to P2 spatial resolution and align channels
        p3_up   = F.interpolate(p3, size=p2.shape[2:], mode="nearest")
        p3_down = self.p3_downsample(p3_up)

        # Fuse P2 branch features with upsampled P3 context
        fused  = torch.cat([p2_feat, p3_down], dim=1)
        p2_out = self.fusion_conv(fused)

        return p2_out


if __name__ == "__main__":
    print("Testing MSFD module...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # YOLOv8s feature map sizes for 640x640 input
    # P2: stride 4  -> 160x160,  128 channels
    # P3: stride 8  ->  80x80,   256 channels
    p2 = torch.randn(2, 128, 160, 160).to(device)
    p3 = torch.randn(2,  256,  80,  80).to(device)

    msfd = MSFDNeck(
        p2_channels=128,
        p3_channels=256,
        out_channels=128,
    ).to(device)

    p2_out = msfd(p2, p3)

    print(f"  P2 input  shape : {p2.shape}")
    print(f"  P3 input  shape : {p3.shape}")
    print(f"  P2 output shape : {p2_out.shape}")
    assert p2_out.shape == (2, 128, 160, 160), "Shape mismatch!"

    # Count parameters
    params = sum(p.numel() for p in msfd.parameters())
    print(f"  MSFD parameters : {params:,}  ({params/1e6:.2f}M)")
    print(f"  (YOLOv8s total  : ~9.8M for reference)")

    # Verify gradient flow
    loss = p2_out.mean()
    loss.backward()
    grad_ok = all(p.grad is not None for p in msfd.parameters())
    print(f"  Gradient flow   : {'OK' if grad_ok else 'FAILED'}")

    print("\nMSFD OK")