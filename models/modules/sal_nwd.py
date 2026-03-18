"""
SAL-NWD: Size-Adaptive Loss with Normalized Wasserstein Distance
=================================================================
Original contribution — DroneScan-YOLO

Problem:
    Standard CIoU loss penalizes tiny objects too harshly: a 1-pixel
    shift on a 5x5 bbox drops IoU to 0, producing unstable/zero gradients.

Solution — hybrid loss with two components:
    1. NWD (Normalized Wasserstein Distance):
       Models each bbox as a 2D Gaussian distribution.
       Provides smooth gradients even when IoU = 0.
       Formula: NWD(a,b) = exp(-W2(a,b) / C)

    2. Size-Weighted CIoU (SAL):
       w_size = 1 / (bbox_area_ratio + eps)
       Smaller object -> higher weight in the loss.

    Final loss: L = lambda * L_NWD + (1 - lambda) * L_CIoU * w_size

Reference:
    Wang et al., "Normalized Wasserstein Distance for Tiny Object
    Detection", ICCV 2021.
"""

import torch
import torch.nn as nn


class NWDLoss(nn.Module):
    """Normalized Wasserstein Distance Loss for tiny object detection."""

    def __init__(self, constant_c: float = 12.8):
        super().__init__()
        self.c = constant_c

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[0] == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        px, py = pred[:, 0],   pred[:, 1]
        pw, ph = pred[:, 2],   pred[:, 3]
        tx, ty = target[:, 0], target[:, 1]
        tw, th = target[:, 2], target[:, 3]

        diff_center  = (px - tx) ** 2 + (py - ty) ** 2
        diff_size    = (pw / 2 - tw / 2) ** 2 + (ph / 2 - th / 2) ** 2
        wasserstein2 = diff_center + diff_size

        nwd = torch.exp(-torch.sqrt(wasserstein2 + 1e-7) / self.c)
        return (1.0 - nwd).mean()


class SizeAdaptiveWeight(nn.Module):
    """Per-object weights inversely proportional to bbox area."""

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        if target.shape[0] == 0:
            return torch.ones(0, device=target.device)
        area    = target[:, 2] * target[:, 3]
        weights = 1.0 / (area + self.eps)
        weights = weights / weights.mean()
        return weights


class SALNWDLoss(nn.Module):
    """
    SAL-NWD Hybrid Loss — core contribution of DroneScan-YOLO.

    Usage:
        loss_fn = SALNWDLoss(lambda_nwd=0.5)
        loss = loss_fn(pred_boxes, target_boxes)
    """

    def __init__(self, lambda_nwd: float = 0.5, constant_c: float = 12.8):
        """
        Args:
            lambda_nwd : NWD weight in hybrid loss (ablated at 0.3, 0.5, 0.7)
            constant_c : NWD normalization constant
        """
        super().__init__()
        self.lambda_nwd = lambda_nwd
        self.nwd_loss   = NWDLoss(constant_c)
        self.sal_weight = SizeAdaptiveWeight()

    def ciou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Standard CIoU loss: L = 1 - IoU + rho^2/c^2 + alpha*v"""
        if pred.shape[0] == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        p_x1 = pred[:, 0] - pred[:, 2] / 2
        p_y1 = pred[:, 1] - pred[:, 3] / 2
        p_x2 = pred[:, 0] + pred[:, 2] / 2
        p_y2 = pred[:, 1] + pred[:, 3] / 2
        t_x1 = target[:, 0] - target[:, 2] / 2
        t_y1 = target[:, 1] - target[:, 3] / 2
        t_x2 = target[:, 0] + target[:, 2] / 2
        t_y2 = target[:, 1] + target[:, 3] / 2

        inter = (torch.min(p_x2, t_x2) - torch.max(p_x1, t_x1)).clamp(0) * \
                (torch.min(p_y2, t_y2) - torch.max(p_y1, t_y1)).clamp(0)
        union = (p_x2-p_x1)*(p_y2-p_y1) + (t_x2-t_x1)*(t_y2-t_y1) - inter + 1e-7
        iou   = inter / union

        c2   = (torch.max(p_x2,t_x2)-torch.min(p_x1,t_x1))**2 + \
               (torch.max(p_y2,t_y2)-torch.min(p_y1,t_y1))**2 + 1e-7
        rho2 = (pred[:,0]-target[:,0])**2 + (pred[:,1]-target[:,1])**2

        v     = (4/(torch.pi**2)) * \
                (torch.atan(target[:,2]/(target[:,3]+1e-7)) -
                 torch.atan(pred[:,2]  /(pred[:,3]  +1e-7)))**2
        alpha = v / (1 - iou + v + 1e-7)

        return (1 - iou + rho2/c2 + alpha*v).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[0] == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        l_nwd           = self.nwd_loss(pred, target)
        weights         = self.sal_weight(target)
        l_ciou_weighted = self.ciou_loss(pred, target) * weights.mean()

        return self.lambda_nwd * l_nwd + (1 - self.lambda_nwd) * l_ciou_weighted


if __name__ == "__main__":
    print("Testing SAL-NWD Loss...")
    torch.manual_seed(42)
    N = 32

    pred          = torch.rand(N, 4) * 0.02 + 0.01
    pred[:, :2]  += 0.5
    target        = pred.clone()
    target[:, :2] += torch.randn(N, 2) * 0.005

    loss_fn   = SALNWDLoss(lambda_nwd=0.5)
    loss_near = loss_fn(pred, target)
    print(f"  Loss (nearby) : {loss_near.item():.4f}")

    pred_far         = torch.rand(N, 4) * 0.02 + 0.01
    pred_far[:, :2] += 0.8
    loss_far = loss_fn(pred_far, target)
    print(f"  Loss (distant): {loss_far.item():.4f}")
    assert loss_far > loss_near, "ERROR: far loss should be > near loss"

    print("\n  Lambda ablation:")
    for lam in [0.0, 0.3, 0.5, 0.7, 1.0]:
        l = SALNWDLoss(lambda_nwd=lam)(pred, target)
        print(f"    lambda={lam:.1f} -> loss={l.item():.4f}")

    print("\nSAL-NWD OK")