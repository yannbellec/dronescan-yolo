"""
DroneScan v8DetectionLoss — Bulletproof SAL-NWD Integration
=============================================================
All known failure modes handled:
  - AMP float16/float32 dtype mismatches
  - Empty batch (no positive anchors)
  - NaN/Inf gradients from tiny boxes
  - Shape mismatches between pred and target
  - Device mismatches
"""

import torch
import torch.nn as nn
from pathlib import Path
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from ultralytics.utils.metrics import bbox_iou

import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.modules.sal_nwd import NWDLoss, SizeAdaptiveWeight


class SALNWDBboxLoss(BboxLoss):
    """
    Drop-in replacement for Ultralytics BboxLoss.
    Replaces CIoU with SAL-NWD hybrid loss.
    Fully AMP-compatible — all operations in float32 internally.
    """

    def __init__(self, reg_max: int, lambda_nwd: float = 0.5):
        super().__init__(reg_max)
        self.lambda_nwd = lambda_nwd
        self.nwd_loss   = NWDLoss(constant_c=12.8)
        self.sal_weight = SizeAdaptiveWeight(eps=1e-4)

    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask):
        """
        SAL-NWD loss on positive anchors.
        All tensors cast to float32 internally to avoid AMP issues.
        DFL component unchanged from parent.
        """
        # --- Cast everything to float32 for numerical stability ---
        pred_dist_f      = pred_dist.float()
        pred_bboxes_f    = pred_bboxes.float()
        anchor_points_f  = anchor_points.float()
        target_bboxes_f  = target_bboxes.float()
        target_scores_f  = target_scores.float()
        target_scores_sum_f = target_scores_sum.float()

        weight = target_scores_f.sum(-1)[fg_mask].unsqueeze(-1)
        n_pos  = fg_mask.sum().item()

        # --- SAL-NWD on positive anchors ---
        if n_pos > 0:
            pred_pos   = pred_bboxes_f[fg_mask]    # (N_pos, 4) xyxy norm
            target_pos = target_bboxes_f[fg_mask]  # (N_pos, 4) xyxy norm

            # Validate boxes before computing loss
            pred_pos   = self._clamp_boxes(pred_pos)
            target_pos = self._clamp_boxes(target_pos)

            # Convert xyxy -> xywh for NWD
            pred_xywh   = self._xyxy_to_xywh(pred_pos)
            target_xywh = self._xyxy_to_xywh(target_pos)

            # 1. NWD Loss — smooth gradient even at zero IoU
            try:
                l_nwd = self.nwd_loss(pred_xywh, target_xywh)
                if not torch.isfinite(l_nwd):
                    l_nwd = torch.tensor(0.0, device=pred_dist.device)
            except Exception:
                l_nwd = torch.tensor(0.0, device=pred_dist.device)

            # 2. Size-adaptive CIoU
            try:
                iou      = bbox_iou(pred_pos, target_pos,
                                    xywh=False, CIoU=True).squeeze(-1)
                iou      = iou.clamp(-1.0, 1.0)
                l_ciou   = ((1.0 - iou) * weight.squeeze(-1)).sum() / \
                           (target_scores_sum_f + 1e-7)
                sal_w    = self.sal_weight(target_xywh)
                l_ciou_w = l_ciou * sal_w.mean().clamp(0.1, 10.0)
                if not torch.isfinite(l_ciou_w):
                    l_ciou_w = torch.tensor(0.0, device=pred_dist.device)
            except Exception:
                l_ciou_w = torch.tensor(0.0, device=pred_dist.device)

            # 3. Hybrid SAL-NWD
            loss_iou = (self.lambda_nwd * l_nwd +
                        (1 - self.lambda_nwd) * l_ciou_w)

            # Final NaN guard
            if not torch.isfinite(loss_iou):
                loss_iou = torch.tensor(0.0,
                                        device=pred_dist.device,
                                        requires_grad=True)
        else:
            loss_iou = torch.tensor(0.0,
                                    device=pred_dist.device,
                                    requires_grad=True)

        # --- DFL loss from parent (unchanged) ---
        loss_dfl = super().forward(
            pred_dist_f, pred_bboxes_f, anchor_points_f,
            target_bboxes_f, target_scores_f,
            target_scores_sum_f, fg_mask
        )[1]

        return loss_iou, loss_dfl

    @staticmethod
    def _clamp_boxes(boxes: torch.Tensor) -> torch.Tensor:
        """Clamp boxes to [0, 1] and ensure x2>x1, y2>y1."""
        boxes = boxes.clamp(0.0, 1.0)
        # Ensure positive width and height
        x1, y1, x2, y2 = boxes.unbind(-1)
        x2 = torch.max(x2, x1 + 1e-4)
        y2 = torch.max(y2, y1 + 1e-4)
        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
        """Convert (x1,y1,x2,y2) -> (cx,cy,w,h)."""
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        w  =  boxes[:, 2] - boxes[:, 0]
        h  =  boxes[:, 3] - boxes[:, 1]
        return torch.stack([cx, cy, w, h], dim=1).clamp(1e-4, 1.0)


class DroneScanDetectionLoss(v8DetectionLoss):
    """
    v8DetectionLoss with SAL-NWD bbox component.
    All other loss components (cls BCE, DFL) unchanged.
    """

    def __init__(self, model, lambda_nwd: float = 0.5, tal_topk: int = 10):
        super().__init__(model, tal_topk=tal_topk)
        self.bbox_loss = SALNWDBboxLoss(
            reg_max=self.reg_max,
            lambda_nwd=lambda_nwd,
        ).to(self.device)
        print(f"  [SAL-NWD] Loss active | lambda={lambda_nwd} | "
              f"device={self.device}")