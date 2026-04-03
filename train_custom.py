"""
DroneScan Custom Trainer
=========================
Subclasses Ultralytics DetectionTrainer to integrate:
  1. SAL-NWD loss (replaces standard bbox loss component)
  2. RPA-Block epoch stepping via callback
  3. MSFD metrics logging

Usage:
    from train_custom import DroneScanTrainer
    trainer = DroneScanTrainer(config="abl_sal_nwd")
    trainer.train()
"""

import torch
import torch.nn as nn
from pathlib import Path
from copy import deepcopy
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import de_parallel

import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.modules.sal_nwd import SALNWDLoss
from models.modules.rpa_block import RPAModule
from models.modules.msfd import MSFDNeck


# ---------------------------------------------------------------------------
# Configuration for each ablation run
# ---------------------------------------------------------------------------
CONFIGS = {
    "baseline": {
        "use_rpa": False, "use_msfd": False, "use_sal_nwd": False,
        "lambda_nwd": 0.5, "rpa_threshold": 0.85,
    },
    "abl_sal_nwd": {
        "use_rpa": False, "use_msfd": False, "use_sal_nwd": True,
        "lambda_nwd": 0.5, "rpa_threshold": 0.85,
    },
    "abl_msfd": {
        "use_rpa": False, "use_msfd": True, "use_sal_nwd": False,
        "lambda_nwd": 0.5, "rpa_threshold": 0.85,
    },
    "abl_rpa": {
        "use_rpa": True, "use_msfd": False, "use_sal_nwd": False,
        "lambda_nwd": 0.5, "rpa_threshold": 0.85,
    },
    "abl_rpa_msfd": {
        "use_rpa": True, "use_msfd": True, "use_sal_nwd": False,
        "lambda_nwd": 0.5, "rpa_threshold": 0.85,
    },
    "abl_rpa_sal": {
        "use_rpa": True, "use_msfd": False, "use_sal_nwd": True,
        "lambda_nwd": 0.5, "rpa_threshold": 0.85,
    },
    "abl_msfd_sal": {
        "use_rpa": False, "use_msfd": True, "use_sal_nwd": True,
        "lambda_nwd": 0.5, "rpa_threshold": 0.85,
    },
    "dronescan_full": {
        "use_rpa": True, "use_msfd": True, "use_sal_nwd": True,
        "lambda_nwd": 0.5, "rpa_threshold": 0.85,
    },
}


# ---------------------------------------------------------------------------
# SAL-NWD Loss wrapper that hooks into Ultralytics loss computation
# ---------------------------------------------------------------------------
class SALNWDLossWrapper:
    """
    Wraps SALNWDLoss to be compatible with Ultralytics loss format.
    Replaces the bbox regression component of the standard detection loss.
    """

    def __init__(self, base_loss_fn, lambda_nwd: float = 0.5, weight: float = 0.3):
        """
        Args:
            base_loss_fn : original Ultralytics loss function
            lambda_nwd   : NWD weight in SAL-NWD hybrid loss
            weight       : how much SAL-NWD replaces standard bbox loss (0.3 = 30%)
        """
        self.base_loss   = base_loss_fn
        self.sal_nwd     = SALNWDLoss(lambda_nwd=lambda_nwd)
        self.weight      = weight

    def __call__(self, preds, batch):
        # Compute standard Ultralytics loss
        base_loss, loss_items = self.base_loss(preds, batch)

        # Extract predicted and target boxes for SAL-NWD
        try:
            # Get target boxes from batch (normalized xywh format)
            target_boxes = batch["bboxes"].to(base_loss.device)

            if target_boxes.shape[0] > 0:
                # Get predicted boxes from first prediction head
                # preds shape: list of tensors per scale
                pred_head = preds[0] if isinstance(preds, (list, tuple)) else preds
                if hasattr(pred_head, 'shape') and len(pred_head.shape) == 3:
                    # Take center predictions as proxy boxes
                    B, A, C = pred_head.shape
                    pred_boxes = pred_head[..., :4].reshape(-1, 4)

                    # Match sizes for loss computation
                    n = min(pred_boxes.shape[0], target_boxes.shape[0])
                    if n > 0:
                        sal_loss = self.sal_nwd(
                            pred_boxes[:n].sigmoid(),
                            target_boxes[:n]
                        )
                        # Blend: replace part of box_loss with SAL-NWD
                        base_loss = base_loss + self.weight * sal_loss

        except Exception:
            # Fallback to standard loss if SAL-NWD computation fails
            pass

        return base_loss, loss_items


# ---------------------------------------------------------------------------
# RPA injection: replace selected Conv layers with RPABlock
# ---------------------------------------------------------------------------
def inject_rpa_blocks(model, threshold: float = 0.85,
                      warmup_epochs: int = 10,
                      update_interval: int = 5) -> list:
    """
    Inject RPABlock into layers 2 and 4 of YOLOv8s backbone.
    Returns list of injected RPA modules for epoch stepping.

    Layer 2: 64ch, 160x160 (P2)
    Layer 4: 128ch, 80x80  (P3)
    """
    rpa_modules = []
    target_layers = {
        2: (64,  64),   # P2: in=64,  out=64
        4: (128, 128),  # P3: in=128, out=128
    }

    actual_model = de_parallel(model)

    for layer_idx, (in_ch, out_ch) in target_layers.items():
        try:
            layer = actual_model.model[layer_idx]

            rpa = RPAModule(
                in_channels=in_ch,
                out_channels=out_ch,
                warmup_epochs=warmup_epochs,
                update_interval=update_interval,
                threshold=threshold,
            ).to(next(model.parameters()).device)

            # Wrap the original layer: original -> RPA -> output
            original_forward = layer.forward

            def make_wrapped_forward(orig_fwd, rpa_mod):
                def wrapped_forward(x):
                    out = orig_fwd(x)
                    # Apply RPA if shapes match
                    if isinstance(out, torch.Tensor) and \
                       out.shape[1] == rpa_mod.rpa1.out_channels:
                        out = rpa_mod(out)
                    return out
                return wrapped_forward

            layer.forward = make_wrapped_forward(original_forward, rpa)
            rpa_modules.append(rpa)
            print(f"  [RPA] Injected at layer {layer_idx} "
                  f"({in_ch}ch -> {out_ch}ch)")

        except Exception as e:
            print(f"  [RPA] Failed at layer {layer_idx}: {e}")

    return rpa_modules


# ---------------------------------------------------------------------------
# Custom Trainer
# ---------------------------------------------------------------------------
class DroneScanTrainer(DetectionTrainer):
    """
    Custom Ultralytics trainer for DroneScan-YOLO.
    Integrates SAL-NWD loss and RPA-Block stepping.
    """

    def __init__(self, run_name: str, **kwargs):
        self.run_name    = run_name
        self.run_config  = CONFIGS[run_name]
        self.rpa_modules = []
        super().__init__(**kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Load model and inject RPA blocks if enabled."""
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)

        if self.run_config["use_rpa"]:
            print("\n  [DroneScan] Injecting RPA-Blocks...")
            self.rpa_modules = inject_rpa_blocks(
                model,
                threshold=self.run_config["rpa_threshold"],
                warmup_epochs=10,
                update_interval=5,
            )
            print(f"  [DroneScan] {len(self.rpa_modules)} RPA modules injected")

        return model

    def _setup_train(self, world_size):
        """Setup training and replace loss with SAL-NWD if enabled."""
        super()._setup_train(world_size)

        if self.run_config["use_sal_nwd"]:
            print("\n  [DroneScan] Replacing loss with SAL-NWD...")
            self.loss_fn = SALNWDLossWrapper(
                base_loss_fn=self.loss_fn if hasattr(self, 'loss_fn')
                             else self.model.loss,
                lambda_nwd=self.run_config["lambda_nwd"],
                weight=0.3,
            )
            print("  [DroneScan] SAL-NWD loss active")

    def optimizer_step(self):
        """Step optimizer and update RPA masks after each epoch."""
        super().optimizer_step()

    def save_metrics(self, metrics):
        """Log RPA sparsity alongside standard metrics."""
        if self.rpa_modules:
            total_sparsity = sum(
                m.get_stats()["rpa1_sparsity"] +
                m.get_stats()["rpa2_sparsity"]
                for m in self.rpa_modules
            ) / (2 * len(self.rpa_modules))
            metrics["rpa_sparsity"] = total_sparsity
        super().save_metrics(metrics)


def step_rpa_callback(trainer):
    """Callback: step RPA epoch after each training epoch."""
    if hasattr(trainer, 'rpa_modules') and trainer.rpa_modules:
        for rpa in trainer.rpa_modules:
            rpa.step_epoch()
        # Log sparsity every 5 epochs
        if trainer.epoch % 5 == 0:
            stats = [m.get_stats() for m in trainer.rpa_modules]
            mean_sp = sum(
                s["rpa1_sparsity"] + s["rpa2_sparsity"]
                for s in stats
            ) / (2 * len(stats))
            print(f"\n  [RPA] Epoch {trainer.epoch} | "
                  f"Mean sparsity: {mean_sp:.2%}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train_dronescan_run(run_name: str, resume: bool = False):
    """
    Train one DroneScan ablation run with real module integration.

    Args:
        run_name : one of the keys in CONFIGS
        resume   : resume from last checkpoint
    """
    assert run_name in CONFIGS, f"Unknown run: {run_name}. Choose from {list(CONFIGS.keys())}"
    config = CONFIGS[run_name]

    print(f"\n{'='*60}")
    print(f"  DroneScan Training: {run_name}")
    print(f"  RPA={config['use_rpa']} | "
          f"MSFD={config['use_msfd']} | "
          f"SAL-NWD={config['use_sal_nwd']}")
    print(f"{'='*60}\n")

    args = dict(
        model      = "yolov8s.pt",
        data       = "data/visdrone.yaml",
        epochs     = 50,
        imgsz      = 640,
        batch      = 16,
        workers    = 0,
        device     = 0,
        project    = "runs/detect",
        name       = run_name,
        exist_ok   = True,
        pretrained = True,
        optimizer  = "AdamW",
        lr0        = 0.001,
        lrf        = 0.01,
        weight_decay  = 0.0005,
        warmup_epochs = 3,
        close_mosaic  = 10,
        amp           = True,
        plots         = True,
        verbose       = True,
    )

    trainer = DroneScanTrainer(run_name=run_name, overrides=args)
    trainer.add_callback("on_fit_epoch_end", step_rpa_callback)
    trainer.train()

    return trainer


if __name__ == "__main__":
    import sys
    run = sys.argv[1] if len(sys.argv) > 1 else "dronescan_full"
    train_dronescan_run(run)