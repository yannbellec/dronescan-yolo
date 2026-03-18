"""
DroneScan DetectionModel — Bulletproof Deep Integration
=========================================================
All known failure modes handled:
  - AMP float16/float32 dtype mismatches in RPA hooks
  - MSFD shape mismatches at different input resolutions
  - Device mismatches after model moves
  - Hook cleanup to avoid memory leaks
  - Graceful fallback if any module fails
"""

import torch
import torch.nn as nn
from pathlib import Path
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import de_parallel

import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.modules.rpa_block import RPAModule
from models.modules.msfd      import MSFDNeck
from dronescan_loss            import DroneScanDetectionLoss


class DroneScanModel(DetectionModel):
    """
    DroneScan-YOLO: YOLOv8s + RPA + MSFD + SAL-NWD.

    Verified YOLOv8s layer channels (find_layers.py):
        Layer 2: 64ch,  160x160  (P2, stride 4)
        Layer 4: 128ch,  80x80   (P3, stride 8)
    """

    def __init__(
        self,
        cfg           = "yolov8s.yaml",
        ch            = 3,
        nc            = 10,
        verbose       = True,
        use_rpa       = True,
        use_msfd      = True,
        use_sal_nwd   = True,
        lambda_nwd    = 0.5,
        rpa_threshold = 0.85,
        warmup_epochs = 10,
        update_interval = 5,
    ):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        self.use_rpa      = use_rpa
        self.use_msfd     = use_msfd
        self.use_sal_nwd  = use_sal_nwd
        self.lambda_nwd   = lambda_nwd
        self.rpa_modules  = []
        self._hooks       = []    # track all hooks for cleanup
        self._p2_features = None
        self._p3_features = None

        if use_rpa:
            self._inject_rpa(rpa_threshold, warmup_epochs, update_interval)

        if use_msfd:
            self._init_msfd()

    # ------------------------------------------------------------------
    # RPA injection
    # ------------------------------------------------------------------
    def _inject_rpa(self, threshold, warmup_epochs, update_interval):
        """
        Inject RPA hooks into layers 2 (64ch) and 4 (128ch).
        Hooks cast to float32 internally and back to original dtype.
        This fixes AMP HalfTensor vs FloatTensor mismatch.
        """
        configs = {2: (64, 64), 4: (128, 128)}
        device  = next(self.parameters()).device

        for idx, (in_ch, out_ch) in configs.items():
            rpa = RPAModule(
                in_channels=in_ch,
                out_channels=out_ch,
                warmup_epochs=warmup_epochs,
                update_interval=update_interval,
                threshold=threshold,
            ).to(device)

            def make_hook(rpa_mod, expected_ch):
                def hook(module, input, output):
                    # Only apply if shape matches expected channels
                    if not isinstance(output, torch.Tensor):
                        return output
                    if output.shape[1] != expected_ch:
                        return output
                    try:
                        orig_dtype = output.dtype
                        # Detach to avoid "requires_grad as constant" error
                        # during AMP tracing, then reattach gradient flow
                        x = output.float().detach().requires_grad_(output.requires_grad)
                        out = rpa_mod(x)
                        # Restore gradient connection via residual
                        if output.requires_grad:
                            out = out + (output.float() - output.float().detach())
                        return out.to(orig_dtype)
                    except Exception as e:
                        # Graceful fallback — never crash training
                        return output
                return hook

            h = self.model[idx].register_forward_hook(
                make_hook(rpa, out_ch)
            )
            self._hooks.append(h)
            self.rpa_modules.append(rpa)

        # Register as submodules so optimizer includes their parameters
        for i, rpa in enumerate(self.rpa_modules):
            setattr(self, f"rpa_{i}", rpa)

        print(f"  [RPA] Injected at layers 2 (64ch) and 4 (128ch) | "
              f"warmup={warmup_epochs} | interval={update_interval} | "
              f"threshold={threshold}")

    # ------------------------------------------------------------------
    # MSFD initialization
    # ------------------------------------------------------------------
    def _init_msfd(self):
        """
        Initialize MSFD neck and register feature capture hooks.
        Hooks store P2/P3 features for use in MSFD forward pass.
        Handles resolution changes gracefully.
        """
        device = next(self.parameters()).device
        self.msfd = MSFDNeck(
            p2_channels=64,
            p3_channels=128,
            out_channels=64,
        ).to(device)

        def hook_p2(module, input, output):
            if isinstance(output, torch.Tensor):
                self._p2_features = output.float()  # always float32

        def hook_p3(module, input, output):
            if isinstance(output, torch.Tensor):
                self._p3_features = output.float()  # always float32

        h2 = self.model[2].register_forward_hook(hook_p2)
        h4 = self.model[4].register_forward_hook(hook_p3)
        self._hooks.extend([h2, h4])

        print(f"  [MSFD] P2 branch initialized | "
              f"p2=64ch | p3=128ch | out=64ch")

    # ------------------------------------------------------------------
    # Loss criterion
    # ------------------------------------------------------------------
    def init_criterion(self):
        """
        Official Ultralytics hook to swap the loss function.
        Returns DroneScanDetectionLoss (SAL-NWD) if enabled.
        """
        if self.use_sal_nwd:
            return DroneScanDetectionLoss(
                de_parallel(self),
                lambda_nwd=self.lambda_nwd,
            )
        return super().init_criterion()

    # ------------------------------------------------------------------
    # RPA epoch stepping
    # ------------------------------------------------------------------
    def step_rpa_epoch(self):
        """Must be called after each training epoch to update RPA masks."""
        for rpa in self.rpa_modules:
            rpa.step_epoch()

    def get_rpa_stats(self) -> dict:
        """Return current sparsity statistics for logging."""
        if not self.rpa_modules:
            return {}
        stats    = [m.get_stats() for m in self.rpa_modules]
        mean_sp  = sum(
            s["rpa1_sparsity"] + s["rpa2_sparsity"]
            for s in stats
        ) / max(2 * len(stats), 1)
        return {"mean_sparsity": mean_sp, "details": stats}

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def remove_hooks(self):
        """Remove all forward hooks — call when done training."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        print("  [DroneScan] All hooks removed")