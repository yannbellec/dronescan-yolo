"""
RPA-Block: Redundancy-Pruned Attention
========================================
Original contribution — DroneScan-YOLO

Problem:
    In YOLOv8 C2f blocks, many filters learn near-identical features
    (redundant). These redundant filters increase FLOPs without
    improving performance — critical issue for UAV embedded deployment.

Solution:
    1. Measure cosine similarity between all filters of a conv layer
    2. Dynamically mask redundant filters (similarity > threshold theta)
    3. Lazy update: recompute mask every N epochs, not every batch
    4. Warm-up: inactive for the first W epochs so filters can learn
       diverse features before pruning begins

Key advantage over post-training pruning:
    The mask evolves during training -> the network gradually adapts
    to sparsity instead of suffering a brutal post-hoc amputation.

Placement in DroneScan-YOLO:
    Applied on C2f blocks handling P2 and P3 features (high resolution)
    to compensate for the extra FLOPs introduced by the MSFD head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RPABlock(nn.Module):
    """
    Single Redundancy-Pruned Attention block.
    Drop-in replacement for a Conv layer inside a C2f block.
    """

    def __init__(
        self,
        in_channels:     int,
        out_channels:    int,
        kernel_size:     int   = 3,
        stride:          int   = 1,
        warmup_epochs:   int   = 10,
        update_interval: int   = 5,
        threshold:       float = 0.85,
    ):
        """
        Args:
            in_channels     : input channels
            out_channels    : output channels
            kernel_size     : convolution kernel size
            stride          : convolution stride
            warmup_epochs   : epochs before pruning activates
            update_interval : recompute mask every N epochs after warm-up
            threshold       : cosine similarity threshold above which
                              a filter is considered redundant (0.85)
        """
        super().__init__()

        self.out_channels    = out_channels
        self.warmup_epochs   = warmup_epochs
        self.update_interval = update_interval
        self.threshold       = threshold

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn  = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

        # Pruning mask: 1 = active filter, 0 = masked filter
        self.register_buffer(
            "pruning_mask",
            torch.ones(out_channels, dtype=torch.float32)
        )

        self._current_epoch  = 0
        self._pruning_active = False
        self._sparsity       = 0.0

    def _compute_redundancy_mask(self) -> torch.Tensor:
        """
        Compute binary pruning mask based on pairwise cosine similarity.

        Algorithm:
          1. Flatten each filter to a vector in R^{Cin * k * k}
          2. L2-normalize each filter vector
          3. Compute (Cout x Cout) cosine similarity matrix
          4. For each pair (i, j) where sim > threshold: mask filter j
             (keep i, discard the duplicate)

        Returns:
            mask: (out_channels,) binary tensor
        """
        with torch.no_grad():
            w      = self.conv.weight.data
            w_flat = w.view(w.shape[0], -1)
            w_norm = F.normalize(w_flat, p=2, dim=1)

            sim_matrix = torch.mm(w_norm, w_norm.t())

            mask = torch.ones(
                self.out_channels,
                dtype=torch.float32,
                device=w.device
            )

            for i in range(self.out_channels):
                if mask[i] == 0:
                    continue
                for j in range(i + 1, self.out_channels):
                    if sim_matrix[i, j] > self.threshold:
                        mask[j] = 0

            return mask

    def step_epoch(self):
        """
        Call after each training epoch.
        Handles warm-up period and lazy mask updates.
        """
        self._current_epoch += 1

        if self._current_epoch <= self.warmup_epochs:
            self._pruning_active = False
            return

        self._pruning_active = True

        epochs_since_warmup = self._current_epoch - self.warmup_epochs
        if epochs_since_warmup % self.update_interval == 0:
            new_mask = self._compute_redundancy_mask()
            self.pruning_mask.copy_(new_mask)
            self._sparsity = 1.0 - self.pruning_mask.mean().item()

    def get_sparsity(self) -> float:
        """Returns current sparsity ratio (0.0 = dense, 1.0 = fully masked)."""
        return self._sparsity

    def get_active_filters(self) -> int:
        """Returns number of active (unmasked) filters."""
        return int(self.pruning_mask.sum().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn(self.conv(x)))

        if self._pruning_active and self.pruning_mask.sum() < self.out_channels:
            mask = self.pruning_mask.view(1, -1, 1, 1)
            out  = out * mask

        return out


class RPAModule(nn.Module):
    """
    Full RPA module: stacks two RPABlocks with an optional residual connection.
    Direct replacement for a C2f block in YOLOv8.

    Architecture:
        x -> RPA1 -> RPA2 -> (+ x if in_ch == out_ch) -> output
    """

    def __init__(
        self,
        in_channels:     int,
        out_channels:    int,
        warmup_epochs:   int   = 10,
        update_interval: int   = 5,
        threshold:       float = 0.85,
    ):
        super().__init__()

        self.use_residual = (in_channels == out_channels)

        self.rpa1 = RPABlock(
            in_channels, out_channels,
            warmup_epochs=warmup_epochs,
            update_interval=update_interval,
            threshold=threshold,
        )
        self.rpa2 = RPABlock(
            out_channels, out_channels,
            warmup_epochs=warmup_epochs,
            update_interval=update_interval,
            threshold=threshold,
        )

    def step_epoch(self):
        """Propagate epoch step to both blocks."""
        self.rpa1.step_epoch()
        self.rpa2.step_epoch()

    def get_stats(self) -> dict:
        """Return sparsity statistics for both blocks."""
        return {
            "rpa1_sparsity":       self.rpa1.get_sparsity(),
            "rpa1_active_filters": self.rpa1.get_active_filters(),
            "rpa2_sparsity":       self.rpa2.get_sparsity(),
            "rpa2_active_filters": self.rpa2.get_active_filters(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rpa2(self.rpa1(x))
        if self.use_residual:
            out = out + x
        return out


if __name__ == "__main__":
    print("Testing RPA-Block...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Simulate P3 feature map (stride 8, 640px input)
    batch  = torch.randn(2, 256, 80, 80).to(device)
    # NOTE: threshold=0.85 is correct for trained weights.
    # Random weights have near-zero cosine similarity, so we use a low
    # threshold here just to validate the pruning logic works correctly.
    module = RPAModule(
        in_channels=256, out_channels=256,
        warmup_epochs=10, update_interval=5, threshold=0.05
    ).to(device)

    out = module(batch)
    print(f"  Input  shape: {batch.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Sparsity (during warm-up): {module.rpa1.get_sparsity():.2%}")

    print("\n  Epoch simulation:")
    for epoch in range(1, 16):
        module.step_epoch()
        stats = module.get_stats()
        if epoch in [5, 10, 11, 15]:
            print(f"    Epoch {epoch:2d} | "
                  f"RPA1 sparsity={stats['rpa1_sparsity']:.2%} "
                  f"({stats['rpa1_active_filters']}/256 active) | "
                  f"RPA2 sparsity={stats['rpa2_sparsity']:.2%}")

    out_pruned = module(batch)
    print(f"\n  Forward after pruning: {out_pruned.shape}")
    diff = (out - out_pruned).abs().mean().item()
    print(f"  Output diff before/after pruning: {diff:.4f} (expected > 0)")

    print("\nRPA-Block OK")


def sensitivity_threshold(device):
    """
    Sensitivity analysis: impact of cosine similarity threshold on sparsity.
    Tests thresholds: 0.50, 0.60, 0.70, 0.80, 0.85, 0.90
    Lower threshold -> more aggressive pruning.
    """
    print("\n--- Sensitivity: RPA cosine threshold ---")
    print(f"  {'Threshold':>10} {'Sparsity':>10} {'Active/64':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10}")
    for thresh in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]:
        block = RPABlock(
            in_channels=64, out_channels=64,
            warmup_epochs=0, update_interval=1,
            threshold=thresh,
        ).to(device)
        block.step_epoch()
        sp = block.get_sparsity()
        af = block.get_active_filters()
        print(f"  {thresh:>10.2f} {sp:>9.2%} {af:>8}/64")


def sensitivity_lazy_update(device):
    """
    Sensitivity analysis: impact of lazy update interval.
    Tests intervals: 1, 3, 5, 10 epochs over 50 training epochs.
    """
    print("\n--- Sensitivity: lazy update interval ---")
    print(f"  {'Interval':>10} {'@epoch15':>10} {'@epoch30':>10} {'@epoch50':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for interval in [1, 3, 5, 10]:
        block = RPABlock(
            in_channels=64, out_channels=64,
            warmup_epochs=10, update_interval=interval,
            threshold=0.05,  # low threshold to show pruning on random weights
        ).to(device)
        sp = {15: 0.0, 30: 0.0, 50: 0.0}
        for epoch in range(1, 51):
            block.step_epoch()
            if epoch in sp:
                sp[epoch] = block.get_sparsity()
        print(f"  {interval:>10} {sp[15]:>9.2%} {sp[30]:>9.2%} {sp[50]:>9.2%}")


def run_sensitivity_analysis(device):
    print("\n" + "="*50)
    print("  RPA Sensitivity Analysis")
    print("="*50)
    sensitivity_threshold(device)
    sensitivity_lazy_update(device)
    print("\nSensitivity analysis complete.")

    # Run sensitivity analysis
    run_sensitivity_analysis(device)