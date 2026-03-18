"""
RPA-Block Sensitivity Analysis
================================
Analyzes the impact of two key hyperparameters:
  1. Cosine similarity threshold (theta)
  2. Lazy update interval (N epochs)

Results go directly into Section 4 of the paper.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.modules.rpa_block import RPABlock


def sensitivity_threshold(device):
    """
    How does the pruning threshold affect sparsity?
    Lower threshold -> more aggressive pruning -> fewer active filters.
    Tested on random weights (worst case: low similarity by default).
    Using threshold=0.05 as baseline to demonstrate the mechanism.
    """
    print("\n--- Sensitivity: cosine similarity threshold ---")
    print(f"  {'Threshold':>10} {'Sparsity':>12} {'Active/64':>12} {'Note':>20}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*20}")

    notes = {
        0.10: "very aggressive",
        0.20: "aggressive",
        0.30: "moderate",
        0.40: "conservative",
        0.50: "default (random)",
        0.85: "default (trained)",
    }

    for thresh in [0.10, 0.20, 0.30, 0.40, 0.50, 0.85]:
        block = RPABlock(
            in_channels=64, out_channels=64,
            warmup_epochs=0,
            update_interval=1,
            threshold=thresh,
        ).to(device)
        block.step_epoch()
        sp = block.get_sparsity()
        af = block.get_active_filters()
        note = notes.get(thresh, "")
        print(f"  {thresh:>10.2f} {sp:>11.2%} {af:>10}/64  {note:>20}")

    print("\n  Note: on trained weights, threshold=0.85 is appropriate.")
    print("  On random weights, most filters are dissimilar -> low sparsity at 0.85.")


def sensitivity_lazy_update(device):
    """
    How does the lazy update interval affect sparsity dynamics?
    Smaller interval -> more frequent mask updates -> faster sparsification.
    Trade-off: accuracy vs. training speed.
    """
    print("\n--- Sensitivity: lazy update interval ---")
    print(f"  {'Interval':>10} {'@ep15':>8} {'@ep25':>8} {'@ep40':>8} {'@ep50':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for interval in [1, 3, 5, 10]:
        block = RPABlock(
            in_channels=64, out_channels=64,
            warmup_epochs=10,
            update_interval=interval,
            threshold=0.05,
        ).to(device)

        sp = {15: 0.0, 25: 0.0, 40: 0.0, 50: 0.0}
        for epoch in range(1, 51):
            block.step_epoch()
            if epoch in sp:
                sp[epoch] = block.get_sparsity()

        print(f"  {interval:>10} "
              f"{sp[15]:>7.2%} "
              f"{sp[25]:>7.2%} "
              f"{sp[40]:>7.2%} "
              f"{sp[50]:>7.2%}")

    print("\n  Chosen value: interval=5 (balance between update cost and sparsity)")
    print("  Warm-up=10 ensures filters learn diverse features before pruning.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 55)
    print("  RPA-Block Sensitivity Analysis")
    print(f"  Device: {device}")
    print("=" * 55)

    sensitivity_threshold(device)
    sensitivity_lazy_update(device)

    print("\n" + "=" * 55)
    print("  These results go into Section 4 (ablation study)")
    print("=" * 55)


if __name__ == "__main__":
    main()