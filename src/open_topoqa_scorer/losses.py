"""Within-target pairwise ranking loss for the scorer.

Absolute MSE regression rewards predicting each target's *mean* DockQ; with docking-decoy sets that
are mostly incorrect, the model minimizes MSE by collapsing every decoy of a target to ~the same low
value and never learns to pick out the near-native ones (within-target ranking ≈ random). This hinge
loss penalizes mis-ordering two decoys of the **same** target: for a pair (i, j) whose true DockQ
differs by more than ``deadband`` (i better than j), the predicted scores must satisfy
``pred_i ≥ pred_j + margin`` or incur ``margin − (pred_i − pred_j)``. Added to MSE it makes the model
spread predictions to match the within-target quality order without abandoning absolute calibration.
"""

from __future__ import annotations

import torch

__all__ = ["within_target_ranking_loss"]


def within_target_ranking_loss(pred, true, target_idx, margin: float = 0.1, deadband: float = 0.02):
    """Mean hinge over same-target pairs (i, j) with ``true_i > true_j + deadband``.

    ``pred``/``true``/``target_idx`` are 1-D tensors of equal length. Returns a scalar tensor; 0 if
    no target has a qualifying pair (so it is a no-op on singleton/tied batches, keeping gradients
    well-defined). Differentiable in ``pred``.
    """
    pred = pred.view(-1)
    true = true.view(-1)
    target_idx = target_idx.view(-1)

    hinge_sum = pred.new_zeros(())
    n_pairs = 0
    for t in torch.unique(target_idx):
        m = target_idx == t
        p = pred[m]
        y = true[m]
        if p.numel() < 2:
            continue
        better = (y.unsqueeze(1) - y.unsqueeze(0)) > deadband  # [i,j]: i meaningfully better than j
        if not better.any():
            continue
        gap = p.unsqueeze(1) - p.unsqueeze(0)  # [i,j]: pred_i - pred_j
        hinge = torch.clamp(margin - gap, min=0.0)
        hinge_sum = hinge_sum + hinge[better].sum()
        n_pairs += int(better.sum())

    if n_pairs == 0:
        return pred.new_zeros(())
    return hinge_sum / n_pairs
