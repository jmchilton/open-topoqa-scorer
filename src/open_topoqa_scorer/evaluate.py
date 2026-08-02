"""Per-target aggregation of ranking metrics (the TopoQA/DProQA evaluation protocol).

The scorer is judged per *target*: for each complex, rank its decoys by predicted score and score
that ranking (Spearman vs true DockQ, top-1 ranking loss, top-10 success), then average across
targets. Absolute-fit metrics (Pearson, MSE) are pooled over all decoys instead. Targets with a
single decoy carry no ranking information and are excluded from the ranking averages.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from open_topoqa_scorer import metrics as M

__all__ = ["per_target_ranking_metrics", "pooled_regression_metrics"]


def per_target_ranking_metrics(scores, labels, top_n: int = 10, threshold: int = 1) -> dict:
    """Ranking metrics averaged over targets. ``scores[i]`` aligns with ``labels[i]``."""
    scores = np.asarray(scores, dtype=float).ravel()
    by_target: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_target[lab.target].append(i)

    spearman, rloss, success = [], [], []
    for idx in by_target.values():
        if len(idx) < 2:  # a lone decoy has no ranking to score
            continue
        pred = scores[idx]
        true = np.array([labels[i].dockq for i in idx])
        capri = np.array([labels[i].capri for i in idx])
        spearman.append(M.spearman(pred, true))
        rloss.append(M.ranking_loss(pred, true))
        success.append(M.top_n_success(pred, capri, n=top_n, threshold=threshold))

    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {
        "targets_total": len(by_target),
        "targets_ranked": len(spearman),
        "spearman_mean": mean(spearman),
        "ranking_loss_mean": mean(rloss),
        f"top{top_n}_success_rate": mean(success),
    }


def pooled_regression_metrics(scores, labels) -> dict:
    """Absolute-fit metrics pooled over all decoys (Pearson + MSE vs true DockQ)."""
    scores = np.asarray(scores, dtype=float).ravel()
    true = np.array([lab.dockq for lab in labels], dtype=float)
    mse = float(np.mean((scores - true) ** 2)) if true.size else 0.0
    return {"pearson": M.pearson(scores, true), "mse": mse, "n": int(true.size)}
