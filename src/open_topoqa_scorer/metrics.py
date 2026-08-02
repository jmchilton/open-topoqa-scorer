"""Evaluation metrics for interface-quality ranking (from the TopoQA/DProQA protocol).

All operate per-target (one array of decoys for one complex); aggregate across targets by
averaging. Kept dependency-free (numpy only) — Spearman is Pearson on average-tied ranks.
"""

from __future__ import annotations

import numpy as np

__all__ = ["pearson", "spearman", "ranking_loss", "top_n_hit_rate", "_average_ranks"]


def pearson(pred, true) -> float:
    """Pearson correlation between predicted and true scores. NaN-safe → 0.0 if degenerate."""
    p = np.asarray(pred, dtype=float).ravel()
    t = np.asarray(true, dtype=float).ravel()
    if p.size < 2 or p.std() == 0 or t.std() == 0:
        return 0.0
    return float(np.corrcoef(p, t)[0, 1])


def _average_ranks(values) -> np.ndarray:
    """Ranks with ties assigned the average of the positions they span (like scipy 'average')."""
    v = np.asarray(values, dtype=float).ravel()
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, dtype=float)
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0  # 1-based average rank
        i = j + 1
    return ranks


def spearman(pred, true) -> float:
    """Spearman rank correlation (Pearson on average-tied ranks)."""
    return pearson(_average_ranks(pred), _average_ranks(true))


def ranking_loss(pred, true) -> float:
    """Top-1 ranking loss: (best true score) − (true score of the model ranked #1 by pred).

    0.0 means the predicted-best decoy really is the best; larger is worse.
    """
    p = np.asarray(pred, dtype=float).ravel()
    t = np.asarray(true, dtype=float).ravel()
    if p.size == 0:
        return 0.0
    picked = int(np.argmax(p))
    return float(t.max() - t[picked])


def top_n_hit_rate(pred, capri, n: int = 10, threshold: int = 1) -> float:
    """Fraction of the top-``n`` predicted decoys that are CAPRI-acceptable or better.

    ``capri`` is an integer CAPRI class per decoy (0 incorrect, 1 acceptable, 2 medium,
    3 high); ``threshold`` is the minimum class counted as a hit (default 1 = acceptable+).
    """
    p = np.asarray(pred, dtype=float).ravel()
    c = np.asarray(capri).ravel()
    if p.size == 0:
        return 0.0
    k = min(n, p.size)
    top = np.argsort(p)[::-1][:k]
    return float(np.mean(c[top] >= threshold))
