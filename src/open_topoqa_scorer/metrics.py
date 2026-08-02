"""Evaluation metrics for interface-quality ranking (from the TopoQA/DProQA protocol).

All operate per-target (one array of decoys for one complex); aggregate across targets by
averaging. Kept dependency-free (numpy only) — Spearman is Pearson on average-tied ranks.
"""

from __future__ import annotations

import numpy as np

__all__ = ["pearson", "spearman", "ranking_loss", "top_n_hit_rate", "top_n_success"]


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


def _top_indices(pred, n: int) -> np.ndarray:
    """Indices of the ``min(n, size)`` highest-scoring decoys (stable, ties by lower index)."""
    p = np.asarray(pred, dtype=float).ravel()
    k = min(n, p.size)
    return np.argsort(-p, kind="stable")[:k]


def top_n_hit_rate(pred, capri, n: int = 10, threshold: int = 1) -> float:
    """Fraction of the top-``n`` predicted decoys that are CAPRI-acceptable or better.

    ``capri`` is an integer CAPRI class per decoy (0 incorrect, 1 acceptable, 2 medium,
    3 high); ``threshold`` is the minimum class counted as a hit (default 1 = acceptable+).
    The denominator is ``min(n, num_decoys)`` — for a target with fewer than ``n`` decoys the
    rate is over what exists (use :func:`top_n_success` for the target-level success metric).
    """
    c = np.asarray(capri).ravel()
    if c.size == 0:
        return 0.0
    return float(np.mean(c[_top_indices(pred, n)] >= threshold))


def top_n_success(pred, capri, n: int = 10, threshold: int = 1) -> float:
    """1.0 if *any* of the top-``n`` predicted decoys is CAPRI ``threshold``-or-better, else 0.0.

    This is the per-target success indicator; averaging it across targets gives the
    top-N success rate reported by the TopoQA/DProQ protocol.
    """
    c = np.asarray(capri).ravel()
    if c.size == 0:
        return 0.0
    return float(np.any(c[_top_indices(pred, n)] >= threshold))
