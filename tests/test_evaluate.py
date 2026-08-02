"""Per-target eval aggregation: grouping, singleton exclusion, perfect-ranking sanity (hermetic)."""

import numpy as np
import pytest

from open_topoqa_scorer.benchmark import DecoyLabel
from open_topoqa_scorer.evaluate import per_target_ranking_metrics, pooled_regression_metrics


def _lab(target, dockq, capri):
    return DecoyLabel(target=target, model=f"{target}-{dockq}", dockq=dockq, capri=capri, pdb_path="")


def test_singletons_excluded_but_counted():
    labels = [_lab("A", 0.1, 0), _lab("A", 0.8, 3), _lab("SOLO", 0.5, 2)]
    scores = np.array([0.1, 0.8, 0.5])
    m = per_target_ranking_metrics(scores, labels)
    assert m["targets_total"] == 2  # A and SOLO both seen
    assert m["targets_ranked"] == 1  # only A has >=2 decoys


def test_perfect_ranking_scores_ideal():
    # two targets, predictions perfectly ordered vs true DockQ -> spearman 1, ranking loss 0
    labels = [
        _lab("A", 0.1, 0), _lab("A", 0.9, 3),
        _lab("B", 0.2, 1), _lab("B", 0.7, 2),
    ]
    scores = np.array([0.1, 0.9, 0.2, 0.7])  # == true order
    m = per_target_ranking_metrics(scores, labels)
    assert m["targets_ranked"] == 2
    assert m["spearman_mean"] == pytest.approx(1.0)
    assert m["ranking_loss_mean"] == 0.0
    assert m["top10_success_rate"] == 1.0  # each target has an acceptable+ decoy in top-10


def test_pooled_regression_matches_hand_calc():
    labels = [_lab("A", 0.0, 0), _lab("A", 1.0, 3)]
    m = pooled_regression_metrics(np.array([0.0, 1.0]), labels)
    assert m["mse"] == 0.0
    assert m["n"] == 2
    assert m["pearson"] == pytest.approx(1.0)
