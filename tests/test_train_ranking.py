"""Ranking-augmented training: batching plumbing (bites) + end-to-end wiring smoke (hermetic)."""

import random

import pytest
import torch

from open_topoqa_scorer.data import graph_from_featurized
from open_topoqa_scorer.train import _target_batches, predict, train_model


def test_target_batches_keep_whole_targets_and_cover_all():
    tidx = [0, 0, 0, 1, 1, 2, 2, 2, 2]  # 3 targets of sizes 3,2,4
    batches = _target_batches(tidx, batch_size=4, rng=random.Random(0))
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(len(tidx)))  # every graph exactly once, none dropped/dup
    for b in batches:
        for t in {tidx[i] for i in b}:  # no target is split across batches
            assert all(i in b for i, tt in enumerate(tidx) if tt == t)


def test_oversized_target_becomes_its_own_batch():
    tidx = [0, 0, 0, 0, 0, 1]  # target 0 has 5 > batch_size
    batches = _target_batches(tidx, batch_size=3, rng=random.Random(0))
    assert [0, 1, 2, 3, 4] in batches  # not split despite exceeding batch_size


def test_ranking_weight_requires_targets(feat_dict):
    g = [graph_from_featurized(feat_dict(3, [(0, 1)], seed=1), y=0.5)]
    with pytest.raises(ValueError, match="train_targets"):
        train_model(g, epochs=1, ranking_weight=1.0)


def test_ranking_training_runs_and_orders_a_tiny_set(feat_dict):
    # end-to-end wiring: target-grouped batching + combined loss + val path all run, and a small
    # memorizable set gets ordered within each target. (Whether ranking *beats* MSE on the real
    # corpus is an empirical experiment, not asserted here.)
    graphs, targets = [], []
    for t in range(4):
        for j, y in enumerate([0.15, 0.85]):
            graphs.append(graph_from_featurized(feat_dict(3, [(0, 1)], seed=t * 7 + j), y=y))
            targets.append(f"T{t}")
    model, history = train_model(
        graphs, graphs, epochs=250, lr=0.01, batch_size=8, seed=0,
        ranking_weight=2.0, ranking_margin=0.2,
        train_targets=targets, val_targets=targets,
        model_kwargs={"hidden": 16, "heads": 2, "num_layers": 1},
    )
    assert "best_epoch" in history and all(map(torch.isfinite, map(torch.tensor, history["val_loss"])))
    scores = predict(model, graphs).view(-1).tolist()
    for t in range(4):  # the true-0.85 decoy must outscore the true-0.15 decoy of the same target
        assert scores[2 * t + 1] > scores[2 * t]
