"""Training-loop behavior: best-validation restore and the divergence guard.

These pin two things the forward/overfit tests don't: that the returned model carries the
best-validation weights (not the final epoch's), and that a diverged run fails loud instead
of silently returning an untrained initialization.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from open_topoqa_scorer.data import graph_from_featurized
from open_topoqa_scorer.train import predict, train_model


def _labeled(feat_dict, seed, n, y):
    edges = [(a, b) for a in range(n) for b in range(a + 1, n)]
    return graph_from_featurized(feat_dict(n, edges, seed=seed), y=y)


def test_returns_best_val_not_final_weights(feat_dict):
    # train targets low, val targets high -> as the model overfits train, val loss rises again,
    # so the best val epoch is NOT the last. The restored model must reproduce that minimum.
    train_set = [_labeled(feat_dict, 200 + i, 4, 0.15) for i in range(4)]
    val_set = [_labeled(feat_dict, 300 + i, 4, 0.85) for i in range(4)]

    model, history = train_model(
        train_set, val_set=val_set, epochs=300, lr=0.02, batch_size=4, seed=0,
        model_kwargs={"dropout": 0.0},
    )
    val_losses = history["val_loss"]
    assert "best_epoch" in history
    assert min(val_losses) < val_losses[-1], "test not discriminating: val never worsened"
    assert val_losses[history["best_epoch"]] == min(val_losses)

    # the returned model reproduces the BEST val loss, proving best_state was restored
    val_y = torch.cat([g.y for g in val_set])
    restored = float(F.mse_loss(predict(model, val_set), val_y))
    assert restored == pytest.approx(min(val_losses), abs=1e-5)
    # and it is strictly better than the final-epoch weights would have given
    assert restored < val_losses[-1]


def test_diverged_training_raises_not_silently_untrained(feat_dict):
    # an absurd LR drives loss to NaN; best_val stays inf -> must raise, not load the init
    train_set = [_labeled(feat_dict, 400 + i, 4, 0.3) for i in range(4)]
    val_set = [_labeled(feat_dict, 500 + i, 4, 0.6) for i in range(4)]
    with pytest.raises(RuntimeError, match="best-validation"):
        train_model(
            train_set, val_set=val_set, epochs=5, lr=1e8, batch_size=4, seed=0,
            model_kwargs={"dropout": 0.0},
        )
