"""Canonical model red-to-green: the loop must drive a tiny batch's MSE toward zero.

If wiring (encoder → convs → pooled node/edge concat → head → sigmoid → MSE → backprop) is
correct, a small model memorizes ~8 graphs. A broken graph (e.g. all graphs pooling to the
same vector) cannot, so this exercises the whole path, not just shapes.
"""

import numpy as np
import torch
import torch.nn.functional as F

from open_topoqa_scorer.data import graph_from_featurized
from open_topoqa_scorer.train import predict, train_model


def _make_dataset(feat_dict):
    rng = np.random.default_rng(7)
    graphs = []
    for i in range(8):
        n = int(rng.integers(3, 7))
        edges = [(a, b) for a in range(n) for b in range(a + 1, n) if rng.random() < 0.5]
        if not edges:
            edges = [(0, 1)]
        y = float(rng.uniform(0.05, 0.95))
        graphs.append(graph_from_featurized(feat_dict(n, edges, seed=100 + i), y=y))
    return graphs


def test_overfit_tiny_batch(feat_dict):
    graphs = _make_dataset(feat_dict)
    # dropout off so the tiny batch can be memorized
    model, history = train_model(
        graphs,
        val_set=None,
        epochs=400,
        lr=0.01,
        batch_size=8,
        seed=0,
        model_kwargs={"dropout": 0.0},
    )
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["train_loss"][-1] < 0.01

    preds = predict(model, graphs)
    targets = torch.cat([g.y for g in graphs])
    assert F.mse_loss(preds, targets) < 0.01
