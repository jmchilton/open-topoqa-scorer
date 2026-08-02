"""Canonical model red-to-green: the loop must drive a tiny batch's MSE toward zero.

The plain overfit below proves the full path (encoder -> convs -> pooled node/edge concat ->
head -> sigmoid -> MSE -> backprop) can memorize a handful of graphs. It does NOT by itself
prove the node path is used — the graphs also differ in edge features, so an edge-only model
could pass. ``test_overfit_requires_node_path`` closes that gap: identical topology and edge
features across graphs, only node features vary, so memorization forces the node encoder +
message passing to carry the signal.
"""

import numpy as np
import torch
import torch.nn.functional as F

from open_topoqa_featurizer import EDGE_WIDTH, NODE_WIDTH
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


def _node_varying_graphs(n_graphs=8, n_nodes=4):
    """Graphs identical in topology AND edge features; only node features (and y) vary."""
    rng = np.random.default_rng(11)
    edges = np.array([(0, 1), (1, 2), (2, 3)], dtype=np.int64)
    fixed_edge_feats = np.ones((edges.shape[0], EDGE_WIDTH), dtype=np.float32)  # SAME for all
    graphs = []
    for i in range(n_graphs):
        feat = {
            "node_ids": [("A", (" ", k, " ")) for k in range(n_nodes)],
            "node_features": rng.standard_normal((n_nodes, NODE_WIDTH)).astype(np.float32),
            "edge_index": edges,
            "edge_features": fixed_edge_feats,
        }
        graphs.append(graph_from_featurized(feat, y=float(rng.uniform(0.05, 0.95))))
    return graphs


def test_overfit_tiny_batch(feat_dict):
    graphs = _make_dataset(feat_dict)
    model, history = train_model(
        graphs, val_set=None, epochs=400, lr=0.01, batch_size=8, seed=0,
        model_kwargs={"dropout": 0.0},
    )
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["train_loss"][-1] < 0.01

    preds = predict(model, graphs)
    targets = torch.cat([g.y for g in graphs])
    assert F.mse_loss(preds, targets) < 0.01


def test_overfit_requires_node_path():
    # edges + edge features are identical across graphs, so the ONLY way to separate them
    # (and drive MSE down) is through the node features -> pins the node encoder + convs.
    graphs = _node_varying_graphs()
    model, history = train_model(
        graphs, val_set=None, epochs=500, lr=0.01, batch_size=8, seed=0,
        model_kwargs={"dropout": 0.0},
    )
    preds = predict(model, graphs)
    targets = torch.cat([g.y for g in graphs])
    assert F.mse_loss(preds, targets) < 0.02
