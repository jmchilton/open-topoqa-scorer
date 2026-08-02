import numpy as np
import torch

from open_topoqa_featurizer import EDGE_WIDTH, NODE_WIDTH
from open_topoqa_scorer.data import graph_from_featurized


def test_shapes_and_target(small_graph_feat):
    data = graph_from_featurized(small_graph_feat, y=0.42)
    assert data.x.shape == (4, NODE_WIDTH)
    assert data.x.dtype == torch.float32
    # 3 undirected contacts -> 6 directed edges
    assert data.edge_index.shape == (2, 6)
    assert data.edge_attr.shape == (6, EDGE_WIDTH)
    assert torch.allclose(data.y, torch.tensor([0.42]))


def test_symmetrization_shares_edge_attr(small_graph_feat):
    data = graph_from_featurized(small_graph_feat)
    ei = data.edge_index.numpy()
    # every (i,j) has a matching (j,i)
    fwd = set(map(tuple, ei.T[:3]))
    rev = set((b, a) for a, b in ei.T[3:])
    assert fwd == rev
    # the reversed half reuses the same 11-dim attrs as the forward half
    assert torch.allclose(data.edge_attr[:3], data.edge_attr[3:])


def test_no_target_leaves_y_unset(small_graph_feat):
    data = graph_from_featurized(small_graph_feat)
    assert "y" not in data


def test_empty_edge_graph(feat_dict):
    data = graph_from_featurized(feat_dict(3, [], seed=2))
    assert data.x.shape == (3, NODE_WIDTH)
    assert data.edge_index.shape == (2, 0)
    assert data.edge_attr.shape == (0, EDGE_WIDTH)
