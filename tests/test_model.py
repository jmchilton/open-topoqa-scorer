import pytest
import torch
from torch_geometric.loader import DataLoader

from open_topoqa_scorer.data import graph_from_featurized
from open_topoqa_scorer.model import ProteinGAT


@pytest.mark.parametrize("conv", ["transformer", "gatv2"])
def test_forward_scalar_in_unit_interval(small_graph_feat, conv):
    model = ProteinGAT(conv=conv).eval()
    out = model(graph_from_featurized(small_graph_feat))
    assert out.shape == (1,)
    assert torch.all((out >= 0) & (out <= 1))


def test_batched_forward_one_score_per_graph(feat_dict):
    graphs = [
        graph_from_featurized(feat_dict(4, [(0, 1), (1, 2), (2, 3)], seed=1), y=0.3),
        graph_from_featurized(feat_dict(6, [(0, 5), (1, 2)], seed=3), y=0.7),
        graph_from_featurized(feat_dict(2, [(0, 1)], seed=4), y=0.5),
    ]
    batch = next(iter(DataLoader(graphs, batch_size=3)))
    out = ProteinGAT().eval()(batch)
    assert out.shape == (3,)


def test_single_node_no_edge_graph(feat_dict):
    # a node with no cross-chain contacts must still score (edge branch falls back to zeros)
    model = ProteinGAT().eval()
    out = model(graph_from_featurized(feat_dict(1, [], seed=5)))
    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_gradients_flow(small_graph_feat):
    model = ProteinGAT().train()
    out = model(graph_from_featurized(small_graph_feat, y=0.4))
    loss = torch.nn.functional.mse_loss(out, torch.tensor([0.4]))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_bad_conv_rejected():
    with pytest.raises(ValueError):
        ProteinGAT(conv="nope")
