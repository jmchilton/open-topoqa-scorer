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


def test_gradients_flow_including_edge_branch(small_graph_feat):
    model = ProteinGAT().train()
    out = model(graph_from_featurized(small_graph_feat, y=0.4))
    loss = torch.nn.functional.mse_loss(out, torch.tensor([0.4]))
    loss.backward()
    # gradient must reach BOTH the node encoder and the edge branch, not just "some param"
    assert model.node_encoder.weight.grad is not None
    assert torch.any(model.node_encoder.weight.grad != 0)
    assert model.edge_reducer.weight.grad is not None
    assert torch.any(model.edge_reducer.weight.grad != 0)


def test_edge_features_change_the_score(small_graph_feat):
    # the edge branch must actually feed the head: perturbing only edge_attr moves the output
    model = ProteinGAT().eval()
    data = graph_from_featurized(small_graph_feat)
    base = model(data).clone()
    data.edge_attr = data.edge_attr + 3.0
    perturbed = model(data)
    assert not torch.allclose(base, perturbed)


def test_batched_matches_per_graph_with_uneven_edges(feat_dict):
    # a mismatched edge->graph pooling map would break batched vs. solo agreement;
    # include graphs with different edge counts and a zero-edge graph
    specs = [
        (5, [(0, 1), (1, 2), (2, 3), (3, 4)], 10),
        (3, [], 11),  # zero-edge graph exercises the edge_pool fallback
        (4, [(0, 3)], 12),
    ]
    graphs = [graph_from_featurized(feat_dict(n, e, seed=s)) for n, e, s in specs]
    model = ProteinGAT().eval()
    solo = torch.cat([model(g) for g in graphs])
    batched = model(next(iter(DataLoader(graphs, batch_size=len(graphs)))))
    assert torch.allclose(solo, batched, atol=1e-5)


def test_hyperparameters_match_paper_spec():
    model = ProteinGAT()
    assert len(model.convs) == 2
    assert model.convs[0].heads == 8
    assert model.convs[0].out_channels == 32  # hidden width, concat=False
    assert model.convs[0].concat is False
    assert abs(model.dropout - 0.25) < 1e-9

    import inspect

    from open_topoqa_scorer.train import train_model

    sig = inspect.signature(train_model)
    assert sig.parameters["lr"].default == 0.005
    assert sig.parameters["epochs"].default == 200


def test_bad_conv_rejected():
    with pytest.raises(ValueError):
        ProteinGAT(conv="nope")
