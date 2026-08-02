import numpy as np
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


def test_batch_with_trailing_empty_graph_scores_all(feat_dict):
    # an out-of-contact decoy featurizes to 0 interface nodes; batched with it LAST, pooling must
    # still emit one score per graph (inferring size from batch.max()+1 would drop it -> shape (2,))
    graphs = [
        graph_from_featurized(feat_dict(4, [(0, 1), (1, 2)], seed=1), y=0.8),
        graph_from_featurized(feat_dict(3, [(0, 2)], seed=2), y=0.6),
        graph_from_featurized(feat_dict(0, [], seed=3), y=0.0),  # empty interface, DockQ≈0
    ]
    batch = next(iter(DataLoader(graphs, batch_size=3)))
    out = ProteinGAT().eval()(batch)
    assert out.shape == (3,)
    assert torch.isfinite(out).all()


def test_gradients_flow_including_edge_branch(small_graph_feat):
    model = ProteinGAT().train()
    out = model(graph_from_featurized(small_graph_feat, y=0.4))
    loss = torch.nn.functional.mse_loss(out, torch.tensor([0.4]))
    loss.backward()
    # gradient must reach the node encoder AND the full edge path (encode -> per-layer update
    # (Eq. 6) -> pooled projection), not just "some param somewhere"
    for name, param in [
        ("node_encoder", model.node_encoder.weight),
        ("edge_encoder", model.edge_encoder.weight),
        ("edge_update[0]", model.edge_updates[0].weight),
        ("edge_pool_proj", model.edge_pool_proj.weight),
    ]:
        assert param.grad is not None and torch.any(param.grad != 0), f"no gradient at {name}"


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


def test_paper_structural_invariants():
    # these ARE mandated by the paper (§5.2.5, Eqs 6-9), independent of tunable widths
    model = ProteinGAT(hidden=24, num_layers=3, edge_hidden=10)
    # an edge-update module per layer (Eq. 6)
    assert len(model.edge_updates) == len(model.convs) == 3
    # pooled edge dim is half the node dim (Eq. 8)
    assert model.edge_pool_dim == 12
    assert model.edge_pool_proj.out_features == 12
    # readout MLP has exactly three linear layers (Eq. 9)
    linears = [m for m in model.head if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 3
    # edge-conditioned attention: the conv consumes the edge embedding
    assert model.convs[0].edge_dim == 10


def test_config_defaults_are_our_tunable_choices():
    # NOT paper values (the paper states none) — pin them so config drift is caught
    model = ProteinGAT()
    assert len(model.convs) == 2
    assert model.convs[0].heads == 4
    assert model.convs[0].out_channels == 32  # hidden width, concat=False
    assert model.convs[0].concat is False
    assert abs(model.dropout - 0.25) < 1e-9

    import inspect

    from open_topoqa_scorer.train import train_model

    sig = inspect.signature(train_model)
    assert sig.parameters["lr"].default == 0.005
    assert sig.parameters["epochs"].default == 200


def test_edge_update_feeds_readout_uses_updated_not_initial_edges(feat_dict):
    # The commit's headline behavior: the readout pools the *updated* edges (Eq. 6 -> 8), not the
    # initial edge encoding. With a SINGLE layer the post-conv edge update feeds ONLY the readout
    # (no later conv consumes it), so zeroing it must move the score. If the readout wrongly pooled
    # the initial encoding, edge_updates[0] would be dead and the score would not change.
    model = ProteinGAT(num_layers=1).eval()
    data = graph_from_featurized(feat_dict(4, [(0, 1), (1, 2), (2, 3)], seed=2))
    base = model(data).clone()
    with torch.no_grad():
        model.edge_updates[0].weight.zero_()
        model.edge_updates[0].bias.zero_()
    assert not torch.allclose(base, model(data))


def test_attention_consumes_edges_independent_of_readout_branch(small_graph_feat):
    # Zero the readout edge branch so edges can influence the score ONLY through attention (Eq. 3).
    # Perturbing edge_attr must still move the output -> the conv genuinely consumes edge features
    # (kills a mutation that drops the edge embedding fed into attention).
    model = ProteinGAT().eval()
    with torch.no_grad():
        model.edge_pool_proj.weight.zero_()
        model.edge_pool_proj.bias.zero_()
    data = graph_from_featurized(small_graph_feat)
    base = model(data).clone()
    # perturb a SINGLE directed edge (not a uniform shift, which softmax would wash out) so the
    # attention distribution genuinely changes; the only route left to the score is the conv.
    data.edge_attr[0] = data.edge_attr[0] + 5.0
    assert not torch.allclose(base, model(data))


def test_mean_pooling_invariant_to_duplication(feat_dict):
    # Eqs 7-8 pool with 1/|V| and 1/|E| (MEAN). Two disjoint copies of the same graph leave every
    # mean-pooled quantity unchanged, so the score must not move; global_add_pool would double the
    # pooled node/edge vectors and change it. Pins mean- (not sum-) pooling.
    base = feat_dict(3, [(0, 1), (1, 2)], seed=7)
    n = base["node_features"].shape[0]
    dup = {
        "node_ids": base["node_ids"] + [("B", (" ", i, " ")) for i in range(n)],
        "node_features": np.concatenate([base["node_features"], base["node_features"]], axis=0),
        "edge_index": np.concatenate([base["edge_index"], base["edge_index"] + n], axis=0),
        "edge_features": np.concatenate([base["edge_features"], base["edge_features"]], axis=0),
    }
    model = ProteinGAT().eval()
    one = model(graph_from_featurized(base))
    two = model(graph_from_featurized(dup))
    assert torch.allclose(one, two, atol=1e-5)


def test_bad_conv_rejected():
    with pytest.raises(ValueError):
        ProteinGAT(conv="nope")
