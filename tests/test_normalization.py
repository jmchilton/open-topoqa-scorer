"""Input standardization: buffers fit, travel in state_dict, and reshape large-feature forward."""

import torch
from torch_geometric.loader import DataLoader

from open_topoqa_scorer.data import feature_stats, graph_from_featurized
from open_topoqa_scorer.model import ProteinGAT


def test_unfitted_model_is_identity_standardization():
    m = ProteinGAT()
    assert torch.allclose(m.x_mean, torch.zeros_like(m.x_mean))
    assert torch.allclose(m.x_inv_std, torch.ones_like(m.x_inv_std))


def test_stats_are_fitted_and_survive_state_dict_roundtrip(feat_dict):
    graphs = [graph_from_featurized(feat_dict(4, [(0, 1), (1, 2)], seed=s), y=0.3) for s in range(6)]
    m = ProteinGAT().eval()
    m.set_feature_stats(*feature_stats(graphs))
    assert not torch.allclose(m.x_mean, torch.zeros_like(m.x_mean))  # actually fitted

    batch = next(iter(DataLoader(graphs, batch_size=6)))
    out1 = m(batch)
    reloaded = ProteinGAT().eval()
    reloaded.load_state_dict(m.state_dict())  # buffers must ride along or predictions diverge
    assert torch.allclose(out1, reloaded(batch))


def test_standardization_tames_a_huge_feature(feat_dict):
    d = feat_dict(4, [(0, 1), (1, 2)], seed=1)
    d["node_features"][:, 40] *= 500.0  # a topological feature blown up, as in the real data
    graph = graph_from_featurized(d, y=0.3)

    m = ProteinGAT().eval()
    raw = m(graph)
    m.set_feature_stats(*feature_stats([graph]))
    norm = m(graph)
    assert torch.isfinite(norm).all()
    assert not torch.allclose(raw, norm)  # standardization genuinely changes the mapping
