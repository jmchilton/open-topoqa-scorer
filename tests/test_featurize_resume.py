"""featurize_subset checkpoints incrementally and resumes without recomputing (hermetic)."""

import open_topoqa_scorer.data as data_mod
from open_topoqa_scorer.benchmark import DecoyLabel, featurize_subset


def _stub_graph(monkeypatch, calls):
    import torch
    from torch_geometric.data import Data

    def fake(pdb_path, y=None, dssp_map=None):
        calls.append(pdb_path)
        return Data(x=torch.zeros(1, 1), y=torch.tensor([float(y)]))

    monkeypatch.setattr(data_mod, "graph_from_complex", fake)


def _labels(n):
    return [DecoyLabel("T", f"m{i}", 0.5, 1, f"/p{i}.pdb") for i in range(n)]


def test_resume_only_featurizes_the_unreached(tmp_path, monkeypatch):
    calls = []
    _stub_graph(monkeypatch, calls)
    cache = str(tmp_path / "c.pt")
    labels = _labels(4)

    g1, k1 = featurize_subset(labels[:2], cache_path=cache, checkpoint_every=1)
    assert len(g1) == len(k1) == 2
    assert len(calls) == 2

    calls.clear()
    g2, k2 = featurize_subset(labels, cache_path=cache, checkpoint_every=1)
    assert len(g2) == len(k2) == 4  # the 2 new ones were added on top of the cached 2
    assert len(calls) == 2  # ...and only the 2 new ones actually recomputed


def test_full_cache_short_circuits(tmp_path, monkeypatch):
    calls = []
    _stub_graph(monkeypatch, calls)
    cache = str(tmp_path / "c.pt")
    labels = _labels(3)

    featurize_subset(labels, cache_path=cache, checkpoint_every=1)
    assert len(calls) == 3
    calls.clear()
    g, k = featurize_subset(labels, cache_path=cache)
    assert len(g) == 3
    assert calls == []  # nothing recomputed
