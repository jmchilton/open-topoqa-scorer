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


def _stub_graph_failing(monkeypatch, fail_paths, calls):
    """graph_from_complex stub that raises for pdb_paths in ``fail_paths``."""
    import torch
    from torch_geometric.data import Data

    def fake(pdb_path, y=None, dssp_map=None):
        calls.append(pdb_path)
        if pdb_path in fail_paths:
            raise RuntimeError(f"boom {pdb_path}")
        return Data(x=torch.zeros(1, 1), y=torch.tensor([float(y)]))

    monkeypatch.setattr(data_mod, "graph_from_complex", fake)


def test_failed_decoy_dropped_from_both_lists_aligned(tmp_path, monkeypatch):
    calls = []
    labels = _labels(4)  # /p0../p3.pdb
    _stub_graph_failing(monkeypatch, {"/p2.pdb"}, calls)
    g, k = featurize_subset(labels, cache_path=str(tmp_path / "c.pt"))
    assert len(g) == len(k) == 3  # the failure is gone from BOTH
    assert "/p2.pdb" not in {lab.pdb_path for lab in k}  # dropped, not silently mislabeled
    assert all(float(gi.y) == lab.dockq for gi, lab in zip(g, k))  # still index-aligned


def test_transient_all_failure_self_heals_on_rerun(tmp_path, monkeypatch):
    # run 1: everything fails (e.g. broken mkdssp). run 2 (fault cleared) must RETRY, not freeze
    # the failures into an empty cache and short-circuit to nothing.
    cache = str(tmp_path / "c.pt")
    labels = _labels(3)
    calls1 = []
    _stub_graph_failing(monkeypatch, {"/p0.pdb", "/p1.pdb", "/p2.pdb"}, calls1)
    g1, k1 = featurize_subset(labels, cache_path=cache)
    assert len(g1) == 0  # all dropped

    calls2 = []
    _stub_graph_failing(monkeypatch, set(), calls2)  # fault cleared
    g2, k2 = featurize_subset(labels, cache_path=cache)
    assert len(g2) == 3  # retried and recovered, not frozen at 0
    assert len(calls2) == 3
