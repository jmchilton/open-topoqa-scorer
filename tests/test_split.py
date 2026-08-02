"""Homology-aware split: the leak-proof invariant (hermetic) + a real mmseqs run on MAF2."""

import os
import shutil

import pytest

from open_topoqa_scorer import split as S
from open_topoqa_scorer.benchmark import DecoyLabel


def _labels(spec):
    """spec: {target: n_decoys} -> flat DecoyLabel list (pdb_path empty; clustering is mocked)."""
    return [
        DecoyLabel(target=t, model=f"{t}_{i}", dockq=0.5, capri=1, pdb_path="")
        for t, n in spec.items()
        for i in range(n)
    ]


def test_no_cluster_or_target_spans_train_val(monkeypatch):
    # A,B homologous (c1); D,E homologous (c3); C alone (c2). A leak-proof split must keep each
    # cluster wholly on one side -> also target-disjoint.
    labels = _labels({"A": 4, "B": 4, "C": 4, "D": 4, "E": 4})
    clusters = {"A": "c1", "B": "c1", "C": "c2", "D": "c3", "E": "c3"}
    monkeypatch.setattr(S, "cluster_targets", lambda *a, **k: clusters)

    train, val = S.split_by_clusters(labels, tmp_dir="/unused", val_frac=0.4, seed=1)
    train_t = {l.target for l in train}
    val_t = {l.target for l in val}
    assert train and val
    assert not (train_t & val_t), "target appears in both splits"
    assert not ({clusters[t] for t in train_t} & {clusters[t] for t in val_t}), "cluster spans split"
    # a target's decoys are never split across sides
    for t in train_t | val_t:
        sides = {("val" if l in val else "train") for l in labels if l.target == t}
        assert len(sides) == 1


def test_unclustered_targets_go_to_train(monkeypatch):
    # a target with no sequence (empty PDB) is absent from the cluster map -> must land in train,
    # never silently dropped.
    labels = _labels({"A": 2, "B": 2, "Z": 2})
    monkeypatch.setattr(S, "cluster_targets", lambda *a, **k: {"A": "c1", "B": "c2"})
    train, val = S.split_by_clusters(labels, tmp_dir="/unused", val_frac=0.5, seed=0)
    assert "Z" in {l.target for l in train}
    assert len(train) + len(val) == len(labels)  # nothing dropped


_MAF2 = os.path.join(os.path.dirname(__file__), "..", "zenodo", "MAF2")


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(_MAF2, "label")),
    reason="MAF2 not extracted locally (git-ignored CC-BY data)",
)
@pytest.mark.skipif(shutil.which("mmseqs") is None, reason="mmseqs not on PATH")
def test_real_maf2_cluster_split_is_leakproof(tmp_path):
    from open_topoqa_scorer.training_data import load_maf2_labels

    labels = load_maf2_labels(_MAF2)
    keep = sorted({l.target for l in labels})[:40]  # a slice keeps the mmseqs run fast
    subset = [l for l in labels if l.target in keep]
    train, val = S.split_by_clusters(subset, tmp_dir=str(tmp_path), val_frac=0.25, seed=0)
    assert train and val
    assert not ({l.target for l in train} & {l.target for l in val})
    assert len(train) + len(val) == len(subset)
