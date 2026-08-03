"""Corpus assembly: the merge concatenates both sources; capri_counts histograms (hermetic)."""

import pytest

from open_topoqa_scorer import corpus as C
from open_topoqa_scorer.benchmark import DecoyLabel


def _lab(target, capri, model="m"):
    return DecoyLabel(target=target, model=model, dockq=0.5, capri=capri, pdb_path="")


def test_is_dockground_target_distinguishes_namespaces():
    assert C.is_dockground_target("1cho")  # lowercase 4-char PDB id
    assert C.is_dockground_target("2bkr")
    assert not C.is_dockground_target("3KQ4")  # MAF2 dir stem (uppercase)
    assert not C.is_dockground_target("3KQ4_af2m")


def test_subset_for_ranking_keeps_all_dg_targets_caps_and_samples_maf2():
    labels = (
        [_lab("1abc", c, f"d{i}") for i, c in enumerate([0, 0, 0, 1, 2, 3])]  # DG, 6 decoys
        + [_lab("2xyz", c, f"d{i}") for i, c in enumerate([0, 0, 3])]          # DG, 3 decoys
        + [_lab(f"MAF{t}", 2, f"m{j}") for t in range(10) for j in range(2)]   # 10 MAF2 targets x2
    )
    s = C.subset_for_ranking(labels, dg_cap=4, maf2_n_targets=3, seed=0)
    from collections import Counter as Ct
    per = Ct(l.target for l in s)
    assert per["1abc"] == 4 and per["2xyz"] == 3  # every DG target kept, capped
    maf2_targets = {l.target for l in s if l.target.startswith("MAF")}
    assert len(maf2_targets) == 3  # only the sampled MAF2 targets
    assert all(per[t] == 2 for t in maf2_targets)  # kept whole (both models)


def test_capri_counts_histograms_classes():
    labels = [_lab("A", 0), _lab("B", 0), _lab("C", 2), _lab("D", 3)]
    assert C.capri_counts(labels) == {0: 2, 2: 1, 3: 1}


def test_load_combined_concatenates_both_sources(monkeypatch):
    monkeypatch.setattr(C, "load_maf2_labels", lambda d, require_files=True: [_lab("maf", 0)])
    monkeypatch.setattr(
        C, "load_dockground_labels", lambda t, e, require_files=True: [_lab("dg", 1), _lab("dg", 2)]
    )
    both = C.load_combined_labels("maf2", "tgz", "extract")
    assert {lab.target for lab in both} == {"maf", "dg"}
    assert len(both) == 3


def test_dockground_extract_dir_required(monkeypatch):
    monkeypatch.setattr(C, "load_maf2_labels", lambda d, require_files=True: [])
    with pytest.raises(ValueError):
        C.load_combined_labels(maf2_dir=None, dockground_tgz_dir="tgz")  # no extract dir


def test_target_id_collision_between_sources_raises(monkeypatch):
    # same target string in both corpora -> would fuse two complexes; must be rejected loudly
    monkeypatch.setattr(C, "load_maf2_labels", lambda d, require_files=True: [_lab("1abc", 0)])
    monkeypatch.setattr(
        C, "load_dockground_labels", lambda t, e, require_files=True: [_lab("1abc", 1)]
    )
    with pytest.raises(ValueError, match="share"):
        C.load_combined_labels("maf2", "tgz", "extract")
