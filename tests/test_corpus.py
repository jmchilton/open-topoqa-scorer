"""Corpus assembly: the merge concatenates both sources; capri_counts histograms (hermetic)."""

import pytest

from open_topoqa_scorer import corpus as C
from open_topoqa_scorer.benchmark import DecoyLabel


def _lab(target, capri):
    return DecoyLabel(target=target, model="m", dockq=0.5, capri=capri, pdb_path="")


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
