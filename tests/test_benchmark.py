"""Loader coverage against the real CC-BY benchmark — skipped unless the data is present.

The DProQ benchmark (Zenodo 6569837) is git-ignored (too large to vendor), so these run only
where it has been extracted locally. They exercise the loader + a single real featurization;
the full pipeline proof is ``scripts/phase_b_smoke.py``.
"""

import os
import shutil

import pytest

from open_topoqa_featurizer import NODE_WIDTH
from open_topoqa_scorer.benchmark import load_labels, resolve_decoy_path

_SUBSET = os.path.join(
    os.path.dirname(__file__), "..", "zenodo", "DProQ_benchmark", "BM55-AF2"
)

pytestmark = [
    pytest.mark.skipif(
        not os.path.exists(os.path.join(_SUBSET, "label_info.csv")),
        reason="DProQ benchmark not extracted locally (git-ignored CC-BY data)",
    ),
    pytest.mark.skipif(shutil.which("mkdssp") is None, reason="mkdssp not on PATH"),
]


def test_load_labels_yields_valid_rows():
    labels = load_labels(_SUBSET)
    assert labels, "no decoys resolved on disk"
    for lab in labels[:50]:
        assert 0.0 <= lab.dockq <= 1.0
        assert lab.capri in (0, 1, 2, 3)
        assert os.path.exists(lab.pdb_path)


def test_resolve_decoy_path_handles_tidy_suffix():
    labels = load_labels(_SUBSET)
    lab = labels[0]
    resolved = resolve_decoy_path(_SUBSET, lab.target, lab.model)
    assert resolved == lab.pdb_path
    # the file carries a suffix beyond the CSV model name (BM55-AF2 uses _tidy)
    assert os.path.basename(resolved).startswith(lab.model)


@pytest.mark.parametrize("nested", [False, True])
def test_resolve_decoy_path_finds_both_layouts(tmp_path, nested):
    # BM55-AF2 stores decoy/<TARGET>/<MODEL>.pdb; HAF2 nests it under a pdb/ subfolder.
    base = tmp_path / "decoy" / "T1"
    (base / "pdb" if nested else base).mkdir(parents=True)
    dst = (base / "pdb" if nested else base) / "modelA.pdb"
    dst.write_text("ATOM\n")
    resolved = resolve_decoy_path(str(tmp_path), "T1", "modelA")
    assert resolved == str(dst)


def test_single_real_decoy_featurizes():
    from open_topoqa_scorer.data import graph_from_complex

    lab = load_labels(_SUBSET)[0]
    data = graph_from_complex(lab.pdb_path, y=lab.dockq)
    assert data.x.shape[1] == NODE_WIDTH
    assert data.x.shape[0] > 0  # a real interface has residues
    assert 0.0 <= float(data.y) <= 1.0
