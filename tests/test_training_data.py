"""MAF2 training-loader coverage — skipped unless the CC-BY MAF2 data is present.

MAF2 (Zenodo 6570843) is git-ignored (758 MB), so these run only where it has been extracted to
``zenodo/MAF2``. They exercise the label-tree walk + a single real featurization; the loader
reuses the benchmark's ``featurize_subset``, covered separately.
"""

import os
import shutil

import pytest

from open_topoqa_featurizer import NODE_WIDTH
from open_topoqa_scorer.training_data import load_maf2_labels

_MAF2 = os.path.join(os.path.dirname(__file__), "..", "zenodo", "MAF2")

pytestmark = [
    pytest.mark.skipif(
        not os.path.isdir(os.path.join(_MAF2, "label")),
        reason="MAF2 not extracted locally (git-ignored CC-BY data)",
    ),
    pytest.mark.skipif(shutil.which("mkdssp") is None, reason="mkdssp not on PATH"),
]


def test_load_maf2_labels_yields_valid_rows():
    labels = load_maf2_labels(_MAF2)
    assert labels, "no MAF2 models resolved on disk"
    for lab in labels[:50]:
        assert 0.0 <= lab.dockq <= 1.0
        assert lab.capri in (0, 1, 2, 3)
        assert os.path.exists(lab.pdb_path)
        # model stem maps directly to the PDB basename (no suffix glob, unlike the benchmark)
        assert os.path.basename(lab.pdb_path) == f"{lab.model}.pdb"


def test_maf2_has_both_af2_and_afmultimer_models():
    # each target carries the base AF2 prediction and the AF-Multimer one (~2 models/target)
    labels = load_maf2_labels(_MAF2)
    by_target: dict[str, set[str]] = {}
    for lab in labels:
        by_target.setdefault(lab.target, set()).add(lab.model)
    assert any(any(m.endswith("_af2m") for m in models) for models in by_target.values())


def test_single_real_maf2_decoy_featurizes():
    from open_topoqa_scorer.data import graph_from_complex

    lab = load_maf2_labels(_MAF2)[0]
    data = graph_from_complex(lab.pdb_path, y=lab.dockq)
    assert data.x.shape[1] == NODE_WIDTH
    assert data.x.shape[0] > 0
    assert 0.0 <= float(data.y) <= 1.0
