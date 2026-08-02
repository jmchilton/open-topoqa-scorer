"""Dockground loader: hermetic DockQ-from-CAPRI math + a real single-target load."""

import os
import shutil

import pytest

from open_topoqa_featurizer import NODE_WIDTH
from open_topoqa_scorer.dockground import (
    dockq_from_capri,
    dockq_to_capri_class,
    load_dockground_labels,
)


def test_dockq_from_capri_endpoints_and_midpoint():
    assert dockq_from_capri(1.0, 0.0, 0.0) == pytest.approx(1.0)  # perfect
    assert dockq_from_capri(0.0, 1e6, 1e6) == pytest.approx(0.0, abs=1e-6)  # hopeless
    # fnat=1, iRMSD=1.5 (half), LRMSD=8.5 (half) -> (1 + 0.5 + 0.5)/3
    assert dockq_from_capri(1.0, 1.5, 8.5) == pytest.approx(2.0 / 3.0)


def test_dockq_to_capri_class_boundaries():
    assert dockq_to_capri_class(0.22) == 0
    assert dockq_to_capri_class(0.23) == 1
    assert dockq_to_capri_class(0.48) == 1
    assert dockq_to_capri_class(0.49) == 2
    assert dockq_to_capri_class(0.79) == 2
    assert dockq_to_capri_class(0.80) == 3


_TGZ_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "dockground", "extracted_tgz", "decoy", "decoy"
)


@pytest.mark.skipif(
    not os.path.isdir(_TGZ_DIR), reason="Dockground not extracted locally (git-ignored public data)"
)
def test_load_one_dockground_target(tmp_path):
    # copy a single archive so the untar stays fast
    src = sorted(f for f in os.listdir(_TGZ_DIR) if f.endswith(".tgz"))[0]
    tgz_dir = tmp_path / "tgz"
    tgz_dir.mkdir()
    shutil.copy(os.path.join(_TGZ_DIR, src), tgz_dir)

    labels = load_dockground_labels(str(tgz_dir), str(tmp_path / "ex"))
    assert labels, "no decoys loaded"
    assert len({l.target for l in labels}) == 1
    for lab in labels:
        assert 0.0 <= lab.dockq <= 1.0
        assert lab.capri in (0, 1, 2, 3)
        assert lab.capri == dockq_to_capri_class(lab.dockq)
        assert os.path.exists(lab.pdb_path)
    # a docking-decoy set is mostly incorrect with a few near-native — spread, not all-equal
    assert len({lab.capri for lab in labels}) > 1


@pytest.mark.skipif(
    not os.path.isdir(_TGZ_DIR), reason="Dockground not extracted locally (git-ignored public data)"
)
@pytest.mark.skipif(shutil.which("mkdssp") is None, reason="mkdssp not on PATH")
def test_single_real_dockground_decoy_featurizes(tmp_path):
    from open_topoqa_scorer.data import graph_from_complex

    src = sorted(f for f in os.listdir(_TGZ_DIR) if f.endswith(".tgz"))[0]
    tgz_dir = tmp_path / "tgz"
    tgz_dir.mkdir()
    shutil.copy(os.path.join(_TGZ_DIR, src), tgz_dir)
    lab = load_dockground_labels(str(tgz_dir), str(tmp_path / "ex"))[0]

    data = graph_from_complex(lab.pdb_path, y=lab.dockq)
    assert data.x.shape[1] == NODE_WIDTH
    assert data.x.shape[0] > 0
    assert 0.0 <= float(data.y) <= 1.0
