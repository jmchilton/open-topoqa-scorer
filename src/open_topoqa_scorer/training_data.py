"""Loader for the MAF2 training decoys (Zenodo 6570843, CC-BY-4.0) — Phase C.

MAF2 is the DProQA/TopoQA training corpus: AlphaFold2 + AF-Multimer complex predictions for
targets from the EVCoupling and DeepHomo sets. Unlike the DProQ *benchmark* (one CSV per subset),
MAF2 stores one label file per model::

    <maf2>/label/<TARGET>/<MODEL>.txt   # a single line: "<DockQ>,<CAPRI>", e.g. "0.026,0.0"
    <maf2>/pred/<TARGET>/<MODEL>.pdb

``<MODEL>`` is the file stem — the base AF2 prediction (``<TARGET>``) or the AF-Multimer one
(``<TARGET>_af2m``) — and maps directly to the PDB of the same name (no suffix glob needed). So
there are ~2 models per target: broad coverage, but shallow within-target ranking depth.

This reuses the benchmark's :class:`DecoyLabel` and :func:`featurize_subset` — only the label-tree
walk differs. The data is CC-BY-4.0 but too large to vendor: git-ignored, pulled locally.
"""

from __future__ import annotations

import os

from open_topoqa_scorer.benchmark import DecoyLabel, featurize_subset

__all__ = ["DecoyLabel", "featurize_subset", "load_maf2_labels"]


def _read_label(path: str) -> tuple[float, int]:
    """Parse a MAF2 label file (``"<DockQ>,<CAPRI>"``) into ``(dockq, capri_class)``."""
    with open(path) as fh:
        dockq_s, capri_s = fh.read().strip().split(",")
    return float(dockq_s), int(float(capri_s))


def load_maf2_labels(maf2_dir: str, require_files: bool = True) -> list[DecoyLabel]:
    """Walk the MAF2 ``label/``+``pred/`` tree into :class:`DecoyLabel` rows.

    With ``require_files`` (default), a model whose PDB is missing on disk is skipped, so a
    partial extraction still yields a coherent, featurizable subset (mirrors ``load_labels``).
    """
    label_root = os.path.join(maf2_dir, "label")
    pred_root = os.path.join(maf2_dir, "pred")
    out: list[DecoyLabel] = []
    for target in sorted(os.listdir(label_root)):
        tdir = os.path.join(label_root, target)
        if not os.path.isdir(tdir):
            continue
        for label_file in sorted(os.listdir(tdir)):
            if not label_file.endswith(".txt"):
                continue
            model = label_file[:-4]  # stem == PDB basename, e.g. "3KQ4" or "3KQ4_af2m"
            pdb_path = os.path.join(pred_root, target, f"{model}.pdb")
            if not os.path.exists(pdb_path):
                if require_files:
                    continue
                pdb_path = ""
            dockq, capri = _read_label(os.path.join(tdir, label_file))
            out.append(
                DecoyLabel(target=target, model=model, dockq=dockq, capri=capri, pdb_path=pdb_path)
            )
    return out
