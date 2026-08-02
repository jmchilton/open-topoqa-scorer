"""Loader for the DProQ benchmark (Zenodo 6569837, CC-BY-4.0) — Phase B pipeline proof.

Layout per subset (``HAF2``, ``BM55-AF2``)::

    <subset>/label_info.csv        # columns: Target, Model, DockQ, CAPRI
    <subset>/native/<TARGET>.pdb
    <subset>/decoy/<TARGET>/<MODEL>[...suffix].pdb

The decoy filename is the CSV ``Model`` plus an optional suffix (BM55-AF2 appends
``_tidy``), so paths are resolved by trying the bare name, the ``_tidy`` name, then a glob.
This data is licensed CC-BY-4.0 but too large to vendor — it is git-ignored and pulled locally.
"""

from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass

__all__ = ["DecoyLabel", "load_labels", "resolve_decoy_path", "featurize_subset"]


@dataclass(frozen=True)
class DecoyLabel:
    target: str
    model: str
    dockq: float
    capri: int
    pdb_path: str


def resolve_decoy_path(subset_dir: str, target: str, model: str) -> str | None:
    """Path to a decoy PDB, tolerating the ``_tidy`` (and other) filename suffixes."""
    base = os.path.join(subset_dir, "decoy", target)
    for cand in (f"{model}.pdb", f"{model}_tidy.pdb"):
        p = os.path.join(base, cand)
        if os.path.exists(p):
            return p
    hits = sorted(glob.glob(os.path.join(base, f"{model}*.pdb")))
    return hits[0] if hits else None


def load_labels(subset_dir: str, require_files: bool = True) -> list[DecoyLabel]:
    """Parse ``label_info.csv`` into ``DecoyLabel`` rows.

    With ``require_files`` (default), rows whose decoy PDB is not on disk are skipped —
    so a partial extraction of the tarball still yields a coherent, featurizable subset.
    """
    csv_path = os.path.join(subset_dir, "label_info.csv")
    out: list[DecoyLabel] = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            target, model = row["Target"], row["Model"]
            path = resolve_decoy_path(subset_dir, target, model)
            if path is None:
                if require_files:
                    continue
                path = ""
            out.append(
                DecoyLabel(
                    target=target,
                    model=model,
                    dockq=float(row["DockQ"]),
                    capri=int(float(row["CAPRI"])),
                    pdb_path=path,
                )
            )
    return out


def featurize_subset(labels, cache_path: str | None = None, progress: bool = False):
    """Featurize ``labels`` into PyG graphs (y = DockQ), one per decoy.

    Returns ``(graphs, kept_labels)`` — decoys that fail featurization are dropped from both,
    keeping the two lists aligned. Results are cached to ``cache_path`` (torch ``.pt``) when
    given, so re-runs skip the expensive mkdssp + persistent-homology pass.
    """
    import torch

    from open_topoqa_scorer.data import graph_from_complex

    if cache_path and os.path.exists(cache_path):
        blob = torch.load(cache_path, weights_only=False)
        return blob["graphs"], blob["labels"]

    graphs, kept = [], []
    for i, lab in enumerate(labels):
        if progress:
            print(f"[{i + 1}/{len(labels)}] {lab.target}/{lab.model}", flush=True)
        try:
            graphs.append(graph_from_complex(lab.pdb_path, y=lab.dockq))
            kept.append(lab)
        except Exception as exc:  # noqa: BLE001 — a bad decoy shouldn't sink the batch
            if progress:
                print(f"  skipped ({exc})", flush=True)

    if cache_path:
        torch.save({"graphs": graphs, "labels": kept}, cache_path)
    return graphs, kept
