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


def featurize_subset(
    labels, cache_path: str | None = None, progress: bool = False, checkpoint_every: int = 50
):
    """Featurize ``labels`` into PyG graphs (y = DockQ), one per decoy.

    Returns ``(graphs, kept_labels)`` — decoys that fail featurization are dropped from both,
    keeping the two lists aligned. Results are cached to ``cache_path`` (torch ``.pt``) when
    given, so re-runs skip the expensive mkdssp + persistent-homology pass.

    The cache is written incrementally (every ``checkpoint_every`` decoys, plus at the end), and
    the ``(target, model)`` of every *attempted* decoy is recorded — so an interrupted run resumes
    from its last checkpoint, re-featurizing only the decoys it never reached. A cache that already
    covers every requested decoy short-circuits the whole pass.
    """
    import torch

    from open_topoqa_scorer.data import graph_from_complex

    graphs, kept, done = [], [], set()
    if cache_path and os.path.exists(cache_path):
        blob = torch.load(cache_path, weights_only=False)
        graphs, kept = blob["graphs"], blob["labels"]
        # legacy caches lack "done"; fall back to the kept labels (re-attempts old failures once)
        done = set(map(tuple, blob.get("done", [(l.target, l.model) for l in kept])))
        if all((lab.target, lab.model) in done for lab in labels):
            return graphs, kept

    def _save():
        if cache_path:
            torch.save({"graphs": graphs, "labels": kept, "done": sorted(done)}, cache_path)

    since_save = 0
    for i, lab in enumerate(labels):
        if (lab.target, lab.model) in done:
            continue
        if progress:
            print(f"[{i + 1}/{len(labels)}] {lab.target}/{lab.model}", flush=True)
        try:
            graphs.append(graph_from_complex(lab.pdb_path, y=lab.dockq))
            kept.append(lab)
        except Exception as exc:  # noqa: BLE001 — a bad decoy shouldn't sink the batch
            if progress:
                print(f"  skipped ({exc})", flush=True)
        done.add((lab.target, lab.model))
        since_save += 1
        if since_save >= checkpoint_every:
            _save()
            since_save = 0

    _save()
    return graphs, kept
