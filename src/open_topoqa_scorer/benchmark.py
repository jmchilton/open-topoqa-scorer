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

__all__ = [
    "DecoyLabel",
    "load_labels",
    "resolve_decoy_path",
    "featurize_subset",
    "featurize_subset_parallel",
    "restrict_to_labels",
]


@dataclass(frozen=True)
class DecoyLabel:
    target: str
    model: str
    dockq: float
    capri: int
    pdb_path: str


def resolve_decoy_path(subset_dir: str, target: str, model: str) -> str | None:
    """Path to a decoy PDB, tolerating the ``_tidy`` (and other) filename suffixes.

    Decoys sit directly under ``decoy/<TARGET>/`` (BM55-AF2) or one level down in a
    ``pdb/`` subfolder (HAF2); both layouts are searched.
    """
    for base in (
        os.path.join(subset_dir, "decoy", target),
        os.path.join(subset_dir, "decoy", target, "pdb"),
    ):
        for cand in (f"{model}.pdb", f"{model}_tidy.pdb"):
            p = os.path.join(base, cand)
            if os.path.exists(p):
                return p
        hits = sorted(glob.glob(os.path.join(base, f"{model}*.pdb")))
        if hits:
            return hits[0]
    return None


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


def restrict_to_labels(graphs, kept, labels):
    """Filter an index-aligned ``(graphs, kept)`` pair down to the ``(target, model)`` keys in
    ``labels``, preserving order.

    :func:`featurize_subset` short-circuits on a complete cache by returning that cache's own
    (possibly *superset*) contents, so a caller that asked for a strict subset — e.g. after
    excluding a target — must apply this or the exclusion silently has no effect.
    """
    wanted = {(lab.target, lab.model) for lab in labels}
    pairs = [(g, lab) for g, lab in zip(graphs, kept) if (lab.target, lab.model) in wanted]
    if not pairs:
        return [], []
    g, k = zip(*pairs)
    return list(g), list(k)


def featurize_subset(
    labels, cache_path: str | None = None, progress: bool = False, checkpoint_every: int = 50
):
    """Featurize ``labels`` into PyG graphs (y = DockQ), one per decoy.

    Returns ``(graphs, kept_labels)`` — decoys that fail featurization are dropped from both,
    keeping the two lists index-aligned. Results are cached to ``cache_path`` (torch ``.pt``) when
    given, so re-runs skip the expensive mkdssp + persistent-homology pass.

    The skip set is derived from the *successfully kept* decoys only, so the cache is written
    incrementally (every ``checkpoint_every`` decoys, plus at the end) yet a re-run **re-attempts
    every decoy that failed or was never reached** — a run that dropped decoys to a transient fault
    (broken mkdssp, OOM) self-heals on the next run rather than freezing the failure into the cache.
    A cache that already covers every requested decoy short-circuits the whole pass. Note the cache
    is bound to the label set it was built from: reusing one ``cache_path`` with a *different* label
    set returns the cache's own (possibly superset) contents, not a set filtered to ``labels``.
    """
    import torch

    from open_topoqa_scorer.data import graph_from_complex

    graphs, kept = [], []
    if cache_path and os.path.exists(cache_path):
        blob = torch.load(cache_path, weights_only=False)
        graphs, kept = blob["graphs"], blob["labels"]
    kept_keys = {(lab.target, lab.model) for lab in kept}  # skip only what actually succeeded
    if all((lab.target, lab.model) in kept_keys for lab in labels):
        return graphs, kept

    def _save():
        if cache_path:
            torch.save({"graphs": graphs, "labels": kept, "done": sorted(kept_keys)}, cache_path)

    attempted = failed = since_save = 0
    for i, lab in enumerate(labels):
        key = (lab.target, lab.model)
        if key in kept_keys:
            continue
        attempted += 1
        if progress:
            print(f"[{i + 1}/{len(labels)}] {lab.target}/{lab.model}", flush=True)
        try:
            graphs.append(graph_from_complex(lab.pdb_path, y=lab.dockq))
            kept.append(lab)
            kept_keys.add(key)
        except Exception as exc:  # noqa: BLE001 — a bad decoy shouldn't sink the batch
            failed += 1
            if progress:
                print(f"  skipped ({exc})", flush=True)
        since_save += 1
        if since_save >= checkpoint_every:
            _save()
            since_save = 0

    _save()
    if progress:
        print(
            f"featurize_subset: kept {len(kept)}/{len(labels)} "
            f"({attempted} attempted this run, {failed} failed)",
            flush=True,
        )
    return graphs, kept


def _featurize_one(lab):
    """Worker: featurize a single decoy. Top-level + import-inside so it pickles to a subprocess.

    Returns ``(lab, graph)`` on success or ``(lab, None)`` on any failure — a bad decoy must not
    crash the pool.
    """
    from open_topoqa_scorer.data import graph_from_complex

    try:
        return lab, graph_from_complex(lab.pdb_path, y=lab.dockq)
    except Exception:  # noqa: BLE001 — a bad decoy shouldn't sink the batch
        return lab, None


def featurize_subset_parallel(
    labels, cache_path: str | None = None, jobs: int = 0, progress: bool = False,
    checkpoint_every: int = 200,
):
    """Parallel :func:`featurize_subset` — same cache/resume/superset semantics, many processes.

    Featurization is CPU-bound and independent per decoy (mkdssp writes to unique temp files, so
    it is process-safe), so it fans out over a ``ProcessPoolExecutor``. ``jobs<=0`` picks
    ``max(1, cpu_count-2)``. The cache is written every ``checkpoint_every`` *completed* decoys and
    the skip set is derived from successes only, so a killed run resumes and retries failures —
    identical to the serial path. Result order is completion order, not input order (irrelevant:
    graphs and labels stay pairwise-aligned).
    """
    import os as _os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import torch

    graphs, kept = [], []
    if cache_path and _os.path.exists(cache_path):
        blob = torch.load(cache_path, weights_only=False)
        graphs, kept = blob["graphs"], blob["labels"]
    kept_keys = {(lab.target, lab.model) for lab in kept}
    todo = [lab for lab in labels if (lab.target, lab.model) not in kept_keys]
    if not todo:
        return graphs, kept
    if jobs <= 0:
        jobs = max(1, (_os.cpu_count() or 2) - 2)

    def _save():
        if cache_path:
            torch.save({"graphs": graphs, "labels": kept, "done": sorted(kept_keys)}, cache_path)

    done = failed = since_save = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_featurize_one, lab) for lab in todo]
        for fut in as_completed(futures):
            lab, graph = fut.result()
            done += 1
            if graph is None:
                failed += 1
            else:
                graphs.append(graph)
                kept.append(lab)
                kept_keys.add((lab.target, lab.model))
            since_save += 1
            if progress and done % 100 == 0:
                print(f"  [{done}/{len(todo)}] kept {len(kept)} failed {failed}", flush=True)
            if since_save >= checkpoint_every:
                _save()
                since_save = 0

    _save()
    if progress:
        print(
            f"featurize_subset_parallel: kept {len(kept)}/{len(labels)} "
            f"({len(todo)} attempted this run, {failed} failed, {jobs} jobs)",
            flush=True,
        )
    return graphs, kept
