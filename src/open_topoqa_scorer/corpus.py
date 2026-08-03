"""Assemble the combined Phase C training corpus: MAF2 breadth + Dockground depth.

MAF2 gives ~2 models across thousands of targets (good for absolute DockQ regression); Dockground
gives ~100 decoys across ~58 targets (good for within-target ranking). Concatenating them into one
:class:`DecoyLabel` list lets the homology split ([[split]]) and stratified sampler ([[sample]])
treat both uniformly. Both PDB-id namespaces coexist; homologs across the two corpora cluster
together in the split, which is the intended behaviour (a shared structure must not span the split).
"""

from __future__ import annotations

from collections import Counter

from open_topoqa_scorer.benchmark import DecoyLabel
from open_topoqa_scorer.dockground import load_dockground_labels
from open_topoqa_scorer.training_data import load_maf2_labels

__all__ = ["load_combined_labels", "capri_counts"]


def load_combined_labels(
    maf2_dir: str | None = None,
    dockground_tgz_dir: str | None = None,
    dockground_extract_dir: str | None = None,
    require_files: bool = True,
) -> list[DecoyLabel]:
    """MAF2 + Dockground decoys as one list. Either source may be omitted (pass ``None``).

    The two corpora use different target-id namespaces (MAF2 dir stems, Dockground lowercase PDB
    ids). Downstream clustering ([[split]]) and per-target ranking ([[evaluate]]) key on the target
    *string*, so a collision would silently fuse two distinct complexes into one target. That
    doesn't happen today (casing differs), but we assert disjointness rather than rely on the
    accident — a real collision raises here instead of corrupting the split/eval.
    """
    maf2_labels: list[DecoyLabel] = []
    if maf2_dir:
        maf2_labels = load_maf2_labels(maf2_dir, require_files=require_files)
    dg_labels: list[DecoyLabel] = []
    if dockground_tgz_dir:
        if not dockground_extract_dir:
            raise ValueError("dockground_extract_dir is required when dockground_tgz_dir is given")
        dg_labels = load_dockground_labels(
            dockground_tgz_dir, dockground_extract_dir, require_files=require_files
        )

    clash = {lab.target for lab in maf2_labels} & {lab.target for lab in dg_labels}
    if clash:
        raise ValueError(
            f"MAF2 and Dockground share {len(clash)} target id(s) (e.g. {sorted(clash)[:3]}); "
            "combining would fuse distinct complexes — disambiguate the namespaces first"
        )
    return maf2_labels + dg_labels


def capri_counts(labels) -> dict:
    """CAPRI-class histogram ``{class: count}`` (for logging split/sample balance)."""
    return dict(sorted(Counter(lab.capri for lab in labels).items()))
