"""Phase C3 (ranking corpus): whole-target sampling tuned for within-target ranking.

Unlike ``phase_c_build.py`` (decoy-level stratified sample, which shreds targets into singletons),
this keeps whole targets — every Dockground target capped at ``--dg-cap`` decoys for ranking depth,
plus a sample of MAF2 targets kept whole for absolute breadth. Writes ``ranking_{train,val}_graphs.pt``.

Featurization reuses any decoy already in the plain ``{train,val}_graphs.pt`` caches (pre-seeded by
``(target, model)``), so only genuinely-new decoys are recomputed.

    uv run python scripts/phase_c_build_ranking.py --dg-cap 25 --maf2-train-targets 200
"""

from __future__ import annotations

import argparse
import os

import torch

from open_topoqa_scorer.benchmark import featurize_subset
from open_topoqa_scorer.corpus import (
    capri_counts,
    is_dockground_target,
    load_combined_labels,
    subset_for_ranking,
)
from open_topoqa_scorer.split import split_by_clusters

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _default(*parts):
    return os.path.join(_ROOT, *parts)


def _preseed_cache(new_labels, source_caches, out_cache):
    """Pre-fill ``out_cache`` with graphs for ``new_labels`` already present in ``source_caches``."""
    if os.path.exists(out_cache):
        return  # a prior (possibly partial) run already owns this cache; let it resume as-is
    have: dict = {}
    for src in source_caches:
        if os.path.exists(src):
            blob = torch.load(src, weights_only=False)
            for g, lab in zip(blob["graphs"], blob["labels"]):
                have[(lab.target, lab.model)] = g
    graphs, labels = [], []
    for lab in new_labels:
        g = have.get((lab.target, lab.model))
        if g is not None:
            graphs.append(g)
            labels.append(lab)
    if graphs:
        keys = sorted({(l.target, l.model) for l in labels})
        torch.save({"graphs": graphs, "labels": labels, "done": keys}, out_cache)
        print(f"  pre-seeded {len(graphs)} cached graphs into {os.path.basename(out_cache)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--maf2-dir", default=_default("zenodo", "MAF2"))
    p.add_argument("--dockground-tgz", default=_default("data", "dockground", "extracted_tgz", "decoy", "decoy"))
    p.add_argument("--dockground-extract", default=_default("data", "dockground", "targets"))
    p.add_argument("--out-dir", default=_default("cache"))
    p.add_argument("--dg-cap", type=int, default=25, help="max decoys kept per Dockground target")
    p.add_argument("--maf2-train-targets", type=int, default=200)
    p.add_argument("--maf2-val-targets", type=int, default=40)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_dir = os.path.join(args.out_dir, "split_tmp")

    print("loading combined MAF2 + Dockground labels ...", flush=True)
    labels = load_combined_labels(args.maf2_dir, args.dockground_tgz, args.dockground_extract)
    print(f"  {len(labels)} decoys / {len({l.target for l in labels})} targets", flush=True)

    print("homology split (mmseqs, min-seq-id 0.3) ...", flush=True)
    train, val = split_by_clusters(labels, tmp_dir=tmp_dir, val_frac=args.val_frac, seed=args.seed)

    train_s = subset_for_ranking(train, args.dg_cap, args.maf2_train_targets, seed=args.seed)
    val_s = subset_for_ranking(val, args.dg_cap, args.maf2_val_targets, seed=args.seed)
    for name, s in (("train", train_s), ("val", val_s)):
        n_dg = len({l.target for l in s if is_dockground_target(l.target)})
        print(f"  {name}: {len(s)} decoys / {len({l.target for l in s})} targets "
              f"({n_dg} Dockground) | CAPRI {capri_counts(s)}", flush=True)

    src = [os.path.join(args.out_dir, "train_graphs.pt"), os.path.join(args.out_dir, "val_graphs.pt")]
    tr_cache = os.path.join(args.out_dir, "ranking_train_graphs.pt")
    va_cache = os.path.join(args.out_dir, "ranking_val_graphs.pt")
    _preseed_cache(train_s, src, tr_cache)
    _preseed_cache(val_s, src, va_cache)

    print("featurizing (cached; ~5 s/decoy) ...", flush=True)
    tr_graphs, _ = featurize_subset(train_s, cache_path=tr_cache, progress=True)
    va_graphs, _ = featurize_subset(val_s, cache_path=va_cache, progress=True)

    for name, sampled, kept in (("train", train_s, tr_graphs), ("val", val_s, va_graphs)):
        rate = len(kept) / len(sampled) if sampled else 1.0
        print(f"  {name}: kept {len(kept)}/{len(sampled)} ({rate:.0%})", flush=True)
        if sampled and rate < 0.5:
            raise SystemExit(f"ABORT: {name} kept only {rate:.0%} — likely an environment fault")
    print(f"done: {len(tr_graphs)} train / {len(va_graphs)} val graphs in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
