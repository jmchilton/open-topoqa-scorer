"""Phase C3 driver: combine MAF2 + Dockground -> homology split -> stratified sample -> featurize.

Produces two cached graph blobs (``train_graphs.pt`` / ``val_graphs.pt``) that Phase D trains on.
Featurization is the slow step (~5 s/decoy, mkdssp + persistent homology); ``featurize_subset``
caches to the ``.pt`` so an interrupted run resumes by re-reading rather than recomputing.

Run (defaults point at the local git-ignored data):

    uv run python scripts/phase_c_build.py --train-n 1500 --val-n 300
"""

from __future__ import annotations

import argparse
import os

from open_topoqa_scorer.benchmark import featurize_subset
from open_topoqa_scorer.corpus import capri_counts, load_combined_labels
from open_topoqa_scorer.sample import stratified_sample
from open_topoqa_scorer.split import split_by_clusters

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _default(*parts):
    return os.path.join(_ROOT, *parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--maf2-dir", default=_default("zenodo", "MAF2"))
    p.add_argument("--dockground-tgz", default=_default("data", "dockground", "extracted_tgz", "decoy", "decoy"))
    p.add_argument("--dockground-extract", default=_default("data", "dockground", "targets"))
    p.add_argument("--out-dir", default=_default("cache"))
    p.add_argument("--train-n", type=int, default=1500, help="stratified sample size from train")
    p.add_argument("--val-n", type=int, default=300, help="stratified sample size from val")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_dir = os.path.join(args.out_dir, "split_tmp")

    print("loading combined MAF2 + Dockground labels ...", flush=True)
    labels = load_combined_labels(args.maf2_dir, args.dockground_tgz, args.dockground_extract)
    print(f"  {len(labels)} decoys / {len({l.target for l in labels})} targets", flush=True)
    print(f"  CAPRI classes: {capri_counts(labels)}", flush=True)

    print("homology split (mmseqs, min-seq-id 0.3) ...", flush=True)
    train, val = split_by_clusters(labels, tmp_dir=tmp_dir, val_frac=args.val_frac, seed=args.seed)
    print(f"  train {len(train)} / val {len(val)} decoys", flush=True)

    train_s = stratified_sample(train, args.train_n, seed=args.seed)
    val_s = stratified_sample(val, args.val_n, seed=args.seed)
    print(f"  sampled train {len(train_s)} {capri_counts(train_s)}", flush=True)
    print(f"  sampled val   {len(val_s)} {capri_counts(val_s)}", flush=True)

    print("featurizing (cached; ~5 s/decoy) ...", flush=True)
    tr_graphs, tr_kept = featurize_subset(
        train_s, cache_path=os.path.join(args.out_dir, "train_graphs.pt"), progress=True
    )
    va_graphs, va_kept = featurize_subset(
        val_s, cache_path=os.path.join(args.out_dir, "val_graphs.pt"), progress=True
    )

    # Surface attrition explicitly and refuse a gutted corpus — a large silent drop (e.g. the
    # historical DSSP-on-decoys fault) otherwise looks like a smaller-but-fine result and would
    # train on near-nothing. Re-running self-heals transient faults (featurize_subset retries).
    for name, sampled, kept in (("train", train_s, tr_graphs), ("val", val_s, va_graphs)):
        rate = len(kept) / len(sampled) if sampled else 1.0
        print(f"  {name}: kept {len(kept)}/{len(sampled)} ({rate:.0%})", flush=True)
        if sampled and rate < 0.5:
            raise SystemExit(
                f"ABORT: {name} featurization kept only {rate:.0%} of {len(sampled)} decoys — "
                "likely an environment fault (mkdssp?). Fix it and re-run; the cache resumes."
            )
    print(f"done: {len(tr_graphs)} train / {len(va_graphs)} val graphs cached in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
