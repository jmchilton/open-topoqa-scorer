"""Phase E: evaluate a trained scorer on the DProQ benchmark test sets, paper-style.

The benchmark decoy folders ship exactly the paper's filtered test targets — BM55-AF2
(15 targets / 449 decoys) and HAF2 (13 targets / 1370 decoys; the paper drops 7ALA -> 12) —
so no re-filtering is needed to compare against the reported numbers.

Two modes:
    --featurize-only   featurize decoys into the cache (resumable) and stop; no checkpoint needed
    (default)          featurize (reusing cache) + score a checkpoint + report per-target metrics

Metrics follow the TopoQA/DProQA protocol: per-target top-1 ranking loss (the paper's primary
selection statistic), per-target Spearman/Pearson averaged over targets, pooled Pearson, and
top-10 CAPRI success. Run:

    export PATH="$HOME/.pixi/bin:$PATH"   # mkdssp
    uv run python scripts/phase_e_benchmark.py --subset zenodo/DProQ_benchmark/BM55-AF2 \
        --checkpoint checkpoints/proteingat_best.pt
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from open_topoqa_scorer.benchmark import featurize_subset, load_labels
from open_topoqa_scorer.evaluate import per_target_ranking_metrics, pooled_regression_metrics
from open_topoqa_scorer.metrics import pearson
from open_topoqa_scorer.model import ProteinGAT
from open_topoqa_scorer.train import predict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Paper-reported test-set numbers (arXiv 2410.17815) for at-a-glance comparison.
_PAPER = {
    "BM55-AF2": {"ranking_loss": 0.069, "pearson": 0.515, "spearman": 0.502},
    "HAF2": {"ranking_loss": 0.110, "pearson": 0.600, "spearman": 0.675},
}


def _load_model(checkpoint_path: str) -> ProteinGAT:
    blob = torch.load(checkpoint_path, weights_only=False)
    model = ProteinGAT(**blob.get("model_kwargs", {}))
    model.load_state_dict(blob["state_dict"])
    return model.eval()


def _per_target_pearson_mean(scores, labels) -> float:
    scores = np.asarray(scores, dtype=float).ravel()
    by_target: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_target[lab.target].append(i)
    rs = []
    for idx in by_target.values():
        if len(idx) < 2:
            continue
        rs.append(pearson(scores[idx], np.array([labels[i].dockq for i in idx])))
    return float(np.mean(rs)) if rs else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", required=True, help="path to a benchmark subset dir")
    p.add_argument("--cache", default=None, help="featurization cache (default <subset>/_phaseE_cache.pt)")
    p.add_argument("--checkpoint", default=os.path.join(_ROOT, "checkpoints", "proteingat_best.pt"))
    p.add_argument("--featurize-only", action="store_true")
    p.add_argument("--drop-target", nargs="*", default=[], help="targets to exclude (e.g. 7ALA for HAF2-12)")
    args = p.parse_args()

    labels = load_labels(args.subset)
    if args.drop_target:
        drop = set(args.drop_target)
        labels = [lab for lab in labels if lab.target not in drop]
    subset_name = os.path.basename(os.path.normpath(args.subset))
    print(f"{subset_name}: {len(labels)} decoys across {len({l.target for l in labels})} targets", flush=True)

    cache = args.cache or os.path.join(args.subset, "_phaseE_cache.pt")
    graphs, kept = featurize_subset(labels, cache_path=cache, progress=True)
    print(f"featurized {len(graphs)}/{len(labels)} decoys "
          f"({len({l.target for l in kept})} targets)", flush=True)
    if args.featurize_only:
        return

    model = _load_model(args.checkpoint)
    scores = predict(model, graphs).view(-1).tolist()
    result = {
        **pooled_regression_metrics(scores, kept),
        **per_target_ranking_metrics(scores, kept),
        "per_target_pearson_mean": _per_target_pearson_mean(scores, kept),
    }
    paper = _PAPER.get(subset_name, {})
    print(f"\n=== {subset_name} vs paper ===", flush=True)
    print(f"  ranking_loss_mean  {result['ranking_loss_mean']:.3f}   (paper {paper.get('ranking_loss', float('nan')):.3f})")
    print(f"  spearman_mean      {result['spearman_mean']:.3f}   (paper {paper.get('spearman', float('nan')):.3f})")
    print(f"  per_target_pearson {result['per_target_pearson_mean']:.3f}   (paper {paper.get('pearson', float('nan')):.3f})")
    print(f"  pooled_pearson     {result['pearson']:.3f}")
    print(f"  top10_success_rate {result['top10_success_rate']:.3f}")
    print(f"  targets_ranked     {result['targets_ranked']}/{result['targets_total']}")
    print("\n" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
