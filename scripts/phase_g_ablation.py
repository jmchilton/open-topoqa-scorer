"""P6a attribution ablation — does the persistent-homology node block add signal? (issue #12)

Trains the SAME ProteinGAT under the SAME protocol on three node-feature slices of the SAME corpus,
and evaluates each on the val split + BM55-AF2 + HAF2-12. Only the node columns differ:

    full               node[  0:172]  conventional (32) + topological (140)  — the shipped model
    conventional_only  node[  0: 32]  21 AA + 8 DSSP SS + 1 SASA + 2 (phi,psi)
    topological_only   node[ 32:172]  7 element channels x 20 persistence summaries

So Delta(full - conventional) isolates what the PH block contributes, and topological_only shows
whether topology alone carries it. Everything else — graph, GNN kwargs, corpus, leak-free split,
lr/epochs/seed, best-val-MSE selection, per-feature standardization — is held identical across arms.

    uv run python scripts/phase_g_ablation.py --epochs 40
"""

from __future__ import annotations

import argparse
import gc
import json
import os

import torch

from open_topoqa_scorer.benchmark import featurize_subset, load_labels, restrict_to_labels
from open_topoqa_scorer.evaluate import per_target_ranking_metrics, pooled_regression_metrics
from open_topoqa_scorer.train import predict, train_model
from open_topoqa_featurizer import NODE_WIDTH
from open_topoqa_featurizer.graph import CONVENTIONAL_WIDTH  # 32

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_OUT = os.path.join(_ROOT, "results", "phase_g_ablation.jsonl")

# (start, end) node-feature columns per arm. Blocks are contiguous (conventional first).
_ARMS = {
    "full": (0, NODE_WIDTH),
    "conventional_only": (0, CONVENTIONAL_WIDTH),
    "topological_only": (CONVENTIONAL_WIDTH, NODE_WIDTH),
}


def _slice(graphs, start, end):
    """New graphs with node features restricted to columns [start:end); edges/y untouched."""
    out = []
    for g in graphs:
        d = g.clone()
        d.x = g.x[:, start:end].contiguous()
        out.append(d)
    return out


def _load_corpus(cache_dir, train_name, val_name):
    tr = torch.load(os.path.join(cache_dir, train_name), weights_only=False)
    va = torch.load(os.path.join(cache_dir, val_name), weights_only=False)
    return tr["graphs"], tr["labels"], va["graphs"], va["labels"]


def _load_benchmark(subset_dir, drop):
    labels = load_labels(subset_dir)
    if drop:
        labels = [l for l in labels if l.target not in set(drop)]
    cache = os.path.join(subset_dir, "_phaseE_cache.pt")
    graphs, kept = featurize_subset(labels, cache_path=cache)
    return restrict_to_labels(graphs, kept, labels)


def _evaluate(model, graphs, labels):
    scores = predict(model, graphs).view(-1).tolist()
    return {**pooled_regression_metrics(scores, labels), **per_target_ranking_metrics(scores, labels)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", default=os.path.join(_ROOT, "cache"))
    p.add_argument("--train-cache", default="full_train_graphs.pt")
    p.add_argument("--val-cache", default="full_val_graphs.pt")
    p.add_argument("--epochs", type=int, default=40, help="identical across arms")
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--conv", default="gatv2")
    p.add_argument("--arms", nargs="+", default=list(_ARMS), choices=list(_ARMS))
    p.add_argument("--bm55", default=os.path.join(_ROOT, "zenodo/DProQ_benchmark/BM55-AF2"))
    p.add_argument("--haf2", default=os.path.join(_ROOT, "zenodo/DProQ_benchmark/HAF2"))
    args = p.parse_args()

    tr_g, tr_l, va_g, va_l = _load_corpus(args.cache_dir, args.train_cache, args.val_cache)
    print(f"corpus: train {len(tr_g)} / val {len(va_g)}", flush=True)
    bm_g, bm_l = _load_benchmark(args.bm55, drop=[])
    ha_g, ha_l = _load_benchmark(args.haf2, drop=["7ALA"])
    print(f"benchmark: BM55-AF2 {len(bm_g)} / HAF2-12 {len(ha_g)}", flush=True)

    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    for arm in args.arms:
        start, end = _ARMS[arm]
        width = end - start
        print(f"\n=== arm {arm} (node_dim {width}, cols {start}:{end}) ===", flush=True)
        tr_s, va_s = _slice(tr_g, start, end), _slice(va_g, start, end)
        bm_s, ha_s = _slice(bm_g, start, end), _slice(ha_g, start, end)

        model_kwargs = {"conv": args.conv, "heads": args.heads, "node_dim": width}
        model, history = train_model(
            tr_s, va_s, epochs=args.epochs, lr=args.lr, seed=args.seed,
            model_kwargs=model_kwargs, verbose=False,
        )
        best_epoch = history.get("best_epoch")
        evals = {
            "val": _evaluate(model, va_s, va_l),
            "BM55-AF2": _evaluate(model, bm_s, bm_l),
            "HAF2-12": _evaluate(model, ha_s, ha_l),
        }
        for subset, m in evals.items():
            row = {
                "arm": arm, "node_dim": width, "subset": subset,
                "epochs": args.epochs, "best_epoch": best_epoch, "seed": args.seed,
                **m,
            }
            with open(_OUT, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(
                f"  {subset:9s} rank_loss {m['ranking_loss_mean']:.3f} "
                f"spearman {m['spearman_mean']:.3f} top10 {m['top10_success_rate']:.3f} "
                f"pooled_pearson {m['pearson']:.3f}",
                flush=True,
            )
        del tr_s, va_s, bm_s, ha_s, model
        gc.collect()

    print(f"\nappended rows -> {_OUT}", flush=True)


if __name__ == "__main__":
    main()
