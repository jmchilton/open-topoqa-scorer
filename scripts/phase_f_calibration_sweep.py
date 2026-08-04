"""Phase F: calibration/capacity sweep against the benchmark ranking loss.

The Codex reference localized our residual BM55 ranking-loss gap to two targets whose failure is
top-cluster *score compression*, not a feature bug (see ``results/phase_e_debrief.md``). Compression
has two principled suspects the paper never pins: readout **dropout** (squeezes output spread) and
**capacity** (width/depth). This sweep retrains the full-corpus scorer across a few settings of
those knobs and reports each on the held-out benchmark's per-target ranking loss.

Model selection stays on **val MSE** (the honest, in-distribution criterion) — the benchmark is the
*test* set and is never used to pick a checkpoint, only to report. Reads the cached full corpus and
the cached benchmark featurizations; no re-featurize. Run:

    export PATH="$HOME/.pixi/bin:$PATH"
    uv run python scripts/phase_f_calibration_sweep.py --epochs 150
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from open_topoqa_scorer.benchmark import featurize_subset, load_labels, restrict_to_labels
from open_topoqa_scorer.evaluate import per_target_ranking_metrics, pooled_regression_metrics
from open_topoqa_scorer.train import predict, train_model

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# (label, model_kwargs beyond conv/heads). Baseline reproduces the shipped full-MSE checkpoint.
_CONFIGS = [
    ("baseline_d25_h32_l2", {"dropout": 0.25, "hidden": 32, "num_layers": 2}),
    ("spread_d10_h32_l2", {"dropout": 0.10, "hidden": 32, "num_layers": 2}),
    ("cap_d10_h64_l3", {"dropout": 0.10, "hidden": 64, "num_layers": 3}),
]


def _load(cache):
    blob = torch.load(cache, weights_only=False)
    return blob["graphs"], blob["labels"]


def _bench(subset, drop=()):
    """Return (graphs, kept_labels) for a benchmark subset from its cache, minus dropped targets."""
    sub = os.path.join(_ROOT, "zenodo", "DProQ_benchmark", subset)
    labels = [l for l in load_labels(sub) if l.target not in drop]
    graphs, kept = featurize_subset(labels, cache_path=os.path.join(sub, "_phaseE_cache.pt"))
    return restrict_to_labels(graphs, kept, labels)


def _eval(model, graphs, labels):
    scores = predict(model, graphs).view(-1).tolist()
    m = {**pooled_regression_metrics(scores, labels), **per_target_ranking_metrics(scores, labels)}
    return {"ranking_loss": m["ranking_loss_mean"], "pooled_pearson": m["pearson"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(_ROOT, "results", "phase_f_sweep.jsonl"))
    args = p.parse_args()

    tr_g, _ = _load(os.path.join(_ROOT, "cache", "full_train_graphs.pt"))
    va_g, _ = _load(os.path.join(_ROOT, "cache", "full_val_graphs.pt"))
    print(f"corpus: {len(tr_g)} train / {len(va_g)} val", flush=True)
    bm55 = _bench("BM55-AF2")
    haf2 = _bench("HAF2", drop={"7ALA"})
    print(f"benchmark: BM55 {len(bm55[0])} decoys / HAF2-12 {len(haf2[0])} decoys", flush=True)

    rows = []
    for label, extra in _CONFIGS:
        kw = {"conv": "gatv2", "heads": 4, **extra}
        print(f"\n=== {label}  {kw} ===", flush=True)
        model, history = train_model(
            tr_g, va_g, epochs=args.epochs, lr=args.lr, seed=args.seed,
            model_kwargs=kw, verbose=False,
        )
        row = {
            "config": label, "model_kwargs": kw, "epochs": args.epochs,
            "best_val_mse": min(history["val_loss"]), "best_epoch": history.get("best_epoch"),
            "BM55-AF2": _eval(model, *bm55), "HAF2-12": _eval(model, *haf2),
        }
        rows.append(row)
        print(f"  best_val_mse {row['best_val_mse']:.4f} @epoch {row['best_epoch']}", flush=True)
        print(f"  BM55  rl {row['BM55-AF2']['ranking_loss']:.4f}  poolP {row['BM55-AF2']['pooled_pearson']:.3f}", flush=True)
        print(f"  HAF2-12 rl {row['HAF2-12']['ranking_loss']:.4f}  poolP {row['HAF2-12']['pooled_pearson']:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print("\n=== SWEEP SUMMARY (paper: BM55 0.069 / HAF2-12 0.110; upstream-corrected 0.077 / 0.147) ===", flush=True)
    print(f"{'config':22s} {'BM55 rl':>8s} {'HAF2-12 rl':>10s} {'BM55 poolP':>11s} {'HAF2 poolP':>11s}", flush=True)
    for r in rows:
        print(f"{r['config']:22s} {r['BM55-AF2']['ranking_loss']:8.4f} {r['HAF2-12']['ranking_loss']:10.4f} "
              f"{r['BM55-AF2']['pooled_pearson']:11.3f} {r['HAF2-12']['pooled_pearson']:11.3f}", flush=True)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
