"""Phase D: train the ProteinGAT scorer on the Phase C3 cached graphs, evaluate, checkpoint.

Consumes ``cache/{train,val}_graphs.pt`` (produced by ``phase_c_build.py``). Trains one model per
requested seed (best-validation weights), reports pooled regression + per-target ranking metrics on
the homology-held-out val set, and saves the best-seed checkpoint. Run:

    uv run python scripts/phase_d_train.py --seeds 0 1 2 --epochs 200
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from open_topoqa_scorer.evaluate import per_target_ranking_metrics, pooled_regression_metrics
from open_topoqa_scorer.train import predict, train_model

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _load(cache_path):
    blob = torch.load(cache_path, weights_only=False)
    return blob["graphs"], blob["labels"]


def _evaluate(model, graphs, labels) -> dict:
    scores = predict(model, graphs).view(-1).tolist()
    return {**pooled_regression_metrics(scores, labels), **per_target_ranking_metrics(scores, labels)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", default=os.path.join(_ROOT, "cache"))
    p.add_argument("--out-dir", default=os.path.join(_ROOT, "checkpoints"))
    p.add_argument("--train-cache", default="train_graphs.pt")
    p.add_argument("--val-cache", default="val_graphs.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--conv", default="gatv2")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--ranking-weight", type=float, default=0.0, help=">0 adds within-target ranking loss")
    p.add_argument("--ranking-margin", type=float, default=0.1)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tr_g, tr_l = _load(os.path.join(args.cache_dir, args.train_cache))
    va_g, va_l = _load(os.path.join(args.cache_dir, args.val_cache))
    print(f"train {len(tr_g)} graphs / val {len(va_g)} graphs "
          f"| ranking_weight {args.ranking_weight}", flush=True)
    model_kwargs = {"conv": args.conv, "heads": args.heads}
    rank_kw = {}
    if args.ranking_weight > 0:
        rank_kw = {
            "ranking_weight": args.ranking_weight,
            "ranking_margin": args.ranking_margin,
            "train_targets": [lab.target for lab in tr_l],
            "val_targets": [lab.target for lab in va_l],
        }

    best = None
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        model, history = train_model(
            tr_g, va_g, epochs=args.epochs, lr=args.lr, seed=seed,
            model_kwargs=model_kwargs, verbose=True, **rank_kw,
        )
        val_metrics = _evaluate(model, va_g, va_l)
        best_val = min(history["val_loss"])
        print(f"seed {seed}: best_val_mse {best_val:.4f} | {json.dumps(val_metrics)}", flush=True)
        record = {"seed": seed, "best_val_mse": best_val, "val_metrics": val_metrics,
                  "model_kwargs": model_kwargs, "history": history}
        if best is None or best_val < best["best_val_mse"]:
            best = {**record, "state_dict": model.state_dict()}

    ckpt = os.path.join(args.out_dir, "proteingat_best.pt")
    torch.save(best, ckpt)
    summary = {k: best[k] for k in ("seed", "best_val_mse", "val_metrics", "model_kwargs")}
    with open(os.path.join(args.out_dir, "phase_d_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nbest seed {best['seed']} -> {ckpt}\n{json.dumps(summary['val_metrics'], indent=2)}", flush=True)


if __name__ == "__main__":
    main()
