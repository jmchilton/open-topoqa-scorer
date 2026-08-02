"""Phase B pipeline proof — featurize real CC-BY benchmark decoys and prove the loop
trains + ranks. This is a *smoke*, not the paper model: a few targets, few epochs.

Data: DProQ benchmark (Zenodo 6569837, CC-BY-4.0), extracted locally under the repo's
git-ignored ``zenodo/``. Needs ``mkdssp`` on PATH (``export PATH="$HOME/.pixi/bin:$PATH"``).

Usage:
    .venv/bin/python scripts/phase_b_smoke.py \
        --subset zenodo/DProQ_benchmark/BM55-AF2 --targets 3SE8 5HGG --epochs 80
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from open_topoqa_scorer.benchmark import featurize_subset, load_labels
from open_topoqa_scorer.metrics import pearson, ranking_loss, spearman
from open_topoqa_scorer.train import predict, train_model


def _per_target_metrics(labels, preds):
    by_target: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        by_target.setdefault(lab.target, []).append(i)
    rows = []
    for target, idx in sorted(by_target.items()):
        true = np.array([labels[i].dockq for i in idx])
        pred = np.array([float(preds[i]) for i in idx])
        rows.append((target, len(idx), spearman(pred, true), pearson(pred, true), ranking_loss(pred, true)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    labels = load_labels(args.subset)
    if args.targets:
        wanted = set(args.targets)
        labels = [lab for lab in labels if lab.target in wanted]
    if not labels:
        raise SystemExit("no decoys found on disk for the requested targets")

    print(f"loaded {len(labels)} decoys across {len({l.target for l in labels})} targets")
    dockq = np.array([l.dockq for l in labels])
    print(f"DockQ target range: {dockq.min():.3f} .. {dockq.max():.3f} (mean {dockq.mean():.3f})")

    cache = args.cache or os.path.join(args.subset, "_phaseB_cache.pt")
    print(f"featurizing (cache: {cache}) ...")
    graphs, kept = featurize_subset(labels, cache_path=cache, progress=True)
    print(f"featurized {len(graphs)} graphs; node dim {graphs[0].x.shape[1]}, edge dim {graphs[0].edge_attr.shape[1]}")

    # sanity: graphs align with labels, targets in [0,1]
    assert len(graphs) == len(kept)
    assert all(0.0 <= float(g.y) <= 1.0 for g in graphs)

    # held-out split by TARGET (no decoy of a val target is seen in training)
    targets = sorted({l.target for l in kept})
    val_target = targets[-1]
    train_idx = [i for i, l in enumerate(kept) if l.target != val_target]
    val_idx = [i for i, l in enumerate(kept) if l.target == val_target]
    train_set = [graphs[i] for i in train_idx]
    val_set = [graphs[i] for i in val_idx]
    print(f"\nsplit by target: train {len(train_set)} decoys ({len(targets) - 1} targets), "
          f"val {len(val_set)} decoys (target {val_target})")

    model, history = train_model(
        train_set, val_set=val_set, epochs=args.epochs, lr=0.005, batch_size=16, seed=args.seed
    )
    print(f"train loss {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")
    if history["val_loss"]:
        print(f"val   loss {history['val_loss'][0]:.4f} -> best {min(history['val_loss']):.4f}")

    preds = predict(model, graphs)
    print("\nper-target ranking (Spearman / Pearson / top-1 ranking-loss):")
    for target, n, sp, pe, rl in _per_target_metrics(kept, preds):
        split = "val " if target == val_target else "train"
        print(f"  [{split}] {target}  n={n:3d}  rho={sp:+.3f}  r={pe:+.3f}  rankloss={rl:.3f}")

    # the smoke passes if training reduced its loss (the loop learns from real graphs)
    assert history["train_loss"][-1] < history["train_loss"][0], "training did not reduce loss"
    print("\nOK: pipeline featurizes real benchmark decoys and the loop trains + produces rankings.")


if __name__ == "__main__":
    main()
