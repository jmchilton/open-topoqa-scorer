"""Aggregate results/phase_g_ablation.jsonl across seeds -> mean +/- range per (arm, subset).

Top-1 ranking loss on 12-15 targets is high-variance, so the ablation is only interpretable as a
mean over several seeds with the spread shown. Prints a table per subset. (issue #12, P6a)

    uv run python scripts/aggregate_ablation.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_JSONL = os.path.join(_ROOT, "results", "phase_g_ablation.jsonl")

_ARM_ORDER = ["full", "conventional_only", "topological_only"]
_SUBSETS = ["val", "BM55-AF2", "HAF2-12"]
_METRICS = ["ranking_loss_mean", "spearman_mean", "top10_success_rate", "pearson"]


def _mean(xs):
    return sum(xs) / len(xs)


def main() -> None:
    rows = [json.loads(l) for l in open(_JSONL)]
    seeds = sorted({r["seed"] for r in rows})
    # (arm, subset, metric) -> [value per seed]
    acc = defaultdict(list)
    for r in rows:
        for m in _METRICS:
            acc[(r["arm"], r["subset"], m)].append(r[m])

    print(f"seeds: {seeds}  (n={len(seeds)})\n")
    for subset in _SUBSETS:
        print(f"=== {subset} ===")
        print(f"{'arm':18s} {'rank_loss':18s} {'spearman':18s} {'top10':16s} {'pooled_r':16s}")
        for arm in _ARM_ORDER:
            cells = []
            for m in _METRICS:
                vs = acc[(arm, subset, m)]
                if not vs:
                    cells.append("--")
                    continue
                cells.append(f"{_mean(vs):+.3f} [{min(vs):+.3f},{max(vs):+.3f}]")
            print(f"{arm:18s} {cells[0]:18s} {cells[1]:18s} {cells[2]:16s} {cells[3]:16s}")
        # Delta(full - conventional) and (topological - conventional) on the mean, per metric.
        for m in ("ranking_loss_mean", "spearman_mean"):
            conv = _mean(acc[("conventional_only", subset, m)])
            full = _mean(acc[("full", subset, m)])
            topo = _mean(acc[("topological_only", subset, m)])
            print(f"  Δ {m:18s}: full-conv {full-conv:+.3f}   topo-conv {topo-conv:+.3f}")
        print()


if __name__ == "__main__":
    main()
