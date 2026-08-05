"""P6b competitiveness leaderboard — rank benchmark decoys by training-free pLDDT baselines.

The DProQ-benchmark decoys are AlphaFold-Multimer models carrying per-residue pLDDT in the
B-factor column, so ranking a target's decoys by mean pLDDT is a standard AF-confidence baseline
needing no model. This scores the *interface* and *global* pLDDT arms on the same subsets + the
same per-target ranking metrics the topological scorer is judged on, so the two go head-to-head.

    export PATH="$HOME/.pixi/bin:$PATH"        # (not needed here — pLDDT needs no mkdssp)
    uv run python scripts/phase_g_leaderboard.py --subset zenodo/DProQ_benchmark/BM55-AF2
    uv run python scripts/phase_g_leaderboard.py --subset zenodo/DProQ_benchmark/HAF2 --drop-target 7ALA

Our-model numbers for the assembled leaderboard come from results/phase_e_full_corpus.jsonl;
published QA-tool numbers are added in the write-up (external papers, not run here).
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from open_topoqa_scorer.baselines import global_plddt, interface_plddt
from open_topoqa_scorer.benchmark import load_labels
from open_topoqa_scorer.evaluate import per_target_ranking_metrics
from open_topoqa_featurizer.graph import parse_structure

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_OUT = os.path.join(_ROOT, "results", "phase_g_leaderboard.jsonl")

# Same expected (targets, decoys) guard as phase_e, so a partial extraction can't produce a
# confident leaderboard row on the wrong set. Checked before --drop-target.
_EXPECTED = {"BM55-AF2": (15, 449), "HAF2": (13, 1370)}


def _score_one(lab):
    """Parse a decoy once, return both pLDDT arms. (lab, None, None) on any failure."""
    try:
        model = parse_structure(lab.pdb_path)
        return lab, interface_plddt(model), global_plddt(model)
    except Exception:  # noqa: BLE001 — a bad decoy must not sink the batch
        return lab, None, None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", required=True, help="path to a benchmark subset dir")
    p.add_argument("--drop-target", nargs="*", default=[], help="targets to exclude (e.g. 7ALA)")
    p.add_argument("--jobs", type=int, default=0, help="processes (<=0 -> cpu_count-2)")
    p.add_argument("--no-expect-check", action="store_true")
    args = p.parse_args()

    subset_name = os.path.basename(os.path.normpath(args.subset))
    labels = load_labels(args.subset)
    if subset_name in _EXPECTED and not args.no_expect_check:
        got = (len({l.target for l in labels}), len(labels))
        if got != _EXPECTED[subset_name]:
            raise SystemExit(
                f"{subset_name}: resolved {got} decoys, expected {_EXPECTED[subset_name]} — "
                f"extraction looks partial/corrupt (pass --no-expect-check to override)"
            )
    if args.drop_target:
        drop = set(args.drop_target)
        labels = [lab for lab in labels if lab.target not in drop]
    resolved = f"{subset_name}-{len({l.target for l in labels})}" if args.drop_target else subset_name
    print(f"{resolved}: {len(labels)} decoys across {len({l.target for l in labels})} targets", flush=True)

    jobs = args.jobs if args.jobs > 0 else max(1, (os.cpu_count() or 2) - 2)
    kept, iface, glob = [], [], []
    failed = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_score_one, lab) for lab in labels]
        for n, fut in enumerate(as_completed(futures), 1):
            lab, i, g = fut.result()
            if i is None:
                failed += 1
            else:
                kept.append(lab)
                iface.append(i)
                glob.append(g)
            if n % 200 == 0:
                print(f"  [{n}/{len(labels)}] kept {len(kept)} failed {failed}", flush=True)
    print(f"scored {len(kept)}/{len(labels)} decoys ({failed} failed, {jobs} jobs)", flush=True)

    rows = []
    for arm, scores in (("interface_plddt", iface), ("global_plddt", glob)):
        m = per_target_ranking_metrics(scores, kept)
        row = {"subset": resolved, "method": arm, "n": len(kept), **m}
        rows.append(row)
        print(
            f"\n=== {resolved} · {arm} ===\n"
            f"  ranking_loss_mean  {m['ranking_loss_mean']:.3f}\n"
            f"  spearman_mean      {m['spearman_mean']:.3f}\n"
            f"  top10_success_rate {m['top10_success_rate']:.3f}\n"
            f"  targets_ranked     {m['targets_ranked']}/{m['targets_total']}",
            flush=True,
        )

    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    with open(_OUT, "a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"\nappended {len(rows)} rows -> {_OUT}", flush=True)


if __name__ == "__main__":
    main()
