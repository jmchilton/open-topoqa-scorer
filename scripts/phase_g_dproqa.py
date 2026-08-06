"""P6b same-protocol external arm — run DProQA on the identical benchmark decoys (issue #12).

DProQA (Chen et al., Bioinformatics 2023; repo BioinfoMachineLearning/DProQ -> jianlin-cheng/DProQA)
is a gated-graph-transformer complex QA model — a *non-topological trained* competitor. Table B of
p6b_leaderboard.md carried its published DBM55-AF2 0.049 / HAF2 0.195 ranking losses, but those come
from its own pipeline, not our decoy-for-decoy metric. This driver runs DProQA's own inference.py on
the SAME DProQ-benchmark decoys we score our model on, then pushes its PRED_DOCKQ through the SAME
per_target_ranking_metrics — promoting the Table B rows into strictly head-to-head Table A.

DProQA lives in a sibling repo with its own venv (arm64/CPU port; see that repo's requirements-m4.txt
and the cuda->cpu / DGL-API / DSSP-4 shims). We shell out to it per target so the feature generation
and forward pass are entirely DProQA's; only the *scoring* is ours.

    uv run python scripts/phase_g_dproqa.py --subset zenodo/DProQ_benchmark/BM55-AF2
    uv run python scripts/phase_g_dproqa.py --subset zenodo/DProQ_benchmark/HAF2 --drop-target 7ALA
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from collections import defaultdict

from open_topoqa_scorer.benchmark import load_labels
from open_topoqa_scorer.evaluate import per_target_ranking_metrics, pooled_regression_metrics

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_OUT = os.path.join(_ROOT, "results", "phase_g_dproqa.jsonl")

# Same (targets, decoys) guard as phase_e/phase_g_leaderboard, checked before --drop-target.
_EXPECTED = {"BM55-AF2": (15, 449), "HAF2": (13, 1370)}

_DEFAULT_DPROQA = os.path.expanduser("~/projects/repositories/DProQA")


def _decoy_dir(subset_dir: str, target: str) -> str:
    """Folder holding a target's decoy PDBs — decoy/<t> (BM55) or decoy/<t>/pdb (HAF2)."""
    for base in (
        os.path.join(subset_dir, "decoy", target, "pdb"),
        os.path.join(subset_dir, "decoy", target),
    ):
        if os.path.isdir(base) and any(f.endswith(".pdb") for f in os.listdir(base)):
            return os.path.abspath(base)  # inference.py runs with cwd=DProQA dir, needs absolute -c
    raise FileNotFoundError(f"no decoy pdb folder for {target}")


def _norm(model_name: str) -> str:
    """DProQA MODEL == clean-tidy basename minus '_tidy.pdb'; BM55 files are pre-'_tidy' named so a
    trailing '_tidy' survives (HAF2 has none). Strip one trailing '_tidy' to match label.model."""
    return model_name[:-5] if model_name.endswith("_tidy") else model_name


def _run_target(dproqa_dir: str, decoy_dir: str, work: str, res: str, threads: int,
                keep_intermediates: bool) -> dict:
    """Run DProQA inference.py for one target (resumable) and parse Ranking.csv -> {norm_model: pred}.

    DProQA's TIDY/DIST/DGL intermediates are huge — the CA/CB/N-O distance maps alone are ~400 MB for
    30 decoys, ~1.5 GB for a big HAF2 target. We only need the tiny Ranking.csv, so unless told to keep
    them we delete ``work`` right after parsing, holding peak disk to a single target's worth.
    """
    ranking_csv = os.path.join(res, "Ranking.csv")
    # Resume only from a Ranking.csv with real data rows — a killed run can leave a header-only file,
    # which must NOT be trusted (it would silently contribute zero scored decoys for the target).
    have_rows = os.path.isfile(ranking_csv) and sum(1 for _ in open(ranking_csv)) > 1
    if not have_rows:
        py = os.path.join(dproqa_dir, ".venv", "bin", "python")
        subprocess.run(
            [py, "inference.py", "-c", decoy_dir, "-w", work, "-r", res, "-t", str(threads)],
            cwd=dproqa_dir, check=True,
        )
    preds: dict[str, float] = {}
    with open(ranking_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            preds[_norm(row["MODEL"])] = float(row["PRED_DOCKQ"])
    if not keep_intermediates and os.path.isdir(work):
        shutil.rmtree(work, ignore_errors=True)  # keep _result/Ranking.csv, drop TIDY/DIST/DGL
    return preds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", required=True, help="path to a benchmark subset dir")
    p.add_argument("--drop-target", nargs="*", default=[], help="targets to exclude (e.g. 7ALA)")
    p.add_argument("--dproqa-dir", default=_DEFAULT_DPROQA)
    p.add_argument("--work-root", default=os.path.join(_ROOT, "results", "dproqa_work"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--keep-intermediates", action="store_true",
                   help="keep TIDY/DIST/DGL (huge); default deletes them per target after scoring")
    p.add_argument("--no-expect-check", action="store_true")
    args = p.parse_args()

    subset_name = os.path.basename(os.path.normpath(args.subset))
    labels = load_labels(args.subset)
    if subset_name in _EXPECTED and not args.no_expect_check:
        got = (len({l.target for l in labels}), len(labels))
        if got != _EXPECTED[subset_name]:
            raise SystemExit(f"{subset_name}: resolved {got}, expected {_EXPECTED[subset_name]}")
    if args.drop_target:
        labels = [l for l in labels if l.target not in set(args.drop_target)]

    by_target: dict[str, list] = defaultdict(list)
    for lab in labels:
        by_target[lab.target].append(lab)

    pred_by_key: dict[tuple[str, str], float] = {}
    for target in by_target:
        decoy_dir = _decoy_dir(args.subset, target)
        work = os.path.join(args.work_root, subset_name, target)
        res = os.path.join(args.work_root, subset_name, "_result", target)
        os.makedirs(res, exist_ok=True)
        preds = _run_target(args.dproqa_dir, decoy_dir, work, res, args.threads, args.keep_intermediates)
        for m, v in preds.items():
            pred_by_key[(target, m)] = v
        print(f"  {target}: {len(preds)} scored", flush=True)

    # Align to labels; a decoy DProQA failed to score sorts last (target-min minus 1) so it is never
    # picked as top-1 — conservative for DProQA, and coverage is reported so it can't hide.
    target_min = {t: min((pred_by_key.get((t, _norm(l.model))) for l in labs
                          if pred_by_key.get((t, _norm(l.model))) is not None), default=0.0)
                  for t, labs in by_target.items()}
    scores, missing = [], []
    for lab in labels:
        v = pred_by_key.get((lab.target, _norm(lab.model)))
        if v is None:
            missing.append((lab.target, lab.model))
            v = target_min[lab.target] - 1.0
        scores.append(v)

    metrics = {**pooled_regression_metrics(scores, labels), **per_target_ranking_metrics(scores, labels)}
    row = {
        "method": "DProQA", "subset": subset_name,
        "targets": len({l.target for l in labels}), "decoys": len(labels),
        "missing": len(missing), **metrics,
    }
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    with open(_OUT, "a") as fh:
        import json
        fh.write(json.dumps(row) + "\n")

    print(f"\n=== DProQA on {subset_name} ({row['targets']} targets / {row['decoys']} decoys) ===")
    if missing:
        print(f"  WARNING: {len(missing)} decoys unscored by DProQA (assigned worst rank): {missing[:5]}...")
    print(f"  ranking_loss {metrics['ranking_loss_mean']:.3f}  spearman {metrics['spearman_mean']:.3f}  "
          f"top10 {metrics['top10_success_rate']:.3f}  pooled_r {metrics['pearson']:.3f}")
    print(f"  appended -> {_OUT}")


if __name__ == "__main__":
    main()
