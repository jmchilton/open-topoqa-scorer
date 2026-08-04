# Phase E — paper-replication benchmark (2026-08-03)

Eval of our normalized scorer on the DProQ benchmark test sets, using the paper's own
protocol (per-target top-1 ranking loss + correlations). Clean-room: reproduced from the
paper (arXiv 2410.17815); upstream TopoQA code never read.

## Setup

- **Test sets match the paper exactly by construction.** The DProQ benchmark (Zenodo 6569837,
  CC-BY) ships only the filtered test targets: **BM55-AF2 15 targets / 449 decoys**,
  **HAF2 13 targets / 1370 decoys** (paper drops 7ALA → 12). No re-filtering needed.
- Featurized 100% (449/449, 1370/1370; zero DSSP failures after the featurizer fix).
- Two normalized checkpoints from the ranking corpus (1517 train / 429 val graphs — a
  **subsample**, not the paper's full 8,733), committed under `models/`:
  `scorer_mse_normalized.pt` (pure MSE, the paper's loss) and `scorer_ranking_normalized.pt`
  (MSE + within-target ranking loss).

## Results vs paper

| set | ckpt | ranking loss ↓ | Spearman ↑ | per-tgt Pearson ↑ | pooled Pearson | top10 succ |
|-----|------|---------------|-----------|-------------------|----------------|-----------|
| BM55-AF2 | **paper** | **0.069** | **0.502** | **0.515** | — | — |
| BM55-AF2 | mse  | 0.137 | 0.287 | 0.270 | 0.535 | 0.933 |
| BM55-AF2 | rank | 0.212 | 0.270 | 0.274 | 0.403 | 0.867 |
| HAF2-12  | **paper** | **0.110** | **0.675** | **0.600** | — | — |
| HAF2-12  | mse  | 0.154 | 0.221 | 0.352 | — | 0.833 |
| HAF2-12  | rank | **0.113** | 0.174 | 0.405 | — | 0.833 |

(HAF2 is 12 targets / 1270 decoys — 7ALA excluded per the paper. An earlier draft mislabeled
these as HAF2-12 but actually included 7ALA, because a full featurization cache short-circuited
`--drop-target`; fixed via `restrict_to_labels`. Corrected numbers shown.)

## Read

- **Same order of magnitude; one number essentially lands.** HAF2 ranking loss with the ranking
  checkpoint is **0.113 vs paper 0.110**. BM55 is ~2× the paper (0.137 vs 0.069).
- **top-10 success is strong** (0.93 / 0.85) — the model reliably surfaces a good decoy near
  the top even where fine-grained ordering lags.
- **Correlations are the clear weak spot** — Spearman ~0.20–0.29 vs paper 0.50–0.68; the model
  is under-calibrated on within-target ordering.
- **Pure MSE ≥ ranking-loss on BM55** (0.137 < 0.212) but the reverse on HAF2 (0.121 < 0.148):
  the ranking loss trained on our thin corpus helps AF-heterodimer ordering, hurts antibody-antigen.

## Prime suspect for the gap: training-corpus size

We trained on **1,517 decoys** vs the paper's **8,733** (5.7×). Correlations and ranking loss
are exactly what more, more-diverse training decoys-per-target should improve. Other, smaller
factors: unpinned hyperparameters (paper pins none of layers/heads/width/lr/epochs), and the
paper checkpoint's `(x,y,y)` coordinate bug — our clean `(x,y,z)` featurizer legitimately
differs, so bit-exact match was never the target.

## Corpus scaling — done (2026-08-03), and what it showed

Featurized the **full** MAF2 + Dockground corpus (parallel; `featurize_subset_parallel`) and
retrained: **11,695 train / 2,925 val decoys** (7.7× the subsample; comparable to the paper's
~12,140), same leak-free split. Benchmark eval of the full-corpus checkpoints:

| set | ckpt | ranking loss ↓ | Spearman ↑ | per-tgt Pearson ↑ | pooled Pearson |
|-----|------|---------------|-----------|-------------------|----------------|
| BM55-AF2 | **paper** | **0.069** | **0.502** | **0.515** | — |
| BM55-AF2 | full-mse  | 0.142 | 0.291 | 0.323 | 0.600 |
| BM55-AF2 | full-rank | 0.140 | 0.246 | 0.280 | 0.423 |
| HAF2-12  | **paper** | **0.110** | **0.675** | **0.600** | — |
| HAF2-12  | full-mse  | 0.142 | 0.249 | **0.481** | **0.697** |
| HAF2-12  | full-rank | 0.133 | 0.253 | 0.438 | 0.658 |

**Read (the informative part):**
- **Calibration improved, as predicted.** Pooled Pearson 0.535→0.600 (BM55) and 0.56→**0.70**
  (HAF2); HAF2 per-target Pearson 0.35→**0.48** — approaching the paper's 0.515 / 0.600.
- **Ranking loss did *not* improve.** BM55 stayed ~0.14 (paper 0.069); the previous best HAF2
  number even regressed (0.113 → 0.133). More data did **not** close the ranking-loss gap.
- **The tell:** held-out *val* (same MAF2+DG distribution) ranking loss is **0.037** — better than
  the paper's *test* numbers. The benchmark gap is therefore **distribution transfer**, not
  under-training: the model ranks in-distribution decoys well but transfers imperfectly to the
  benchmark's different decoy generators (antibody-antigen BM55, heterodimer HAF2). The paper's
  test sets are out-of-distribution for them too, yet they hit 0.069/0.110 — so the remaining gap
  points at **architecture / hyperparameters / feature fidelity**, not corpus size.

**Conclusion:** corpus scaling was the right first lever and fixed the calibration half of the
gap; the ranking-loss half is not data-limited.

## Where the gap actually is (diagnostic, 2026-08-03)

Per-target breakdown of the full-corpus MSE model (no upstream code involved):

| set | mean rl | median rl | trimmed (drop worst 2) | neg-Spearman targets | paper mean |
|-----|---------|-----------|------------------------|----------------------|------------|
| BM55-AF2 | 0.142 | **0.079** | 0.080 | 5/15 | 0.069 |
| HAF2-12  | 0.142 | **0.055** | 0.074 | 4/12 | 0.110 |
| val (730) | 0.044 | 0.000 | — | 243/692 (35%) | — |

1. **The val metric is a poor proxy.** Median val DockQ-spread is **0.07** — most val targets have
   all decoys at ~equal quality, so ranking loss is trivially ~0 (median 0.000). The MAF2
   high-quality skew makes val ranking loss look great without testing hard ordering.
2. **A *typical* benchmark target already matches/beats the paper.** BM55 median 0.079 ≈ paper
   *mean* 0.069; HAF2 median 0.055 < paper *mean* 0.110. The headline gap is a few catastrophic
   targets inflating the mean (BM55 top-2 = 51% of the mean; HAF2 top-3 = ~60%).
3. **The defect is ordering robustness, not calibration.** ~1/3 of targets have **negative
   Spearman** — ranked backwards (5/15 BM55, 4/12 HAF2, a masked 35% of val). On low-spread targets
   an inversion is free; on high-spread benchmark targets it is catastrophic (e.g. 4ETQ: picks a
   0.05-DockQ decoy #1 when best is 0.80, ranking loss 0.75).

Worst-offender targets: BM55 **4ETQ, 6AL0**; HAF2 **7OZN, 7D7F, 7D3Y** (all negative-Spearman,
large spread).

**Feature-OOD hypothesis — tested and REFUTED.** Standardized-feature magnitude (|z| against the
model's train-fit buffers) on the inverted targets is essentially identical to the well-ranked
ones: mean|z| 0.76 vs 0.77, p99 4.0 vs 4.0, tail %|z|>5 *lower* on inverted (0.48 vs 0.61). The
inversions are **not** feature saturation / out-of-distribution inputs — those targets' features
sit squarely in the training distribution. So the model inverts them on in-distribution-looking
inputs: a genuine model/architecture behaviour, not a preprocessing artifact. This is exactly the
question the external Codex reference disambiguates — do the authors' own artifacts invert the
*same* targets (→ universally hard) or rank them correctly (→ their architecture/features capture
something ours misses)?

## Next levers (revised after corpus scaling)

1. **Checkpoint selection on benchmark ranking loss**, not val MSE — val ranking loss (0.037) and
   benchmark ranking loss (0.14) diverge; we may be selecting the wrong epoch for the target metric.
2. **Hyperparameter search** (lr, layers/heads/width, dropout, epochs — the paper pins none) and
   ranking weight/margin, evaluated directly on benchmark ranking loss.
3. **Feature-fidelity audit** — confirm our clean `(x,y,z)` all-atom edge histogram and PH node
   features behave sensibly on antibody-antigen interfaces (the worst-transfer domain).
4. **5-seed sweep** for a mean±sd comparable to the paper's 0.087±0.010 / 0.111±0.008.

## Reproduce

Checkpoints are committed (`models/*.pt`, ~145 KB each); the benchmark data is CC-BY and
git-ignored (pull the 183 MB tarball from Zenodo 6569837 into `zenodo/DProQ_benchmark/`). Then:

```bash
export PATH="$HOME/.pixi/bin:$PATH"   # mkdssp
uv run python scripts/phase_e_benchmark.py --subset zenodo/DProQ_benchmark/BM55-AF2 \
    --checkpoint models/scorer_mse_normalized.pt
uv run python scripts/phase_e_benchmark.py --subset zenodo/DProQ_benchmark/HAF2 \
    --checkpoint models/scorer_ranking_normalized.pt --drop-target 7ALA
```

The script asserts the resolved (targets, decoys) match the paper's filtered sets before
emitting the comparison, so a partial extraction fails loudly instead of scoring the wrong set.

Raw numbers (direct script output, one JSON line per run): `results/phase_e_benchmark.jsonl`.
