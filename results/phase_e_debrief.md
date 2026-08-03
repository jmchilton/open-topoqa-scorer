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
  **subsample**, not the paper's full 8,733): `ckpt_rank_mse` (pure MSE, the paper's loss)
  and `ckpt_rank` (MSE + within-target ranking loss).

## Results vs paper

| set | ckpt | ranking loss ↓ | Spearman ↑ | per-tgt Pearson ↑ | pooled Pearson | top10 succ |
|-----|------|---------------|-----------|-------------------|----------------|-----------|
| BM55-AF2 | **paper** | **0.069** | **0.502** | **0.515** | — | — |
| BM55-AF2 | mse  | 0.137 | 0.287 | 0.270 | 0.535 | 0.933 |
| BM55-AF2 | rank | 0.212 | 0.270 | 0.274 | 0.403 | 0.867 |
| HAF2-12  | **paper** | **0.110** | **0.675** | **0.600** | — | — |
| HAF2-12  | mse  | 0.148 | 0.204 | 0.368 | 0.560 | 0.846 |
| HAF2-12  | rank | **0.121** | 0.156 | 0.387 | 0.393 | 0.769 |

## Read

- **Same order of magnitude; one number nearly lands.** HAF2 ranking loss with the ranking
  checkpoint is **0.121 vs paper 0.110**. BM55 is ~2× the paper (0.137 vs 0.069).
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

## Next levers (in impact order)

1. **Scale the training corpus toward the full 8,733** (re-featurize the un-subsampled Phase C
   corpus). Biggest expected gain on correlations + ranking loss.
2. **Tune** ranking weight/margin and select checkpoints on benchmark ranking loss, not val MSE.
3. **5-seed sweep** for a mean±sd comparable to the paper's 0.087±0.010 (BM55) / 0.111±0.008 (HAF2).

Raw numbers: `results/phase_e_benchmark.jsonl`.
