# P6b same-protocol external arm — DProQA on the identical decoys (issue #12)

**Question:** Table B of `p6b_leaderboard.md` carried DProQA's *published* ranking losses
(DBM55-AF2 **0.049**, HAF2 **0.195**) — a non-topological gated-graph-transformer that reports the
best DBM55-AF2 number of any method. Those come from DProQA's own pipeline, not our decoy-for-decoy
metric. This arm runs **DProQA's own inference** on the **same** DProQ-benchmark decoys we score our
model on, then pushes its `PRED_DOCKQ` through the **same** `per_target_ranking_metrics` — promoting
the Table B rows into strictly head-to-head **Table A**.

**Date:** 2026-08-05. Driver: `scripts/phase_g_dproqa.py` (shells out to DProQA per target; only the
*scoring* is ours). Results: `results/phase_g_dproqa.jsonl`.

## What DProQA is

DProQA (Chen, Morehead, Liu, Cheng — *Bioinformatics* 39 Suppl. 1, ISMB 2023;
`jianlin-cheng/DProQA`, successor to `BioinfoMachineLearning/DProQ`). A Gated Graph Transformer over a
KNN residue graph. Node features: sequence one-hot + DSSP SS3/RASA/(φ,ψ) + Laplacian positional
encoding (N×35); edge features from CA-CA / CB-CB / N-O distance maps (E×6). Multitask head predicts a
DockQ regression score + a CAPRI class. Shipped checkpoint `pre_train_seed_222.ckpt` (139,673 params).
It is *trained* (unlike the pLDDT baseline) but carries **no persistent-homology features** — the clean
"non-topological learned QA" contrast to our model.

## Porting DProQA to run here (arm64 macOS / M4, CPU)

Upstream `environment.yml` pins a 2022 CUDA stack (torch 1.10-cu113, dgl-cuda11.1 0.8.1, DSSP 3.0)
that cannot run on Apple Silicon and needs no GPU for inference-only. Ported to a modern arm64/CPU
stack in the sibling repo's `.venv` (pins in `DProQA/requirements-m4.txt`). Changes, all minimal and
documented in-place:

| Layer | Upstream | Shim (why) |
|---|---|---|
| torch / dgl | 1.10-cu113 / dgl-cuda 0.8.1 | torch 2.1.0 + dgl 2.2.0 (CPU, arm64); newest dgl graphbolt dylib is for torch ≤2.3 |
| setuptools | (implicit) | 65.5.0 — keeps both `pkg_resources.packaging` (torch 2.1) and `setuptools.extern` (dgl) |
| numpy / torchdata | — | numpy 1.26 (torch 2.1 ABI); torchdata 0.7.1 (dgl datapipes) |
| lightning / torchmetrics | 1.6.3 / 0.8.2 | pl 1.9.5 (keeps `pl.utilities.seed`); tm 0.10.3 (`Accuracy()` w/o `task=`) |
| device | `torch.device('cuda')` hardcoded | `cuda if available else cpu` + `map_location` on load |
| DataLoader | `num_workers=4, pin_memory=True` | `0` / `False` — DGL-graph workers crash under macOS multiprocessing; pinning is CUDA-only |
| DSSP | 3.0.0 | mkdssp 4.6.1 (brew); the extra `P` (polyproline-II) SS code 4.x emits reduced to coil in `ss3_one_hot` (standard SS3) |
| `dgl.function` | `src_mul_edge`, `copy_edge` | `u_mul_e`, `copy_e` (renamed in DGL 1.x+) |
| Laplacian PE | old torch-sparse `.coalesce()`/`.values()` | dense adjacency from `g.edges()` (dgl 2.x `adjacency_matrix()` returns a `SparseMatrix`) |

The shipped checkpoint loads and evals unchanged (Lightning auto-upgrades v1.5.10→v1.9.5); the shims
are environment/API compatibility only, not model changes. The forward pass and all feature generation
remain DProQA's.

## Protocol (identical to our model's rows)

- **Same decoys:** DProQA's `inference.py -c <decoy_folder>` scores every PDB in a target's folder;
  we point it at the exact `DProQ_benchmark` decoy dirs our loader resolves (BM55-AF2 `decoy/<t>/`,
  HAF2 `decoy/<t>/pdb/`). BM55-AF2 = 15 targets / 449 decoys; HAF2 drops 7ALA → 12 targets / 1270
  decoys — matching Table A exactly.
- **Same metric:** `PRED_DOCKQ` joined to our `DecoyLabel` by model name (a trailing `_tidy` from
  DProQA's clean step is stripped), then `per_target_ranking_metrics` + `pooled_regression_metrics`.
- **Coverage:** every decoy is expected to score; any DProQA feature-gen failure is reported as
  `missing` and assigned a worst-rank sentinel (never picked top-1) so it cannot silently inflate.

## Results — Table A (recomputed, same decoys + same metric)

Both subsets scored with **0 missing decoys** (full coverage). `results/phase_g_dproqa.jsonl`.

| Method | DBM55-AF2 rank-loss ↓ | Spearman ↑ | top-10 ↑ | HAF2-12 rank-loss ↓ | Spearman ↑ | top-10 ↑ |
|---|---|---|---|---|---|---|
| DProQA **(same-protocol, ours)** | **0.060** | 0.159 | 0.800 | **0.200** | 0.008 | 0.833 |
| DProQA *(published, Table B)* | 0.049 | — | — | 0.195 | — | — |
| Topological (ours) | 0.142 | **0.291** | 0.867 | 0.142 | **0.249** | 0.833 |
| pLDDT (global) | 0.139 | 0.123 | 0.800 | 0.136 | −0.094 | 0.833 |

## Verdict

**1. DProQA's published numbers hold up decoy-for-decoy.** Same-protocol 0.060 / 0.200 vs published
0.049 / 0.195 — within a few points on both sets. The small gap is consistent with the DSSP-4-vs-3
feature shift and the arm64/CPU port, not a protocol inflation. This is the honest-baseline mirror
image of the TopoQA row: TopoQA's published 0.069/0.119 *collapsed* to ~0.142 once the `(x,y,y)`
coordinate bug was removed, whereas DProQA's numbers **survive** a same-decoy re-run. Table B's DProQA
row can be promoted into Table A with confidence.

**2. On the literature's headline metric, a non-topological trained model beats the topological one
outright.** DProQA's 0.060 top-1 ranking loss on DBM55-AF2 is less than half our topological model's
0.142 (and better than the free pLDDT baseline's 0.139). On HAF2 it is 0.200 vs our 0.142 — here the
topological model wins, but see below.

**3. …yet DProQA barely ranks.** Its Spearman is 0.159 on BM55 and **0.008 — statistically
indistinguishable from zero — on HAF2** (pooled Pearson 0.04). Like pLDDT (Spearman −0.09 on HAF2), it
scores well on top-1 ranking loss while carrying almost no rank information: it reliably avoids a
catastrophic #1 pick on these AF pools without ordering the rest. Our topological model is the only
method with meaningful positive rank correlation on **both** sets (0.29 / 0.25).

**4. The combined P6 conclusion, sharpened.** Two *different* strong competitors — a free confidence
baseline (pLDDT) and a trained non-topological model (DProQA) — both beat or match the topological
model on top-1 ranking loss, and **both do it with near-zero rank correlation.** That is the strongest
available evidence that top-1 ranking loss, the metric the QA literature leads with, is saturated and
actively rewards non-ranking behaviour. The topological approach's genuine, defensible edge is
*rank-ordering quality* (Spearman / top-k), where it leads every competitor here — not the top-1
numbers, where it does not. Enabling the honest comparison required running the competitor; the
competitor's own headline turns out to be the weakest part of the story.

*(Metric note: top-1 ranking loss on 12–15 targets is high-variance — see P6a, where the same
architecture's BM55 loss swings 0.09–0.15 across seeds. DProQA's BM55 win is large enough to clear
that noise; its HAF2 loss is not distinguishable from ours. The Spearman gaps are the stable signal.)*

## Deferred: AF2Rank (to GPU)

The other standout Table B row, **AF2Rank** (0.261 / 0.125), is **not** run here. It needs the full
AlphaFold2 codebase + ~5 GB params and an AF-Multimer forward pass per decoy — infeasible on this
GPU-less M4 (days–weeks of CPU over ~1720 decoys) and AF2Rank ships monomer-only, so complexes are an
adaptation on top. Its marginal value is also low: the published 0.261 is already the worst trained
method here, and its premise (*rank by AlphaFold confidence*) is already captured by the training-free
pLDDT arm in Table A. Deferred to a GPU machine if a fuller external leaderboard is ever wanted.
