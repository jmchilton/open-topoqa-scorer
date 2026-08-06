# P6b — competitiveness leaderboard (issue #12)

**Question:** is the topological interface-QA step *useful* vs. what a practitioner would otherwise
run — not "does topology add signal within our model" (that is P6a), but "does the whole model beat
the cheap standard baseline?"

**Date:** 2026-08-05. Baselines and our-model rows in **Table A** are computed here on the identical
DProQ-benchmark decoys with the identical per-target ranking metrics (`evaluate.py`), so they are
strictly head-to-head. **Table B** is published context and is *not* recomputed — read its caveats.

---

## Table A — controlled, same decoys + same metric (recomputed here)

Our model = `scorer_full_mse` (correct `(x,y,z)` coordinates), from `results/phase_e_full_corpus.jsonl`.
pLDDT arms = mean AlphaFold-Multimer pLDDT (per-residue B-factor) over interface residues / all
residues, ranked per target; training-free. Higher = predicted-better. `scripts/phase_g_leaderboard.py`.

### DBM55-AF2 (15 targets / 449 decoys)

| Method | ranking loss ↓ | Spearman ↑ | top-10 success ↑ |
|---|---|---|---|
| DProQA (same-protocol) | **0.060** | 0.159 | 0.800 |
| **Topological (ours)** | 0.142 | **0.291** | **0.867** |
| pLDDT (interface) | 0.139 | 0.047 | 0.800 |
| pLDDT (global) | 0.139 | 0.123 | 0.800 |

### HAF2-12 (12 targets / 1270 decoys)

| Method | ranking loss ↓ | Spearman ↑ | top-10 success ↑ |
|---|---|---|---|
| DProQA (same-protocol) | 0.200 | 0.008 | 0.833 |
| **Topological (ours)** | 0.142 | **0.249** | 0.833 |
| pLDDT (interface) | 0.143 | 0.001 | 0.833 |
| pLDDT (global) | **0.136** | −0.094 | 0.833 |

DProQA = the trained non-topological competitor, run on these exact decoys via the same metric — see
`results/dproqa_rerun.md`. Its published rows (Table B) hold up decoy-for-decoy (0.049→0.060,
0.195→0.200), so it is promoted here into Table A.

**Reading Table A — the honest finding:**

1. **On top-1 ranking loss, a training-free AF-confidence baseline is competitive with our model on
   both sets** (pLDDT-global 0.139 / 0.136 vs ours 0.142 / 0.142). *Caveat — "ours" here is the single
   shipped `scorer_full_mse` checkpoint, which sits at the bad end of the seed spread: the same
   architecture's 3-seed mean (P6a) is **0.126** on BM55, which actually beats pLDDT's 0.139. So on
   BM55 the "tie/edge" is within seed-noise and if anything favors ours; on HAF2 pLDDT is genuinely at
   least as good (P6a full mean 0.164 > 0.136).* Either way, the topological step buys ~nothing
   **reliably** on this one metric — which is the point: it is saturated.
2. **But top-1 ranking loss is nearly uninformative here.** pLDDT achieves it with ~zero or *negative*
   rank correlation (Spearman 0.05 / 0.12 on BM55, 0.00 / −0.09 on HAF2) — it is not ordering decoys by
   quality, it just avoids a catastrophic #1 pick on these AF-generated pools (where the most-confident
   model is usually decent). This matches the earlier replication finding that the BM55 gap was two
   targets' top-1 picks — the metric saturates.
3. **A trained non-topological model (DProQA) beats the topological one on top-1 — and it *also* barely
   ranks.** DProQA's same-protocol top-1 ranking loss is **0.060 on BM55** (less than half our 0.142,
   better than pLDDT) and 0.200 on HAF2. But its Spearman is 0.159 (BM55) and **0.008 (HAF2) — zero**.
   So the second, stronger competitor repeats pLDDT's pattern: it wins the headline metric while
   carrying essentially no rank information on the harder set. Two independent competitors beating top-1
   ranking loss with near-zero rank correlation is decisive evidence the metric is saturated.
4. **Where topology earns its keep is ranking quality, not top-1.** Our model is the only method with
   meaningful positive Spearman on **both** sets (0.25–0.29) — higher than DProQA on both and the only
   positive number on HAF2 — and it wins top-10 success on BM55 (0.87 vs 0.80). If you are selecting
   top-k or trusting the ordering, the topological model is genuinely the best ranker here; if you only
   ever take the single top pose, DProQA (or even free pLDDT) is better.

*(Note: the ranking-loss-trained variant `scorer_full_ranking` does marginally better on ranking loss —
BM55 0.140, HAF2-12 0.133 — with similar Spearman; it does not change the conclusion.)*

---

## Table B — published numbers (context only; NOT recomputed, NOT same footing)

Reported ranking loss from the TopoQA paper's comparison tables (arXiv 2410.17815, Tables S1/S3) and
the DProQ paper (arXiv 2205.10627). **These use each tool's own evaluation pipeline** — do not read
them as head-to-head with Table A.

| Method | DBM55-AF2 | HAF2 |
|---|---|---|
| TopoQA *(published — see caveat)* | 0.069 | 0.119 |
| DProQA | 0.049 | 0.195 |
| AF2Rank (AF-Multimer based) | 0.261 | 0.125 |
| GOAP | 0.123 | — |
| ZRANK2 | 0.372 | 0.153 |
| GNN-DOVE | 0.365 | — |
| ComplexQA | 0.26 | — |
| TRScore | 0.292 | — |

**Caveats (why Table B is not a leaderboard with Table A):**

- **The published TopoQA row is the `(x,y,y)`-defect-inflated number.** Our clean-room reproduction
  with corrected `(x,y,z)` coordinates scores ~0.142 on both sets (Table A), not 0.069/0.119 — the
  published figure leans on the released coordinate bug (yubingapril/TopoQA#1) plus early-epoch
  selection. So the paper's own row overstates the topological approach.
- **The other tools' numbers come from their own papers'** filtering/metric, which we have not
  verified matches ours decoy-for-decoy. DProQA's 0.049 on DBM55-AF2 (a non-topological gated graph
  transformer beating everything on that set) was the standout — and it has now been **re-run
  same-protocol** (moved to Table A): it holds (0.060 / 0.200 vs published 0.049 / 0.195). The
  remaining Table B rows (AF2Rank, GOAP, ZRANK2, GNN-DOVE, ComplexQA, TRScore) are still unverified.

---

## Verdict

- **Is the step competitive?** On the practical "rank a target's decoys" task, **two independent
  competitors beat or match our model on top-1 ranking loss**: free AF-Multimer pLDDT trades blows
  within seed-noise, and the trained non-topological DProQA wins outright on BM55 (0.060 vs 0.142).
  **But both do it with near-zero rank correlation** (pLDDT Spearman ≈0.05–0.12 / −0.09; DProQA 0.159 /
  0.008). On the metrics that reflect actual ranking *quality* (Spearman, top-10), the topological model
  is the best ranker of the three on both sets — the only one with positive Spearman on HAF2.
- **What that means for the metric:** top-1 ranking loss — the QA literature's headline — is saturated;
  it is won here by methods that do not rank. The topological approach's defensible value is
  rank-ordering quality, which that headline metric hides.
- **Practical guidance that falls out:** if a user only wants the single best pose, DProQA (or even free
  AF pLDDT) is competitive-to-better; the topological QA step pays off when they want a *trustworthy
  ordering* or a top-k shortlist.
- **The literature's headline overstates topology** (bug-inflated TopoQA row), while DProQA's strong
  DBM55-AF2 number **holds up** under a same-protocol re-run (`results/dproqa_rerun.md`) — that deferred
  arm is now done.

## Notes / deferred (from #12)
- **`global_plddt` is unguarded against hetero residues** (waters/ligands with junk B-factors would
  dilute it); AF-Multimer decoys here are protein-only so it is harmless in practice, and the primary
  `interface_plddt` arm is safe (it only touches `interface_nodes`).
- **P6a attribution ablation** is done — see `results/p6a_ablation.md`. It complements this competitive
  view: topology's value is generalizable rank quality, not top-1.
- **Same-protocol external re-run of DProQA** is **done** — see `results/dproqa_rerun.md` and Table A.
  The 0.049 / 0.195 claims hold decoy-for-decoy (0.060 / 0.200).
- **AF2Rank is deferred to a GPU machine, not run here.** It requires the full AlphaFold2 codebase +
  ~5 GB of params and runs an AF-Multimer forward pass per decoy — infeasible on this GPU-less arm64
  laptop (days–weeks of CPU over ~1720 decoys), and AF2Rank ships monomer-only so the complex setup is
  itself an adaptation. Marginal value is low: its published BM55 loss (0.261) is already the worst
  trained method here, and its core idea — *rank by AlphaFold confidence* — is already represented in
  Table A by the training-free pLDDT arm. Revisit on GPU only if a full leaderboard is wanted.
- The other Table B tools (GOAP, ZRANK2, GNN-DOVE, ComplexQA, TRScore) remain unverified.
