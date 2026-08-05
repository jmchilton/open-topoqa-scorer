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
| **Topological (ours)** | 0.142 | **0.291** | **0.867** |
| pLDDT (interface) | 0.139 | 0.047 | 0.800 |
| pLDDT (global) | **0.139** | 0.123 | 0.800 |

### HAF2-12 (12 targets / 1270 decoys)

| Method | ranking loss ↓ | Spearman ↑ | top-10 success ↑ |
|---|---|---|---|
| **Topological (ours)** | 0.142 | **0.249** | 0.833 |
| pLDDT (interface) | 0.143 | 0.001 | 0.833 |
| pLDDT (global) | **0.136** | −0.094 | 0.833 |

**Reading Table A — the honest finding:**

1. **On top-1 ranking loss, a training-free AF-confidence baseline ties or edges our model on both
   sets** (pLDDT-global 0.139 / 0.136 vs ours 0.142 / 0.142). The topological step buys ~nothing on
   this one metric.
2. **But top-1 ranking loss is nearly uninformative here.** pLDDT achieves it with ~zero or *negative*
   rank correlation (Spearman 0.05 / 0.12 on BM55, 0.00 / −0.09 on HAF2) — it is not ordering decoys by
   quality, it just avoids a catastrophic #1 pick on these AF-generated pools (where the most-confident
   model is usually decent). This matches the earlier replication finding that the BM55 gap was two
   targets' top-1 picks — the metric saturates.
3. **Where topology earns its keep is ranking quality, not top-1.** Our model is the only one with
   meaningful positive Spearman (0.25–0.29) and it wins top-10 success on BM55 (0.87 vs 0.80). If you
   are selecting top-k or trusting the ordering, the topological model is genuinely better than AF
   confidence; if you only ever take the single top pose, it is not.

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
  transformer beating everything on that set) is the standout — but it is exactly the kind of claim
  that needs a **same-protocol re-run** before we trust it against Table A.

---

## Verdict

- **Is the step competitive?** On the practical "rank a target's decoys" task, **AF-Multimer pLDDT is a
  strong, free baseline that our topological model does *not* beat on top-1 ranking loss** — but that
  metric is saturated on these sets. On the metrics that reflect actual ranking quality (Spearman,
  top-10), the topological model is clearly the better ranker than AF confidence.
- **Practical guidance that falls out:** if a user only wants the single best pose, plain AF confidence
  is competitive and free; the topological QA step pays off when they want a *trustworthy ordering* or
  a top-k shortlist.
- **The literature's headline overstates topology** (bug-inflated TopoQA row), and a strong
  non-topological trained model (DProQA) reports the best DBM55-AF2 number of all — motivating the one
  deferred arm below.

## Deferred (from #12)
- **Same-protocol external re-run** of DProQ/DProQA (and, if recoverable, AF2Rank) on these identical
  decoys, to promote Table B rows into Table A. This is the fair way to test the DProQA 0.049 claim.
- **P6a attribution ablation** (full / conventional-32 / topological-140) — isolates what the PH block
  contributes *within* our model; complementary to this competitive view.
