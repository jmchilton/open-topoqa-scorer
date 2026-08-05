# P6a — attribution ablation: does the persistent-homology block add signal? (issue #12)

**Question:** *within* the model (not vs. external baselines — that is P6b), what does the 140-dim
persistent-homology node block actually contribute? Isolated by training the **same** ProteinGAT
under the **same** protocol on three node-feature slices of the **same** corpus, only the columns
differing:

- **full** — node[0:172] = 32 conventional + 140 topological (the shipped model)
- **conventional_only** — node[0:32] = 21 AA + 8 DSSP SS + 1 SASA + 2 (φ,ψ)
- **topological_only** — node[32:172] = 7 element channels × 20 persistence summaries

Held identical across arms: graph, GNN kwargs (`gatv2`, heads 4, hidden 32, 2 layers, dropout 0.25),
corpus (11,695 train / 2,925 val, leak-free MMseqs2 split), lr 0.005, 40 epochs, best-val-MSE
selection, per-feature standardization. **3 seeds (0,1,2)**; values below are mean [min, max].
`scripts/phase_g_ablation.py` + `scripts/aggregate_ablation.py`.

Top-1 ranking loss on 12–15 targets is inherently high-variance, so it is read only as a mean with
the spread shown; the interpretable signal is rank correlation (Spearman), where several arm
differences are **non-overlapping across all three seeds** (marked ✓robust).

## Results (mean [min, max] over 3 seeds)

### val (in-distribution)
| arm | ranking loss ↓ | Spearman ↑ | top-10 ↑ | pooled r |
|---|---|---|---|---|
| full | 0.043 [.041,.045] | 0.259 [.237,.272] | 0.860 | 0.837 |
| conventional_only | **0.040** | **0.318 [.307,.329]** | 0.861 | 0.834 |
| topological_only | 0.044 | 0.270 [.254,.298] | 0.860 | 0.834 |

### BM55-AF2 (15 targets)
| arm | ranking loss ↓ | Spearman ↑ | top-10 ↑ | pooled r |
|---|---|---|---|---|
| full | 0.126 [.094,.146] | 0.282 [.265,.312] | 0.844 | 0.562 |
| conventional_only | **0.103 [.086,.134]** | 0.300 [.248,.353] | 0.822 | **0.599** |
| topological_only | 0.130 | **0.330 [.323,.338]** | 0.822 | 0.589 |

### HAF2-12 (12 targets)
| arm | ranking loss ↓ | Spearman ↑ | top-10 ↑ | pooled r |
|---|---|---|---|---|
| full | 0.164 [.133,.184] | 0.215 [.182,.237] | 0.833 | **0.683** |
| conventional_only | 0.159 | 0.125 [.114,.145] | **0.917** | 0.602 |
| topological_only | **0.151** | **0.249 [.181,.294]** | 0.833 | 0.672 |

**Δ on the mean (Spearman):** val full−conv **−0.058**, topo−conv −0.048 · BM55 full−conv −0.018,
topo−conv +0.030 · HAF2 full−conv **+0.091**, topo−conv **+0.124**.

## What the PH block actually does

1. **Top-1 ranking loss cannot attribute anything here.** Ranges overlap across every arm on every set
   (full BM55 swings 0.094–0.146 across seeds alone), and adding topology never reliably lowers it — if
   anything it is slightly worse in-distribution. This corroborates P6b: the metric is saturated.

2. **In-distribution, topology *reduces* rank quality (✓robust).** On val, `conventional_only` Spearman
   0.318 [.307,.329] sits entirely **above** `full` 0.259 [.237,.272] — non-overlapping across all 3
   seeds. The 32 conventional features already capture the training distribution; the topological block
   adds noise there. *(Caveat: `val` doubles as the best-val-MSE checkpoint-selection set — there is no
   third split — so this rests on less-independent data than the external sets; the selection metric is
   MSE not Spearman and identical across arms, so it is not circular, but the load-bearing OOD evidence
   is HAF2/BM55, not this row.)*

3. **Out-of-distribution on HAF2, topology *increases* rank quality (✓robust).** `conventional_only`
   Spearman 0.125 [.114,.145] sits entirely **below** both `full` 0.215 [.182,.237] and
   `topological_only` 0.249 — non-overlapping. Pooled Pearson agrees (full/topo ~0.68 vs conv 0.60).
   Topology is the more **generalizable** representation: it earns its keep on the harder set the
   conventional features generalize to worst.

4. **The 32 conventional features are a far stronger baseline than the literature implies.** They give
   the best in-distribution Spearman, the best BM55 mean ranking loss, and the best HAF2 top-10 of any
   arm. Much of what a topological QA model "knows" is recoverable from AA identity + DSSP SS/SASA +
   backbone angles.

5. **Combining the blocks (full) rarely beats the better single block on rank correlation** — on both
   benchmark sets `topological_only` out-Spearmans `full`. At this capacity the model does not cleanly
   fuse the two representations; more capacity or better regularization might, but that is speculation.

## Verdict

The persistent-homology block's contribution is **real but narrow and directional**: it *trades
in-distribution ranking for out-of-distribution generalization*, helping clearly on HAF2 rank
correlation, being neutral-to-harmful in-distribution and on the saturated top-1 metric, and never
rescuing top-1 ranking loss. Crucially, that benefit is carried by topology **alone** —
`topological_only` out-Spearmans `full` on *both* benchmark sets, so fusing the conventional block back
in (the shipped `full` model) does not improve rank-ordering; the shipped model's ranking is
essentially topology-dominated. There is **no uniform "topology adds signal"** — the honest statement is
"topology-alone adds *generalization* to interface rank-ordering, conventional features are a strong
baseline, and combining them doesn't beat the better single block." Together with P6b (a free pLDDT
baseline trades top-1 blows within seed-noise), the two halves converge: the topological approach's
genuine value is generalizable *ranking quality*, not the top-1 numbers the literature leads with.

## Caveats / deferred
- Single corpus + split; 3 seeds (enough to show top-1 is noise and the two ✓robust Spearman gaps, not
  enough to tighten BM55). More seeds would narrow the BM55 picture.
- "Held identical across arms" has one nuisance exception: because `node_dim` differs, `ProteinGAT`'s
  input `Linear` draws a different number of global-RNG values at init, desynchronizing the shuffle
  RNG, so each arm sees a different minibatch order at the same seed. Absorbed into the 3-seed average;
  not a per-arm advantage.
- Plain MSE objective, best-val-MSE selection (shipped model's best epoch ≈ 2–25; 40 amply covers it).
- Deferred (from #12): same-protocol external re-run (P6b Table B) and a `petls` persistent-Laplacian
  feature arm (topology-flavor vs topology-flavor).
