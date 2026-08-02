# open-topoqa-scorer

Open **clean-room** retrain of the [TopoQA](https://doi.org/10.1093/bib/bbaf083)
interface-quality scorer (Han et al., 2025, *Briefings in Bioinformatics* 26(2), bbaf083).
MIT-licensed. Consumes the interface graphs emitted by
[`open-topoqa-featurizer`](https://github.com/jmchilton/open-topoqa-featurizer).

**Provenance.** The model architecture and training protocol are reproduced **from the
paper** (arXiv 2410.17815 / bbaf083, including its supplementary tables S1–S11) — **not**
from the upstream code, which is unlicensed
(`yubingapril/TopoQA`) and was **not read, cloned, or decompiled**. Architectures and
training recipes are not copyrightable; this independent implementation is ours, MIT.
Part of the [bio-topo-foundry](https://github.com/jmchilton/bio-topo-foundry) cleanroom
epic (#1), issue #5.

## Why a retrain exists

TopoQA's checkpoint + inference code ship without a software license, and the featurizer it
depended on carried a coordinate defect (fixed by construction in the sibling featurizer).
Retraining the model from the paper spec yields a **fully redistributable** structure-QA
vertical (foundry pipeline P1) that never touches the unlicensed checkpoint. Because the
inputs differ from the released checkpoint (corrected featurizer), matching upstream weights
is a **non-goal** — the target is a working, open, paper-faithful scorer.

## Status — Phase A (modeling machinery; not data-gated)

- ✅ **ProteinGAT model** (`model.ProteinGAT`): faithful to the paper (§5.2.5, Eqs 3–9) —
  additive edge-conditioned multi-head attention (`GATv2Conv`, `edge_dim`; the faithful reading
  of Eq. 3's `σ(W_s x_i + W_t x_j + W_e e_ij)`) that **updates both
  node and edge embeddings each layer** (Eq. 6: `e_ij ← Θ_e·[x_i‖x_j‖e_ij]`); the *updated*
  edges are mean-pooled and reduced to **half the node width** (Eq. 8), concatenated with the
  mean-pooled nodes, and fed to a **three-linear MLP → sigmoid** (Eq. 9). Node in-dim **172**,
  edge **11**.
- ✅ **Data adapter** (`data.graph_from_featurized`): featurizer dict → PyG `Data`
  (symmetrized bidirectional edges, DockQ target).
- ✅ **Training loop** (`train.train_model`): Adam, MSE vs DockQ, LR 0.005, 200 epochs,
  best-validation selection, seeded.
- ✅ **Metrics** (`metrics`): ranking loss (top-1), Pearson, Spearman, top-N CAPRI hit rate +
  target-level success rate.
- ✅ **Red-to-green tests** (32): overfit-a-tiny-batch (+ a node-path-isolating variant),
  edge-update gradient/perturbation, batched-vs-solo on uneven edges, best-val restore +
  divergence guard, paper-structural invariants, metric unit tests. Hermetic where possible;
  a real-`mkdssp` benchmark test runs when the data is present.

## Status — Phase B (pipeline proof on real CC-BY benchmark data)

- ✅ **Benchmark loader** (`benchmark`): reads the DProQ benchmark (Zenodo 6569837,
  CC-BY-4.0) — `label_info.csv` (`Target,Model,DockQ,CAPRI`) + `decoy/<TARGET>/<MODEL>*.pdb`
  (tolerating the `_tidy` filename suffix), featurizes with caching.
- ✅ **Pipeline proof** (`scripts/phase_b_smoke.py`): featurized 59 real BM55-AF2 decoys
  (172-dim nodes / 11-dim edges), trained on one target (3SE8: train Spearman +0.76), and
  produced per-target rankings on a held-out target. This is a **smoke** that the pipeline
  featurizes real decoys and the loop trains + ranks — a single held-out target is far too
  noisy to compare architectures or estimate accuracy (that waits for full training, Phase D).

The benchmark data is CC-BY but too large to vendor, so it is git-ignored and pulled locally;
`tests/test_benchmark.py` exercises the loader on it when present and skips otherwise.

**Not yet done (later phases, see foundry #5):** the MAF2 training-decoy gate (Phase C) and
the full train/eval/ship (Phase D).

## What the paper fixes vs. what we choose (clean-room)

The paper's Methods (§5.2.5) **do** pin the model *structure*, and the implementation follows
it exactly:

- **Edge-updating attention** (Eqs 3, 6) — attention uses `x_i, x_j, e_ij`, and **edges are
  updated every layer** from the two updated endpoint embeddings + the previous edge. Neither
  stock PyG conv updates edges, so it is an explicit per-layer module.
- **Readout** (Eqs 7–9) — pool the *updated* edges (not raw attrs); reduce to **half the node
  width**; concat with pooled nodes; **three-linear MLP**; sigmoid. Loss is **MSE** vs DockQ.

Neither the paper's main text nor its supplementary tables (S1–S11, which are results/ablation
tables) states **any** value for the training/capacity hyperparameters — no number for layers,
heads, hidden/edge width, dropout, optimizer, learning rate, epochs, or batch size (only the
**MSE** loss is pinned, §5.2.5). These are therefore **our tunable choices**, *not* paper-derived
— pulled from common practice and to be tuned on validation, never claimed as reproductions:

- node-attention sublayer `GATv2Conv` (default, the additive Eq. 3 form; `TransformerConv` via
  `conv="transformer"` is a dot-product **divergence**, offered only as a knob),
  `heads=4` averaged back to `hidden=32` (`concat=False`; GATv2's additive attention is
  seed-unstable at `heads=8` on small graphs, so 4 is the default), `edge_hidden=16`, `num_layers=2`,
  `dropout=0.25`; training `Adam`, `lr=0.005`, `200` epochs, `batch_size=32`.
- A prior clean-room source for reasonable defaults is DProQA's gated graph transformer (a
  separate, cited paper) — TopoQA reuses its train/val/test split but not, as stated, its
  hyperparameters.

## Develop

```bash
uv venv --python 3.11 && uv pip install -e . pytest
.venv/bin/python -m pytest -q
```
