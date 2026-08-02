# open-topoqa-scorer

Open **clean-room** retrain of the [TopoQA](https://doi.org/10.1093/bib/bbaf083)
interface-quality scorer (Han et al., 2025, *Briefings in Bioinformatics* 26(2), bbaf083).
MIT-licensed. Consumes the interface graphs emitted by
[`open-topoqa-featurizer`](https://github.com/jmchilton/open-topoqa-featurizer).

**Provenance.** The model architecture and training protocol are reproduced **from the
paper and its supplement** — **not** from the upstream code, which is unlicensed
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

- ✅ **ProteinGAT model** (`model.ProteinGAT`): 2-layer graph-attention encoder with
  edge-conditioned attention (PyG `TransformerConv`/`GATv2Conv`, `edge_dim=11`), 8 heads,
  hidden 32, dropout 0.25; node + edge representations mean-pooled separately, concatenated,
  and passed through an MLP → sigmoid score in [0, 1]. Node in-dim **172**, edge **11**.
- ✅ **Data adapter** (`data.graph_from_featurized`): featurizer dict → PyG `Data`
  (symmetrized bidirectional edges, DockQ target).
- ✅ **Training loop** (`train.train_model`): Adam, MSE vs DockQ, LR 0.005, 200 epochs,
  best-validation selection, seeded.
- ✅ **Metrics** (`metrics`): ranking loss (top-1), Pearson, Spearman, top-N CAPRI hit rate.
- ✅ **Red-to-green tests** (18): overfit-a-tiny-batch, forward shape/grad, single-node /
  no-edge graphs, metric unit tests. Hermetic (synthetic graphs; no `mkdssp` needed).

**Not yet done (later phases, see foundry #5):** pipeline proof on the CC-BY benchmark
(Phase B), the MAF2 training-decoy gate (Phase C), and the full train/eval/ship (Phase D).

## Underspecified points made explicit

The paper does not fully pin the attention block; these are our documented choices (config,
swappable):

- **Conv variant** — default `TransformerConv` (graph-transformer framing); `GATv2Conv`
  available via `conv="gatv2"`. Both condition attention on the 11-dim edge attributes.
- **Head aggregation** — `concat=False` so 8 heads average back to hidden 32 across both
  layers (rather than 8×32).
- **Edge reduction before concat** — the pooled edge branch reduces 11 → `edge_out` (default
  16) before concatenation with the node pool.

## Develop

```bash
uv venv --python 3.11 && uv pip install -e . pytest
.venv/bin/python -m pytest -q
```
