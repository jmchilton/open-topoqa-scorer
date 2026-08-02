"""ProteinGAT — the TopoQA interface-quality regressor, reproduced from the paper spec.

Architecture (TopoQA supplement, hyperparameter table): a graph-attention encoder whose
attention depends on the source node, the destination node, and the **edge embedding**,
followed by separate mean-pooling of the node and edge representations, concatenation, and
an MLP head with a sigmoid output in [0, 1] (the predicted DockQ score).

Paper-underspecified points, made explicit config here (documented in the foundry #5 plan):

* **Conv variant** (plan Q2): the paper says only "graph attention with edge features".
  Both PyG ``TransformerConv`` and ``GATv2Conv`` accept an ``edge_dim`` and condition
  attention on edge attributes; we default to ``TransformerConv`` (a graph-transformer
  block, matching the "ProteinGAT"/transformer framing) and keep it swappable.
* **Head aggregation**: 8 heads with a fixed hidden width of 32 → ``concat=False`` so each
  layer averages its heads back to ``hidden`` (rather than 8×32=256), keeping the width the
  paper states across both layers.
* **Edge reduction before concat** (plan Q3): the pooled edge branch reduces the raw 11-dim
  edge attributes to ``edge_out`` (default 16) before concatenation with the node pool.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, TransformerConv, global_mean_pool

from open_topoqa_featurizer import EDGE_WIDTH, NODE_WIDTH

__all__ = ["ProteinGAT"]

_CONVS = {"transformer": TransformerConv, "gatv2": GATv2Conv}


class ProteinGAT(nn.Module):
    """Interface-graph quality regressor → scalar DockQ prediction in [0, 1]."""

    def __init__(
        self,
        node_dim: int = NODE_WIDTH,
        edge_dim: int = EDGE_WIDTH,
        hidden: int = 32,
        heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.25,
        edge_out: int = 16,
        conv: str = "transformer",
    ):
        super().__init__()
        if conv not in _CONVS:
            raise ValueError(f"conv must be one of {sorted(_CONVS)}, got {conv!r}")
        self.dropout = dropout
        self.node_encoder = nn.Linear(node_dim, hidden)
        conv_cls = _CONVS[conv]
        self.convs = nn.ModuleList(
            conv_cls(hidden, hidden, heads=heads, concat=False, edge_dim=edge_dim, dropout=dropout)
            for _ in range(num_layers)
        )
        self.edge_reducer = nn.Linear(edge_dim, edge_out)
        self.head = nn.Sequential(
            nn.Linear(hidden + edge_out, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, data) -> torch.Tensor:
        """Return one predicted score per graph, shape ``(num_graphs,)`` in [0, 1]."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        x = F.relu(self.node_encoder(x))
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_pool = global_mean_pool(x, batch)

        # edge branch: reduce raw edge attrs, pool per graph via the source node's graph id
        if edge_index.size(1):
            edge_batch = batch[edge_index[0]]
            edge_repr = F.relu(self.edge_reducer(edge_attr))
            edge_pool = global_mean_pool(edge_repr, edge_batch, size=node_pool.size(0))
        else:
            edge_pool = node_pool.new_zeros(node_pool.size(0), self.edge_reducer.out_features)

        graph_repr = torch.cat([node_pool, edge_pool], dim=-1)
        out = self.head(graph_repr).squeeze(-1)
        return torch.sigmoid(out)
