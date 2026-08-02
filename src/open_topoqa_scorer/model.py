"""ProteinGAT — the TopoQA interface-quality regressor, reproduced from the paper.

Reproduced from the paper's Methods (Han et al. 2025, bbaf083, §5.2.5, Eqs 3–9) — NOT from
the upstream code, which is unlicensed and was never read. What the paper actually specifies:

* A custom ``ProteinGAT`` layer using **multi-head attention that updates BOTH node and edge
  embeddings**. The attention coefficient c_ij is computed from ``x_i``, ``x_j`` and the edge
  embedding ``e_ij`` (Eq. 3): an **additive** score over three weight matrices W_s, W_t, W_e,
  normalized by softmax over N(i) ∪ {i} (Eqs 4–5). PyG ``GATv2Conv`` implements exactly this
  additive, edge-conditioned form (hence the **default**, matching the paper's "GAT"). PyG
  ``TransformerConv`` (``conv="transformer"``) is offered as a knob but uses *dot-product*
  attention — a deliberate divergence from Eq. 3, not a faithful reproduction.
* **Edges are updated every layer** (Eq. 6): ``e_ij^(l+1) = Θ_e·[x_i^(l+1) ‖ x_j^(l+1) ‖
  e_ij^(l)]`` — a learned map of the two *updated* endpoint embeddings and the previous edge
  embedding. Neither stock PyG conv does this, so it is an explicit per-layer module here.
* **Readout** (Eqs 7–9): mean-pool nodes; mean-pool the *updated* edges; a linear reduces the
  pooled edge vector to **half the pooled-node dimension**; concatenate; feed through an **MLP
  of three stacked linear layers**; ``sigmoid`` → predicted DockQ in [0, 1]. Loss is MSE.

What the paper does NOT specify (grep-confirmed: no value stated) — number of layers, attention
heads, node/edge embedding widths, dropout, optimizer, learning rate, epochs, batch size. Those
are therefore **our** tunable choices (defaults below), not paper-mandated values. The structural
facts above (edge updating, edge-dim = half node-dim, 3-layer readout, sigmoid) ARE the paper's.
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
    """Interface-graph quality regressor → scalar DockQ prediction in [0, 1].

    Hyperparameters are our tunable defaults (the paper is silent on them); the *structure*
    — edge-updating attention, half-width pooled edges, a three-linear readout — is the paper's.
    """

    def __init__(
        self,
        node_dim: int = NODE_WIDTH,
        edge_dim: int = EDGE_WIDTH,
        hidden: int = 32,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.25,
        edge_hidden: int = 16,
        conv: str = "gatv2",
    ):
        super().__init__()
        if conv not in _CONVS:
            raise ValueError(f"conv must be one of {sorted(_CONVS)}, got {conv!r}")
        self.dropout = dropout
        self.edge_pool_dim = hidden // 2  # paper: pooled edge dim = half pooled node dim

        self.node_encoder = nn.Linear(node_dim, hidden)
        self.edge_encoder = nn.Linear(edge_dim, edge_hidden)

        conv_cls = _CONVS[conv]
        self.convs = nn.ModuleList(
            conv_cls(hidden, hidden, heads=heads, concat=False, edge_dim=edge_hidden, dropout=dropout)
            for _ in range(num_layers)
        )
        # Eq. 6: per-layer edge update from the two updated node embeddings + previous edge.
        # The paper writes a bare projection Θ_e·[·]; the ReLU + dropout wrapped around it in
        # forward() are OURS (regularization), not part of Eq. 6.
        self.edge_updates = nn.ModuleList(
            nn.Linear(2 * hidden + edge_hidden, edge_hidden) for _ in range(num_layers)
        )
        # Eq. 8: reduce pooled edge embedding to half the node width. Paper's W_pool is a bare
        # matrix; the bias here is ours (it is what an edgeless graph's edge feature collapses to).
        self.edge_pool_proj = nn.Linear(edge_hidden, self.edge_pool_dim)
        # Eq. 9: MLP with three stacked linear layers
        self.head = nn.Sequential(
            nn.Linear(hidden + self.edge_pool_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.edge_pool_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.edge_pool_dim, 1),
        )

    def forward(self, data) -> torch.Tensor:
        """Return one predicted score per graph, shape ``(num_graphs,)`` in [0, 1]."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        src, dst = edge_index  # (0,)-length slices when a graph/batch has no edges

        x = F.relu(self.node_encoder(x))
        e = F.relu(self.edge_encoder(edge_attr))  # (E, edge_hidden); (0, edge_hidden) if no edges

        for conv, edge_update in zip(self.convs, self.edge_updates):
            x = F.relu(conv(x, edge_index, edge_attr=e))
            x = F.dropout(x, p=self.dropout, training=self.training)
            # Eq. 6: update each edge from the two just-updated node embeddings + previous edge.
            # A no-op on an edgeless graph (empty slices), keeping single- and batched-forward
            # identical rather than branching on edge presence.
            e = F.relu(edge_update(torch.cat([x[src], x[dst], e], dim=-1)))
            e = F.dropout(e, p=self.dropout, training=self.training)

        # Explicit graph count so a graph with no interface nodes (an out-of-contact decoy, a
        # legitimate DockQ≈0 example) still gets a row — inferring size from batch.max()+1 would
        # silently drop trailing empty graphs and misalign predictions against the targets.
        size = getattr(data, "num_graphs", None)
        if size is None:
            size = int(batch.max()) + 1 if batch.numel() else 1
        node_pool = global_mean_pool(x, batch, size=size)  # Eq. 7 (empty graph -> zero row)
        # Eq. 8: pool updated edges (a graph with no edges pools to zeros), reduce to half node
        # dim. Projecting the pooled vector (not branching to raw zeros) makes an edgeless graph's
        # edge feature the projection bias consistently in solo and batched forwards.
        edge_pool = global_mean_pool(e, batch[src], size=size)
        edge_pool = self.edge_pool_proj(edge_pool)

        out = self.head(torch.cat([node_pool, edge_pool], dim=-1)).squeeze(-1)  # Eq. 9
        return torch.sigmoid(out)
