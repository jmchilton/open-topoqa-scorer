"""Convert featurizer output into PyG ``Data`` graphs for the scorer.

The featurizer (``open_topoqa_featurizer.featurize_complex``) returns an undirected
interface graph with a single ``(i, j)`` row per cross-chain contact (``i < j``).
PyG message passing is directed, so each contact is materialized as both ``(i, j)``
and ``(j, i)`` with the same 11-dim edge attributes — a symmetric bidirectional graph.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from open_topoqa_featurizer import EDGE_WIDTH, NODE_WIDTH

__all__ = ["graph_from_featurized", "graph_from_complex"]


def graph_from_featurized(feat: dict, y: float | None = None) -> Data:
    """Build a PyG ``Data`` from a featurizer output dict.

    ``feat`` has ``node_features`` (N, 172), ``edge_index`` (E, 2, i<j), and
    ``edge_features`` (E, 11). Edges are symmetrized to (2, 2E). ``y`` (the DockQ
    target, a scalar in [0, 1]) is stored as a (1,) tensor when provided.
    """
    x = torch.as_tensor(np.asarray(feat["node_features"]), dtype=torch.float32)
    if x.numel() == 0:
        x = x.reshape(0, NODE_WIDTH)

    ei = np.asarray(feat["edge_index"], dtype=np.int64).reshape(-1, 2)
    ea = np.asarray(feat["edge_features"], dtype=np.float32).reshape(-1, EDGE_WIDTH)
    if ei.shape[0]:
        # symmetrize: (i,j) and (j,i) share edge attributes
        src = np.concatenate([ei[:, 0], ei[:, 1]])
        dst = np.concatenate([ei[:, 1], ei[:, 0]])
        edge_index = torch.as_tensor(np.stack([src, dst]), dtype=torch.long)
        edge_attr = torch.as_tensor(np.concatenate([ea, ea]), dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_WIDTH), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = torch.tensor([float(y)], dtype=torch.float32)
    return data


def graph_from_complex(pdb_path: str, y: float | None = None, dssp_map: dict | None = None) -> Data:
    """Featurize a PDB complex and return its scorer graph. Convenience wrapper."""
    from open_topoqa_featurizer import featurize_complex

    return graph_from_featurized(featurize_complex(pdb_path, dssp_map=dssp_map), y=y)
