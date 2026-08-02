"""Synthetic graph fixtures — hermetic, no featurizer/mkdssp needed.

Builds featurizer-shaped dicts (node_features N×172, edge_index E×2 with i<j, edge_features
E×11) so the data/model layers can be exercised without a real complex.
"""

import numpy as np
import pytest

from open_topoqa_featurizer import EDGE_WIDTH, NODE_WIDTH


def _feat_dict(n_nodes, edges, seed=0):
    rng = np.random.default_rng(seed)
    node_features = rng.standard_normal((n_nodes, NODE_WIDTH)).astype(np.float32)
    edge_index = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    edge_features = rng.standard_normal((edge_index.shape[0], EDGE_WIDTH)).astype(np.float32)
    return {
        "node_ids": [("A", (" ", i, " ")) for i in range(n_nodes)],
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
    }


@pytest.fixture
def feat_dict():
    return _feat_dict


@pytest.fixture
def small_graph_feat():
    # 4 nodes, a small chain of cross-chain contacts (i<j)
    return _feat_dict(4, [(0, 1), (1, 2), (2, 3)], seed=1)
