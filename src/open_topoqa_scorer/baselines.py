"""Non-topological ranking baselines for the P6b competitiveness leaderboard (issue #12).

The DProQ-benchmark decoys are AlphaFold-Multimer models that carry per-residue pLDDT in the
B-factor column. Ranking a target's decoys by their mean pLDDT is a standard, *training-free*
AlphaFold-confidence baseline — the practical thing a bench scientist reaches for before any
external QA model. Measuring it over the **interface** residues (the same Cα-within-10 Å residues
the topological scorer features) keeps it apples-to-apples with the scorer; ``global_plddt`` is the
whole-model variant for contrast.

These are ranking signals only (higher pLDDT = more confident), not DockQ predictions, so they feed
the ranking metrics (per-target ranking loss / Spearman / top-N success), not the regression ones.
"""

from __future__ import annotations

import numpy as np
from Bio.PDB.Structure import Structure

from open_topoqa_featurizer.graph import CA_CUTOFF, interface_nodes, parse_structure

__all__ = ["residue_plddt", "interface_plddt", "global_plddt"]


def residue_plddt(residue) -> float:
    """Per-residue pLDDT = its Cα B-factor (AF writes one pLDDT per residue across all atoms).

    Falls back to the mean atom B-factor for a residue with no Cα, and 0.0 for an empty residue.
    """
    if residue.has_id("CA"):
        return float(residue["CA"].get_bfactor())
    bs = [a.get_bfactor() for a in residue.get_atoms()]
    return float(np.mean(bs)) if bs else 0.0


def _model(pdb_or_model):
    """Accept a PDB path or an already-parsed Biopython Structure/Model."""
    if isinstance(pdb_or_model, str):
        return parse_structure(pdb_or_model)
    if isinstance(pdb_or_model, Structure):  # a full structure -> first model
        return pdb_or_model[0]
    return pdb_or_model  # assume a Model


def interface_plddt(pdb_or_model, cutoff: float = CA_CUTOFF) -> float:
    """Mean pLDDT over interface residues (Cα within ``cutoff`` of another chain).

    Returns 0.0 for a decoy with no cross-chain interface (its decoys sort last, which is the
    right behaviour for an out-of-contact pose).
    """
    model = _model(pdb_or_model)
    nodes = interface_nodes(model, cutoff=cutoff)
    if not nodes:
        return 0.0
    return float(np.mean([residue_plddt(res) for _cid, res in nodes]))


def global_plddt(pdb_or_model) -> float:
    """Mean pLDDT over every residue in the model (whole-structure confidence)."""
    model = _model(pdb_or_model)
    vals = [residue_plddt(res) for chain in model for res in chain]
    return float(np.mean(vals)) if vals else 0.0
