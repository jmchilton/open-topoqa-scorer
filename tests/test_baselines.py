"""Tests for the pLDDT ranking baselines (P6b)."""

from __future__ import annotations

import numpy as np

from open_topoqa_scorer.baselines import global_plddt, interface_plddt


def _atom(serial, chain, resseq, x, y, z, b, name="CA", resname="ALA", element="C") -> str:
    # Column-accurate PDB ATOM record (name 13-16, coords 31-54, occ 55-60, bfactor 61-66).
    return (
        f"ATOM  {serial:>5} {name:^4} {resname:>3} {chain}{resseq:>4}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{b:6.2f}          {element:>2}\n"
    )


def _write_two_chain_pdb(path) -> None:
    """A→B with one cross-chain interface contact and two far-apart residues.

    Interface (Cα within 10 Å of the other chain): A1 (b=90) ↔ B1 (b=80) only.
    A2 (b=10) and B2 (b=20) are >10 Å from every other chain, so they are non-interface.
    -> mean interface pLDDT = 85.0 ; mean global pLDDT = (90+10+80+20)/4 = 50.0.
    """
    lines = [
        _atom(1, "A", 1, 0.0, 0.0, 0.0, 90.0),
        _atom(2, "A", 2, 50.0, 0.0, 0.0, 10.0),
        _atom(3, "B", 1, 5.0, 0.0, 0.0, 80.0),
        _atom(4, "B", 2, 100.0, 0.0, 0.0, 20.0),
        "END\n",
    ]
    path.write_text("".join(lines))


def test_interface_and_global_plddt(tmp_path):
    pdb = tmp_path / "toy.pdb"
    _write_two_chain_pdb(pdb)
    assert interface_plddt(str(pdb)) == 85.0
    assert global_plddt(str(pdb)) == 50.0


def test_no_interface_returns_zero(tmp_path):
    # Two chains, both residues far apart -> no cross-chain contact -> 0.0 interface score.
    pdb = tmp_path / "noiface.pdb"
    pdb.write_text(
        _atom(1, "A", 1, 0.0, 0.0, 0.0, 90.0)
        + _atom(2, "B", 1, 80.0, 0.0, 0.0, 80.0)
        + "END\n"
    )
    assert interface_plddt(str(pdb)) == 0.0
    assert np.isclose(global_plddt(str(pdb)), 85.0)
