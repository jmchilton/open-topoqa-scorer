"""Loader for the Dockground 1.0 docking-decoy set (Phase C, the second training corpus).

Dockground (Liu/Vakser Lab, KU) is the *deep* half of the DProQA/TopoQA training data — ~58
targets, ~100 Gramm-X decoys each — complementing MAF2's ~2-models-per-target breadth. Each
target ships as ``<pdbid>_<rec>_<lig>.tgz`` containing::

    <pdbid>_<rec>:<lig>/rmsd.list           # Rank R_rmsd L_rmsd I_rmsd fnat fnon-nat
    <pdbid>_<rec>:<lig>/decoys/r-l_<Rank>.pdb

Dockground ships the CAPRI components but no DockQ, so we compute DockQ from them directly via the
standard formula (Basu & Wallner, PLOS ONE 2016) — this is exactly the quantity DockQ is defined
on, so no native reconstruction or re-scoring is needed. Column meanings are from the bundled
``help_decoy.html`` (L_rmsd = ligand backbone RMSD after receptor superposition; I_rmsd = interface
backbone RMSD; fnat = fraction of native contacts). Data is public but git-ignored (pulled locally).
"""

from __future__ import annotations

import os
import tarfile

from open_topoqa_scorer.benchmark import DecoyLabel, featurize_subset

__all__ = ["DecoyLabel", "featurize_subset", "dockq_from_capri", "dockq_to_capri_class",
           "load_dockground_labels"]


def dockq_from_capri(fnat: float, irmsd: float, lrmsd: float) -> float:
    """DockQ ∈ [0, 1] from its three CAPRI components (Basu & Wallner 2016).

    ``DockQ = (Fnat + 1/(1+(iRMSD/1.5)^2) + 1/(1+(LRMSD/8.5)^2)) / 3``.
    """
    scaled_irmsd = 1.0 / (1.0 + (irmsd / 1.5) ** 2)
    scaled_lrmsd = 1.0 / (1.0 + (lrmsd / 8.5) ** 2)
    return (fnat + scaled_irmsd + scaled_lrmsd) / 3.0


def dockq_to_capri_class(dockq: float) -> int:
    """DockQ → CAPRI quality class 0/1/2/3 (Incorrect / Acceptable / Medium / High)."""
    if dockq < 0.23:
        return 0
    if dockq < 0.49:
        return 1
    if dockq < 0.80:
        return 2
    return 3


def _parse_rmsd_list(path: str) -> list[tuple[int, float, float, float]]:
    """Yield ``(rank, lrmsd, irmsd, fnat)`` per line of a Dockground ``rmsd.list``."""
    rows = []
    with open(path) as fh:
        for line in fh:
            f = line.split()
            if len(f) != 6:
                continue
            rank, _r_rmsd, lrmsd, irmsd, fnat, _fnon = f
            rows.append((int(rank), float(lrmsd), float(irmsd), float(fnat)))
    return rows


def _extract_target(tgz_path: str, extract_dir: str) -> str | None:
    """Untar a target archive into ``extract_dir`` (cached); return its inner dir, or None."""
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tgz_path) as tf:
        top = tf.getnames()[0].split("/")[0]  # e.g. "1cho_E:I"
        inner = os.path.join(extract_dir, top)
        if not os.path.isdir(inner):
            tf.extractall(extract_dir)  # noqa: S202 — trusted local Dockground archive
    return inner if os.path.isdir(inner) else None


def load_dockground_labels(
    tgz_dir: str, extract_dir: str, require_files: bool = True
) -> list[DecoyLabel]:
    """Load Dockground decoys as :class:`DecoyLabel` rows with DockQ computed from ``rmsd.list``.

    ``tgz_dir`` holds the per-target ``*.tgz``; each is untarred into ``extract_dir`` once (cached).
    ``target`` is the 4-char PDB id (so multiple interaction sets of one entry cluster together);
    ``model`` is unique per decoy (``<archive-stem>_r-l_<rank>``). Reuses :class:`DecoyLabel` and
    :func:`featurize_subset`.
    """
    out: list[DecoyLabel] = []
    for name in sorted(os.listdir(tgz_dir)):
        if not name.endswith(".tgz"):
            continue
        stem = name[:-4]  # e.g. "1cho_E_I"
        pdbid = stem.split("_")[0]  # e.g. "1cho"
        inner = _extract_target(os.path.join(tgz_dir, name), extract_dir)
        if inner is None:
            continue
        rmsd_list = os.path.join(inner, "rmsd.list")
        if not os.path.exists(rmsd_list):
            continue
        for rank, lrmsd, irmsd, fnat in _parse_rmsd_list(rmsd_list):
            pdb_path = os.path.join(inner, "decoys", f"r-l_{rank}.pdb")
            if not os.path.exists(pdb_path):
                if require_files:
                    continue
                pdb_path = ""
            dockq = dockq_from_capri(fnat, irmsd, lrmsd)
            out.append(
                DecoyLabel(
                    target=pdbid,
                    model=f"{stem}_r-l_{rank}",
                    dockq=dockq,
                    capri=dockq_to_capri_class(dockq),
                    pdb_path=pdb_path,
                )
            )
    return out
