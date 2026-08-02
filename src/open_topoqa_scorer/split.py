"""Homology-aware train/val split via MMseqs2 clustering (Phase C2).

TopoQA/DProQA follow GNN-DOVE and split so that no train and val targets share >30% sequence
identity — otherwise validation leaks and model selection is optimistic. We reproduce that here:

1. extract one representative sequence per *target* (concatenated chains of a representative PDB),
2. cluster targets with ``mmseqs easy-cluster --min-seq-id 0.3``,
3. assign whole clusters to train or val — every decoy of a target follows its target's cluster,
   so a homology cluster never spans the split.

The final CC-BY benchmark (HAF2 / DBM55-AF2) stays the held-out *test* set; this split only
governs training + model selection. Requires ``mmseqs`` on PATH (``pixi global install
-c bioconda mmseqs2``).
"""

from __future__ import annotations

import os
import random
import subprocess
from collections import defaultdict

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

__all__ = ["target_sequence", "cluster_targets", "split_by_clusters"]

_PARSER = PDBParser(QUIET=True)


def target_sequence(pdb_path: str) -> str:
    """Concatenated one-letter sequence of all chains (chain-id order) of a PDB's first model.

    Non-standard residues collapse to ``X``; empty for a structure with no standard residues.
    """
    structure = _PARSER.get_structure("t", pdb_path)
    model = next(iter(structure))
    parts = []
    for chain in sorted(model, key=lambda c: c.id):
        resnames = [res.resname for res in chain if res.id[0] == " "]  # standard residues only
        if resnames:
            parts.append(seq1("".join(resnames), undef_code="X", custom_map={}))
    return "".join(parts)


def _representative_pdb(labels) -> dict[str, str]:
    """First on-disk PDB per target (sequence is shared across a target's decoys)."""
    rep: dict[str, str] = {}
    for lab in labels:
        if lab.target not in rep and lab.pdb_path and os.path.exists(lab.pdb_path):
            rep[lab.target] = lab.pdb_path
    return rep


def cluster_targets(
    labels,
    tmp_dir: str,
    min_seq_id: float = 0.3,
    coverage: float = 0.8,
    mmseqs: str = "mmseqs",
) -> dict[str, str]:
    """Cluster targets by sequence identity; return ``{target: cluster_id}``.

    ``cluster_id`` is the mmseqs cluster representative — targets sharing one are homologous
    above ``min_seq_id`` and must not be split across train/val.
    """
    reps = _representative_pdb(labels)
    if not reps:
        return {}
    os.makedirs(tmp_dir, exist_ok=True)
    fasta = os.path.join(tmp_dir, "targets.fasta")
    with open(fasta, "w") as fh:
        for target, pdb in sorted(reps.items()):
            seq = target_sequence(pdb)
            if seq:
                fh.write(f">{target}\n{seq}\n")

    out_prefix = os.path.join(tmp_dir, "clu")
    subprocess.run(
        [mmseqs, "easy-cluster", fasta, out_prefix, os.path.join(tmp_dir, "mmseqs_tmp"),
         "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", "1", "-v", "1"],
        check=True, capture_output=True, text=True,
    )
    clusters: dict[str, str] = {}
    with open(f"{out_prefix}_cluster.tsv") as fh:
        for line in fh:
            rep, member = line.rstrip("\n").split("\t")
            clusters[member] = rep
    return clusters


def split_by_clusters(labels, tmp_dir: str, val_frac: float = 0.2, seed: int = 0, **cluster_kw):
    """Split ``labels`` into ``(train, val)`` with no homology cluster spanning both.

    Whole clusters are assigned to val (in a seeded shuffle) until val reaches ~``val_frac`` of
    all *decoys*; decoys of an unclustered target (no sequence) go to train. Every decoy of a
    target follows its target's cluster, so target- and cluster-disjointness both hold.
    """
    clusters = cluster_targets(labels, tmp_dir, **cluster_kw)

    decoys_per_cluster: dict[str, int] = defaultdict(int)
    for lab in labels:
        cid = clusters.get(lab.target)
        if cid is not None:
            decoys_per_cluster[cid] += 1

    cluster_ids = sorted(decoys_per_cluster)
    random.Random(seed).shuffle(cluster_ids)
    target_val = val_frac * sum(decoys_per_cluster.values())
    val_clusters: set[str] = set()
    running = 0
    for cid in cluster_ids:
        if running >= target_val:
            break
        val_clusters.add(cid)
        running += decoys_per_cluster[cid]

    train, val = [], []
    for lab in labels:
        cid = clusters.get(lab.target)
        (val if cid in val_clusters else train).append(lab)
    return train, val
