"""Stratified subsampling of a decoy corpus (Phase C3).

The combined corpus is class-imbalanced and pulled two ways: Dockground's Gramm-X decoys skew
low (~88% Incorrect) while MAF2's AlphaFold predictions skew high (median DockQ ~0.74), leaving
Acceptable the rare middle band (~6% of the whole). A uniform random subsample would drift toward
whichever class dominates and can starve the rarest quality band. We sample proportionally *within
CAPRI-class strata* via the largest-remainder method, so the subsample mirrors the corpus's quality
distribution and every well-populated class keeps its share.

The draw is a deterministic function of ``seed`` and returns a subset of the input (never
fabricates or duplicates rows), so sampling one side of a split can never surface a target
from the other side. A stratum so rare it rounds below one seat can still drop out entirely —
raise ``n`` if the smallest classes must be guaranteed representation.
"""

from __future__ import annotations

import random
from collections import defaultdict

__all__ = ["stratified_sample", "sample_by_target"]


def _quotas(sizes: dict, n: int) -> dict:
    """Largest-remainder apportionment of ``n`` seats across strata of the given ``sizes``.

    Each stratum gets ``floor(n*size/total)`` seats, then leftover seats go to the largest
    fractional remainders; every quota is clamped to its stratum size.
    """
    total = sum(sizes.values())
    raw = {k: n * sz / total for k, sz in sizes.items()}
    quota = {k: min(int(r), sizes[k]) for k, r in raw.items()}
    leftover = n - sum(quota.values())
    while leftover > 0:
        spare = [k for k in sizes if quota[k] < sizes[k]]
        if not spare:
            break
        spare.sort(key=lambda k: (raw[k] - int(raw[k]), str(k)), reverse=True)
        for k in spare[:leftover]:
            quota[k] += 1
        leftover = n - sum(quota.values())
    return quota


def stratified_sample(labels, n: int, seed: int = 0, key=lambda lab: lab.capri) -> list:
    """Return ``min(n, len(labels))`` labels, stratified by ``key`` (default CAPRI class).

    Deterministic given ``seed``; the result is a subset of ``labels`` (no fabrication,
    no duplication).
    """
    labels = list(labels)
    if n >= len(labels):
        return labels

    strata: dict = defaultdict(list)
    for lab in labels:
        strata[key(lab)].append(lab)

    quota = _quotas({k: len(v) for k, v in strata.items()}, n)

    rng = random.Random(seed)
    out: list = []
    for k in sorted(strata, key=str):  # stable stratum order for reproducibility
        group = sorted(strata[k], key=lambda lab: (lab.target, lab.model))
        out.extend(rng.sample(group, quota[k]))
    return out


def sample_by_target(labels, decoys_per_target: int, seed: int = 0, key=lambda lab: lab.capri) -> list:
    """Keep every target, capping each at ``decoys_per_target`` decoys.

    Unlike :func:`stratified_sample` (which samples individual decoys and shreds within-target decoy
    sets), this preserves whole targets — the structure within-target *ranking* needs. A target with
    at most ``decoys_per_target`` decoys is kept entirely; a larger one is thinned via
    :func:`stratified_sample` on its own decoys (stratified by ``key``, default CAPRI class) so the
    kept subset still spans the target's quality gradient, near-native included. Deterministic.
    """
    by_target: dict = defaultdict(list)
    for lab in labels:
        by_target[lab.target].append(lab)

    out: list = []
    for i, target in enumerate(sorted(by_target)):
        group = by_target[target]
        if len(group) <= decoys_per_target:
            out.extend(group)
        else:
            out.extend(stratified_sample(group, decoys_per_target, seed=seed * 100003 + i, key=key))
    return out
