"""Stratified sampler: proportional quotas, determinism, subset & no-leak invariants (hermetic)."""

from collections import Counter

from open_topoqa_scorer.benchmark import DecoyLabel
from open_topoqa_scorer.sample import sample_by_target, stratified_sample


def _labels(counts):
    """counts: {capri_class: n} -> DecoyLabels with distinct targets/models per class."""
    dq_for = [0.05, 0.35, 0.6, 0.9]
    return [
        DecoyLabel(target=f"T{cls}_{i}", model=f"m{i}", dockq=dq_for[cls], capri=cls, pdb_path="")
        for cls, n in counts.items()
        for i in range(n)
    ]


def test_sample_size_and_is_subset():
    labels = _labels({0: 800, 1: 100, 2: 60, 3: 40})  # 1000, MAF2-like imbalance
    s = stratified_sample(labels, 100, seed=0)
    assert len(s) == 100
    ids = {id(lab) for lab in labels}
    assert all(id(lab) in ids for lab in s)  # only real inputs, nothing fabricated
    assert len(s) == len({(lab.target, lab.model) for lab in s})  # no duplicates


def test_every_populated_stratum_survives_proportionally():
    labels = _labels({0: 800, 1: 100, 2: 60, 3: 40})
    by = Counter(lab.capri for lab in stratified_sample(labels, 100, seed=0))
    # the whole point: an Incorrect-dominated corpus must NOT collapse to all-Incorrect
    assert set(by) == {0, 1, 2, 3}
    assert 75 <= by[0] <= 85  # ~proportional to its 80% share
    assert by[3] >= 3  # rare near-native class keeps its ~4% share


def test_deterministic_by_seed():
    labels = _labels({0: 200, 1: 50, 2: 30, 3: 20})
    ident = lambda s: [(lab.target, lab.model) for lab in s]
    assert ident(stratified_sample(labels, 50, seed=7)) == ident(stratified_sample(labels, 50, seed=7))
    assert ident(stratified_sample(labels, 50, seed=7)) != ident(stratified_sample(labels, 50, seed=8))


def test_n_at_least_population_returns_all():
    labels = _labels({0: 5, 1: 3})
    s = stratified_sample(labels, 100, seed=0)
    assert {(lab.target, lab.model) for lab in s} == {(lab.target, lab.model) for lab in labels}


def test_sampling_a_train_side_cannot_surface_other_targets():
    # sampler is a pure subset op; sampling after a split can never introduce a held-out target
    train = _labels({0: 100, 1: 20})
    train_targets = {lab.target for lab in train}
    s = stratified_sample(train, 30, seed=1)
    assert {lab.target for lab in s} <= train_targets


def _multi_target(spec):
    """spec: {target: [capri classes]} -> DecoyLabels."""
    return [
        DecoyLabel(target=t, model=f"{t}-{i}", dockq=0.5, capri=c, pdb_path="")
        for t, classes in spec.items()
        for i, c in enumerate(classes)
    ]


def test_sample_by_target_keeps_every_target_and_caps():
    labels = _multi_target({
        "SMALL": [0, 3],                       # 2 decoys, under cap -> kept whole
        "BIG": [0, 0, 0, 0, 1, 2, 3, 3],       # 8 decoys -> capped to 4
    })
    s = sample_by_target(labels, decoys_per_target=4, seed=0)
    from collections import Counter as C
    per = C(lab.target for lab in s)
    assert per["SMALL"] == 2  # kept whole
    assert per["BIG"] == 4    # capped
    assert set(per) == {"SMALL", "BIG"}  # no target dropped


def test_sample_by_target_preserves_quality_gradient_and_is_deterministic():
    # a target with many Incorrect + a couple near-native must retain the near-native under the cap
    labels = _multi_target({"T": [0] * 30 + [3, 3]})  # 32 decoys, only 2 High
    a = sample_by_target(labels, decoys_per_target=10, seed=3)
    b = sample_by_target(labels, decoys_per_target=10, seed=3)
    assert [(l.target, l.model) for l in a] == [(l.target, l.model) for l in b]  # deterministic
    assert 3 in {l.capri for l in a}  # stratification kept a High decoy, not all-Incorrect
    assert len(a) == 10
