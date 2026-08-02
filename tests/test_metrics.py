import numpy as np
from pytest import approx

from open_topoqa_scorer.metrics import (
    _average_ranks,
    pearson,
    ranking_loss,
    spearman,
    top_n_hit_rate,
    top_n_success,
)


def test_pearson_perfect_and_anti():
    x = np.array([0.1, 0.2, 0.3, 0.4])
    assert pearson(x, 2 * x + 1) == approx(1.0)
    assert pearson(x, -x) == approx(-1.0)


def test_pearson_degenerate_is_zero():
    assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0
    assert pearson([1.0], [1.0]) == 0.0


def test_average_ranks_handles_ties():
    # values 3,1,1,4 -> the two 1s share ranks (1,2)->1.5; 3->3; 4->4
    r = _average_ranks([3.0, 1.0, 1.0, 4.0])
    assert list(r) == [3.0, 1.5, 1.5, 4.0]


def test_spearman_monotonic_nonlinear_is_one():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # strictly increasing but nonlinear -> Spearman 1.0 though Pearson < 1
    y = x**3
    assert spearman(x, y) == approx(1.0)
    assert pearson(x, y) < 1.0


def test_ranking_loss_zero_when_best_picked():
    true = np.array([0.1, 0.9, 0.5])
    pred = np.array([0.2, 0.8, 0.4])  # argmax(pred)=1 == argmax(true)=1
    assert ranking_loss(pred, true) == 0.0


def test_ranking_loss_penalizes_wrong_pick():
    true = np.array([0.1, 0.9, 0.5])
    pred = np.array([0.9, 0.1, 0.2])  # picks decoy 0 (true 0.1); best is 0.9
    assert ranking_loss(pred, true) == 0.8


def test_top_n_hit_rate_distinct_answers():
    # pred ranks decoys 0>1>2>3>4; classes at those ranks are 0,2,0,1,0
    capri = np.array([0, 2, 0, 1, 0])
    pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    assert top_n_hit_rate(pred, capri, n=3, threshold=1) == approx(1 / 3)  # 0,2,0 -> one hit
    assert top_n_hit_rate(pred, capri, n=2, threshold=3) == 0.0  # no high-quality in top-2
    # n beyond the population: denominator clamps to 5 -> two hits (classes 2 and 1)
    assert top_n_hit_rate(pred, capri, n=99, threshold=1) == approx(2 / 5)


def test_top_n_success_is_target_level_indicator():
    capri = np.array([0, 2, 0, 1, 0])
    pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    assert top_n_success(pred, capri, n=2, threshold=1) == 1.0  # decoy 1 (class 2) is in top-2
    assert top_n_success(pred, capri, n=1, threshold=1) == 0.0  # only decoy 0 (class 0)
    assert top_n_success(pred, capri, n=5, threshold=3) == 0.0  # no high-quality anywhere


def test_top_n_deterministic_on_ties():
    # equal predictions -> stable argsort picks the lower index for top-1
    assert top_n_hit_rate([0.5, 0.5], np.array([1, 0]), n=1, threshold=1) == 1.0  # decoy 0, class 1
    assert top_n_hit_rate([0.5, 0.5], np.array([0, 1]), n=1, threshold=1) == 0.0  # decoy 0, class 0
