"""Within-target ranking loss: penalizes mis-ordering, ignores cross-target & tied pairs (hermetic)."""

import torch

from open_topoqa_scorer.losses import within_target_ranking_loss


def test_correct_order_beyond_margin_is_zero():
    # one target, predictions ordered like truth with a comfortable gap -> no penalty
    pred = torch.tensor([0.9, 0.5, 0.1])
    true = torch.tensor([0.8, 0.4, 0.0])
    tgt = torch.tensor([0, 0, 0])
    assert within_target_ranking_loss(pred, true, tgt, margin=0.1).item() == 0.0


def test_inversion_is_penalized():
    # the better decoy (true 0.8) is predicted LOWER -> positive loss
    pred = torch.tensor([0.2, 0.7])
    true = torch.tensor([0.8, 0.1])
    tgt = torch.tensor([0, 0])
    assert within_target_ranking_loss(pred, true, tgt, margin=0.1).item() > 0.0


def test_only_within_target_pairs_count():
    # a cross-target "inversion" must NOT contribute: two singleton targets -> zero pairs -> 0 loss
    pred = torch.tensor([0.2, 0.9])
    true = torch.tensor([0.8, 0.1])  # target 0 better than target 1, but pred inverts across targets
    tgt = torch.tensor([0, 1])
    assert within_target_ranking_loss(pred, true, tgt, margin=0.1).item() == 0.0


def test_tied_truth_within_deadband_ignored():
    # true DockQ nearly equal (within deadband) -> no ordering to enforce -> 0 even if preds differ
    pred = torch.tensor([0.1, 0.9])
    true = torch.tensor([0.50, 0.51])
    tgt = torch.tensor([0, 0])
    assert within_target_ranking_loss(pred, true, tgt, margin=0.1, deadband=0.02).item() == 0.0


def test_differentiable_and_drives_scores_apart():
    # gradient should push the under-predicted better decoy up / the over-predicted worse one down
    pred = torch.tensor([0.2, 0.7], requires_grad=True)
    true = torch.tensor([0.8, 0.1])
    tgt = torch.tensor([0, 0])
    loss = within_target_ranking_loss(pred, true, tgt, margin=0.1)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad[0] < 0  # decreasing loss means increasing pred[0] (the better decoy)
    assert pred.grad[1] > 0  # ...and decreasing pred[1] (the worse decoy)
