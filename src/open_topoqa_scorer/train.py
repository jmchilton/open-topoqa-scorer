"""Training loop for the ProteinGAT scorer.

Paper spec is only the loss: **MSE** against the DockQ target (§5.2.5). The optimizer (Adam),
learning rate (0.005), epoch budget (200), and best-validation checkpoint selection are OUR
tunable choices — the paper states no value for them. Deterministic given a seed.

Optionally augments MSE with a **within-target pairwise ranking loss** (``ranking_weight > 0``):
MSE alone lets the model collapse a docking-decoy set to its mean DockQ (good absolute fit, random
within-target ranking); the ranking term (see :mod:`losses`) penalizes mis-ordering decoys of the
same target. When enabled, batches are grouped so each holds whole targets (so same-target pairs
exist), and both training and validation use the combined objective. This is our extension, not
paper-specified.
"""

from __future__ import annotations

import copy
import random
from collections import defaultdict

import torch
from torch_geometric.loader import DataLoader

from open_topoqa_scorer.data import feature_stats
from open_topoqa_scorer.losses import within_target_ranking_loss
from open_topoqa_scorer.model import ProteinGAT

__all__ = ["train_model", "set_seed", "predict"]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model: ProteinGAT, dataset, batch_size: int = 32) -> torch.Tensor:
    """Predicted scores for every graph in ``dataset``, in order."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return torch.cat([model(batch) for batch in loader]) if len(dataset) else torch.empty(0)


def _target_index_map(targets) -> list[int]:
    """Map a list of target ids to contiguous ints (stable in first-seen order)."""
    order: dict = {}
    return [order.setdefault(t, len(order)) for t in targets]


def _target_batches(target_idx: list[int], batch_size: int, rng: random.Random) -> list[list[int]]:
    """Pack dataset indices into batches of *whole* targets (~``batch_size`` graphs each).

    Keeping every decoy of a target in one batch is what lets the ranking loss form same-target
    pairs. Target order is shuffled per call; a target larger than ``batch_size`` becomes its own
    (oversized) batch rather than being split.
    """
    by_target: dict = defaultdict(list)
    for i, t in enumerate(target_idx):
        by_target[t].append(i)
    targets = list(by_target)
    rng.shuffle(targets)
    batches: list[list[int]] = []
    cur: list[int] = []
    for t in targets:
        idx = by_target[t]
        if cur and len(cur) + len(idx) > batch_size:
            batches.append(cur)
            cur = []
        cur.extend(idx)
    if cur:
        batches.append(cur)
    return batches


def _attach_target_idx(dataset, targets) -> list[int]:
    """Attach a per-graph ``target_idx`` (so PyG collates it) and return the mapping."""
    idx = _target_index_map(targets)
    for data, t in zip(dataset, idx):
        data.target_idx = torch.tensor(t, dtype=torch.long)
    return idx


def _batch_loss(model, batch, mse_fn, ranking_weight, margin, deadband):
    pred = model(batch)
    y = batch.y.view(-1)
    loss = mse_fn(pred, y)
    if ranking_weight > 0:
        loss = loss + ranking_weight * within_target_ranking_loss(
            pred, y, batch.target_idx, margin=margin, deadband=deadband
        )
    return loss


def train_model(
    train_set,
    val_set=None,
    *,
    epochs: int = 200,
    lr: float = 0.005,
    batch_size: int = 32,
    seed: int = 0,
    model_kwargs: dict | None = None,
    verbose: bool = False,
    normalize: bool = True,
    ranking_weight: float = 0.0,
    ranking_margin: float = 0.1,
    ranking_deadband: float = 0.02,
    train_targets=None,
    val_targets=None,
):
    """Train and return ``(model, history)``.

    ``model`` carries the best-validation weights (or the final weights if no ``val_set``).
    ``history`` is a dict of per-epoch ``train_loss`` and, when validating, ``val_loss``.

    With ``ranking_weight > 0`` the objective is ``MSE + ranking_weight · ranking_loss`` and
    ``train_targets`` (a target id per graph, aligned to ``train_set``) is required; pass
    ``val_targets`` too to validate on the same combined objective. With ``ranking_weight == 0``
    it is plain shuffled-MSE (targets ignored) — the original behavior.
    """
    set_seed(seed)
    model = ProteinGAT(**(model_kwargs or {}))
    if normalize and len(train_set):
        model.set_feature_stats(*feature_stats(train_set))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_fn = torch.nn.MSELoss()

    ranking = ranking_weight > 0
    if ranking:
        if train_targets is None:
            raise ValueError("ranking_weight > 0 requires train_targets (one target id per graph)")
        train_tidx = _attach_target_idx(train_set, train_targets)
        val_tidx = _attach_target_idx(val_set, val_targets) if (val_set and val_targets) else None
        batch_rng = random.Random(seed)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    validating = val_set is not None and len(val_set)
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        if ranking:
            batches = _target_batches(train_tidx, batch_size, batch_rng)
            loader = DataLoader(train_set, batch_sampler=batches)
        else:
            loader = train_loader
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            loss = _batch_loss(model, batch, mse_fn, ranking_weight, ranking_margin, ranking_deadband)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * batch.num_graphs
        train_loss = total / max(1, len(train_set))
        history["train_loss"].append(train_loss)

        if validating:
            model.eval()
            with torch.no_grad():
                if ranking and val_tidx is not None:
                    vbatches = _target_batches(val_tidx, batch_size, random.Random(0))
                    vloader = DataLoader(val_set, batch_sampler=vbatches)
                else:
                    vloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
                vtotal = sum(
                    float(_batch_loss(model, b, mse_fn, ranking_weight, ranking_margin, ranking_deadband))
                    * b.num_graphs
                    for b in vloader
                )
                val_loss = vtotal / len(val_set)
            history["val_loss"].append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            if verbose:
                print(f"epoch {epoch}: train {train_loss:.4f} val {val_loss:.4f}")
        elif verbose:
            print(f"epoch {epoch}: train {train_loss:.4f}")

    if validating:
        if best_epoch < 0:
            # every val loss was NaN/inf — loading the init would return an untrained model
            raise RuntimeError(
                "validation loss never improved on the initial weights (diverged to NaN?); "
                "no best-validation checkpoint to restore"
            )
        history["best_epoch"] = best_epoch
        model.load_state_dict(best_state)
    return model, history
