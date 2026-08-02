"""Training loop for the ProteinGAT scorer.

Paper spec: Adam, MSE against the DockQ target, LR 0.005, up to 200 epochs, best-validation
checkpoint selection. Deterministic given a seed.
"""

from __future__ import annotations

import copy

import torch
from torch_geometric.loader import DataLoader

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


def _epoch(model, loader, optimizer, loss_fn):
    model.train()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * batch.num_graphs
    return total / max(1, len(loader.dataset))


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
):
    """Train and return ``(model, history)``.

    ``model`` carries the best-validation weights (or the final weights if no ``val_set``).
    ``history`` is a dict of per-epoch ``train_loss`` and, when validating, ``val_loss``.
    """
    set_seed(seed)
    model = ProteinGAT(**(model_kwargs or {}))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    validating = val_set is not None and len(val_set)
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        train_loss = _epoch(model, train_loader, optimizer, loss_fn)
        history["train_loss"].append(train_loss)

        if validating:
            model.eval()
            with torch.no_grad():
                val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
                val_loss = sum(
                    float(loss_fn(model(b), b.y.view(-1))) * b.num_graphs for b in val_loader
                ) / len(val_set)
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
