"""Two tower retrieval model in PyTorch. This is the only module that imports torch.

Item tower  MLP over the item feature row, output is a 64-d unit vector.
User tower  mean of the item vectors in the viewer's history. It has no
            parameters and no user id table on purpose: a new viewer with one
            watched title gets a vector at once, and serving needs only the
            item table and numpy (see `evaluation.two_tower_scorer`).
Loss        full softmax over the recommendable catalogue with a temperature.
            For a training pair (viewer, item) the viewer vector is built from
            the viewer's other training items only (leave one out), so the
            positive never leaks into its own query.

Ablations multiply the feature row by a 0/1 mask (see `ItemFeatures.mask_without`).
The defaults below are the pre-registered settings. `tuning.py` searches around
them on a validation split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import sparse
from torch import nn


@dataclass
class TrainConfig:
    embed_dim: int = 64
    hidden: int = 256
    dropout: float = 0.2
    temperature: float = 0.07
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 10
    batch_size: int = 2048
    seed: int = 42
    activation: str = "relu"


ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


class ItemTower(nn.Module):
    def __init__(self, n_features: int, cfg: TrainConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, cfg.hidden),
            ACTIVATIONS[cfg.activation](),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), dim=-1)


def _sparse_to_torch(m: sparse.csr_matrix) -> torch.Tensor:
    coo = m.tocoo()
    idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, values, coo.shape, check_invariants=True).coalesce()


def train_two_tower(
    item_features: np.ndarray,
    history: sparse.csr_matrix,
    cfg: TrainConfig = TrainConfig(),
    feature_mask: np.ndarray | None = None,
    log: bool = True,
) -> tuple[ItemTower, np.ndarray, list[float]]:
    """Train on (viewer, item) pairs in `history`. Returns the tower, the item table and the loss per epoch."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    feats = item_features.astype(np.float32)
    if feature_mask is not None:
        feats = feats * feature_mask[None, :].astype(np.float32)
    x_items = torch.tensor(feats)
    hist = _sparse_to_torch(history)
    n_hist = torch.tensor(np.asarray(history.sum(axis=1)).ravel(), dtype=torch.float32)

    coo = history.tocoo()
    pairs = torch.tensor(np.vstack([coo.row, coo.col]).T, dtype=torch.long)
    generator = torch.Generator().manual_seed(cfg.seed)

    tower = ItemTower(feats.shape[1], cfg)
    opt = torch.optim.Adam(tower.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    losses: list[float] = []

    for epoch in range(cfg.epochs):
        tower.train()
        perm = pairs[torch.randperm(len(pairs), generator=generator)]
        total, steps = 0.0, 0
        for start in range(0, len(perm), cfg.batch_size):
            batch = perm[start : start + cfg.batch_size]
            users, items = batch[:, 0], batch[:, 1]

            item_emb = tower(x_items)  # (n_items, d)
            user_sum = torch.sparse.mm(hist, item_emb)  # (n_users, d)
            # Leave one out: remove the positive from its own viewer's history.
            query = user_sum[users] - item_emb[items]
            denom = (n_hist[users] - 1).clamp(min=1).unsqueeze(1)
            query = nn.functional.normalize(query / denom, dim=-1)

            logits = query @ item_emb.T / cfg.temperature  # (batch, n_items)
            loss = nn.functional.cross_entropy(logits, items)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        losses.append(total / steps)
        if log:
            print(f"    epoch {epoch + 1:2d}/{cfg.epochs} loss {losses[-1]:.4f}")

    tower.eval()
    with torch.no_grad():
        table = tower(x_items).numpy().astype(np.float32)
    return tower, table, losses


def embed_items(tower: ItemTower, item_features: np.ndarray, feature_mask: np.ndarray | None = None) -> np.ndarray:
    """Item vectors for any feature rows, including titles the tower never trained on (cold start)."""
    feats = item_features.astype(np.float32)
    if feature_mask is not None:
        feats = feats * feature_mask[None, :].astype(np.float32)
    tower.eval()
    with torch.no_grad():
        return tower(torch.tensor(feats)).numpy().astype(np.float32)
