"""Two tower retrieval model in PyTorch.

Item tower   MLP over the item feature vector -> 64-d unit vector.
User tower   mean of the item embeddings in the user's history -> unit vector.
             It has no parameters and no user ID table on purpose: a new user
             with one watched title gets a vector at once, and the model can be
             served from a single item embedding table with numpy.
Loss         full softmax over the whole recommendable catalogue with a
             temperature. For a training pair (user, item) the user vector is
             computed from the user's other training items only (leave one out),
             so the positive never leaks into its own query.

Ablations zero out feature groups before the item tower (see `feature_mask`).
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


class ItemTower(nn.Module):
    def __init__(self, n_features: int, cfg: TrainConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), dim=-1)


def _sparse_to_torch(m: sparse.csr_matrix) -> torch.Tensor:
    coo = m.tocoo()
    idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(coo.data, dtype=torch.float32), coo.shape, check_invariants=True).coalesce()


def train_two_tower(
    item_features: np.ndarray,
    history: sparse.csr_matrix,
    cfg: TrainConfig = TrainConfig(),
    feature_mask: np.ndarray | None = None,
    log: bool = True,
) -> tuple[ItemTower, np.ndarray]:
    """Train on (user, item) pairs from `history`. Returns the tower and the item embedding table."""
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

    tower = ItemTower(feats.shape[1], cfg)
    opt = torch.optim.Adam(tower.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.epochs):
        tower.train()
        perm = pairs[torch.randperm(len(pairs))]
        total, steps = 0.0, 0
        for start in range(0, len(perm), cfg.batch_size):
            batch = perm[start : start + cfg.batch_size]
            users, items = batch[:, 0], batch[:, 1]

            item_emb = tower(x_items)                          # (n_items, d)
            user_sum = torch.sparse.mm(hist, item_emb)         # (n_users, d)
            # leave one out: remove the positive from its own user's history
            query = user_sum[users] - item_emb[items]
            denom = (n_hist[users] - 1).clamp(min=1).unsqueeze(1)
            query = nn.functional.normalize(query / denom, dim=-1)

            logits = query @ item_emb.T / cfg.temperature     # (batch, n_items)
            loss = nn.functional.cross_entropy(logits, items)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        if log:
            print(f"  epoch {epoch + 1:2d}/{cfg.epochs} loss {total / steps:.4f}")

    tower.eval()
    with torch.no_grad():
        table = tower(x_items).numpy().astype(np.float32)
    return tower, table
