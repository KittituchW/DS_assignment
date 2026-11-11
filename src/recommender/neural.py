from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class ProjectionNetwork(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(inputs), p=2, dim=-1)


@dataclass
class TrainingResult:
    model: ProjectionNetwork
    loss_history: list[float]
    triplet_count: int


def _build_triplets(metadata: pd.DataFrame, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = metadata.reset_index(drop=True).copy()
    frame["group"] = (
        frame["type"].astype(str)
        + "|"
        + frame["primary_genre"].astype(str)
        + "|"
        + frame["language"].astype(str)
    )
    group_members = {
        key: values.index.to_numpy() for key, values in frame.groupby("group") if len(values) >= 2
    }
    type_members = {
        key: values.index.to_numpy() for key, values in frame.groupby("type")
    }

    triplets: list[tuple[int, int, int]] = []
    for anchor, row in frame.iterrows():
        positives = group_members.get(row["group"])
        if positives is None:
            continue
        positive_pool = positives[positives != anchor]
        if not len(positive_pool):
            continue
        positive = int(rng.choice(positive_pool))

        negative_pool = type_members[row["type"]]
        # Draw a same-type hard negative with a different primary taste group.
        negative = None
        for _ in range(20):
            candidate = int(rng.choice(negative_pool))
            if frame.at[candidate, "primary_genre"] != row["primary_genre"]:
                negative = candidate
                break
        if negative is not None:
            triplets.append((anchor, positive, negative))
    rng.shuffle(triplets)
    return np.asarray(triplets, dtype=np.int64)


def train_projection_network(
    train_features: np.ndarray,
    train_metadata: pd.DataFrame,
    embedding_dim: int = 128,
    epochs: int = 15,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> TrainingResult:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    triplets = _build_triplets(train_metadata, seed=seed)
    if not len(triplets):
        raise ValueError("No training triplets could be constructed")
    dataset = TensorDataset(torch.from_numpy(triplets))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    feature_tensor = torch.from_numpy(train_features.astype(np.float32))
    model = ProjectionNetwork(train_features.shape[1], embedding_dim=embedding_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.TripletMarginLoss(margin=0.25, p=2)
    losses: list[float] = []

    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        examples = 0
        for (batch_triplets,) in loader:
            anchor = model(feature_tensor[batch_triplets[:, 0]])
            positive = model(feature_tensor[batch_triplets[:, 1]])
            negative = model(feature_tensor[batch_triplets[:, 2]])
            loss = criterion(anchor, positive, negative)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(batch_triplets)
            epoch_loss += float(loss.detach()) * count
            examples += count
        losses.append(epoch_loss / examples)
    return TrainingResult(model=model, loss_history=losses, triplet_count=len(triplets))


def encode(model: ProjectionNetwork, features: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size].astype(np.float32))
            outputs.append(model(batch).cpu().numpy())
    return np.vstack(outputs).astype(np.float32)
