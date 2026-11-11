"""Train/test split, metrics, and the two non-neural baselines.

Split: per user, the last 20% of events by timestamp (at least 2) are held
out. Training only ever sees earlier events, so the model cannot peek at the
future of the same user.

Metrics at k:
    recall@k        share of a user's held-out items in the top k
    ndcg@k          rank-weighted version of recall
    coverage@k      share of the recommendable catalogue that appears in anyone's top k
    non_en_share@k  share of recommended slots filled by non-English titles
    tv_share@k      share of recommended slots filled by TV titles

Every scorer is a callable: (train_history_matrix) -> score matrix (users x items).
Items a user already watched in training are masked before ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse

Scorer = Callable[[sparse.csr_matrix], np.ndarray]


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    user_index: dict[str, int]
    item_index: dict[str, int]

    def history_matrix(self) -> sparse.csr_matrix:
        rows = self.train.user_id.map(self.user_index).to_numpy()
        cols = self.train.item_id.map(self.item_index).to_numpy()
        m = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                              shape=(len(self.user_index), len(self.item_index)))
        m.data[:] = 1.0
        return m

    def relevant(self) -> list[set[int]]:
        out: list[set[int]] = [set() for _ in self.user_index]
        for u, i in zip(self.test.user_id.map(self.user_index), self.test.item_id.map(self.item_index)):
            out[u].add(int(i))
        return out


def temporal_split(events: pd.DataFrame, item_ids: list[str], test_frac: float = 0.2) -> Split:
    events = events.sort_values(["user_id", "ts"])
    counts = events.groupby("user_id").size()
    n_test = np.maximum(2, (counts * test_frac).round().astype(int))
    rank_from_end = events.groupby("user_id").cumcount(ascending=False)
    is_test = rank_from_end < events.user_id.map(n_test).to_numpy()
    users = sorted(events.user_id.unique())
    return Split(
        train=events[~is_test].reset_index(drop=True),
        test=events[is_test].reset_index(drop=True),
        user_index={u: i for i, u in enumerate(users)},
        item_index={it: i for i, it in enumerate(item_ids)},
    )


def top_k(scores: np.ndarray, history: sparse.csr_matrix, k: int) -> np.ndarray:
    scores = scores.copy()
    scores[history.nonzero()] = -np.inf  # never recommend what was already watched
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def metrics(recs: np.ndarray, relevant: list[set[int]], item_meta: pd.DataFrame, k: int = 10) -> dict[str, float]:
    recall, ndcg = [], []
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for row, rel in zip(recs, relevant):
        if not rel:
            continue
        hits = np.array([int(i) in rel for i in row[:k]], dtype=float)
        recall.append(hits.sum() / len(rel))
        ideal = discounts[: min(len(rel), k)].sum()
        ndcg.append((hits * discounts).sum() / ideal)
    flat = recs[:, :k].ravel()
    return {
        f"recall@{k}": float(np.mean(recall)),
        f"ndcg@{k}": float(np.mean(ndcg)),
        f"coverage@{k}": float(len(np.unique(flat)) / len(item_meta)),
        f"non_en_share@{k}": float(1 - item_meta.is_english.to_numpy()[flat].mean()),
        f"tv_share@{k}": float(item_meta.type.to_numpy()[flat].__eq__("tv").mean()),
    }


# ---------------------------------------------------------------- baselines

def popularity_scorer(item_meta: pd.DataFrame) -> Scorer:
    """Same ranking for everyone: Bayesian weighted rating times popularity percentile."""
    base = (item_meta.weighted_rating * item_meta.popularity_pct).to_numpy(dtype=np.float32)

    def score(history: sparse.csr_matrix) -> np.ndarray:
        return np.tile(base, (history.shape[0], 1))

    return score


def most_watched_scorer() -> Scorer:
    """Same ranking for everyone: how many training users watched each item."""

    def score(history: sparse.csr_matrix) -> np.ndarray:
        counts = np.asarray(history.sum(axis=0)).ravel().astype(np.float32)
        return np.tile(counts, (history.shape[0], 1))

    return score


def content_knn_scorer(item_features: np.ndarray) -> Scorer:
    """No learning: a user is the mean of the feature rows they watched; rank by cosine."""
    feats = item_features / np.maximum(np.linalg.norm(item_features, axis=1, keepdims=True), 1e-8)

    def score(history: sparse.csr_matrix) -> np.ndarray:
        counts = np.maximum(np.asarray(history.sum(axis=1)), 1)
        profile = np.asarray(history @ feats) / counts
        profile /= np.maximum(np.linalg.norm(profile, axis=1, keepdims=True), 1e-8)
        return (profile @ feats.T).astype(np.float32)

    return score


def evaluate(scorer: Scorer, split: Split, item_meta: pd.DataFrame, k: int = 10) -> dict[str, float]:
    history = split.history_matrix()
    recs = top_k(scorer(history), history, k)
    return metrics(recs, split.relevant(), item_meta, k)


def two_tower_scorer(item_table: np.ndarray) -> Scorer:
    """numpy-only inference for the trained model: user = mean of watched item
    embeddings, score = cosine. Lives here so serving never imports torch."""
    table = item_table / np.maximum(np.linalg.norm(item_table, axis=1, keepdims=True), 1e-8)

    def score(history: sparse.csr_matrix) -> np.ndarray:
        counts = np.maximum(np.asarray(history.sum(axis=1)), 1)
        user = np.asarray(history @ table) / counts
        user /= np.maximum(np.linalg.norm(user, axis=1, keepdims=True), 1e-8)
        return (user @ table.T).astype(np.float32)

    return score
