"""Split, per-viewer metrics, baselines, numpy inference and the bootstrap.

Split: for each viewer, the last 20% of events by time (at least 2) are held
out. Training never sees a viewer's future.

Metrics at k, computed per viewer so they can be bootstrapped:
    recall@k        share of the viewer's held-out items found in the top k
    ndcg@k          rank-weighted recall
    hit@k           1 if at least one held-out item is in the top k (exploratory)
    recall_w@k      recall where each held-out item counts by its watch ratio, so a
                    title the viewer finished counts more than one they quit (exploratory)
    non_en@k        share of the top k that is non-English
    tv@k            share of the top k that is TV
    heldout_non_en  share of the viewer's held-out items that is non-English (H2 reference)
    counts          movie/TV and non-English counts in the top k and in the held-out items,
                    pooled later to read H2 within each content type
Catalogue level:
    coverage@k      share of recommendable items that appear in anyone's top k

Cold start (exploratory): `cold_start_split` hides a share of titles from
training entirely and `evaluate_cold` ranks only those titles.

A scorer maps a block of viewer history rows to a score matrix (viewers x items).
Items a viewer already watched in training are never recommended.
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
    user_ids: list[str]
    item_index: dict[str, int]

    def _matrix(self, frame: pd.DataFrame, values: np.ndarray | None = None) -> sparse.csr_matrix:
        user_index = {u: i for i, u in enumerate(self.user_ids)}
        rows = frame["user_id"].map(user_index).to_numpy()
        cols = frame["item_id"].map(self.item_index).to_numpy()
        m = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32) if values is None else values.astype(np.float32), (rows, cols)),
            shape=(len(self.user_ids), len(self.item_index)),
        )
        if values is None:
            m.data[:] = 1.0
        m.sort_indices()
        return m

    def history_matrix(self) -> sparse.csr_matrix:
        return self._matrix(self.train)

    def test_matrix(self) -> sparse.csr_matrix:
        return self._matrix(self.test)

    def test_ratio_matrix(self) -> sparse.csr_matrix | None:
        """Held-out items weighted by watch ratio, or None when the events carry no ratio."""
        if "watch_ratio" not in self.test.columns:
            return None
        return self._matrix(self.test, self.test["watch_ratio"].to_numpy())


def temporal_split(events: pd.DataFrame, item_ids: list[str], test_frac: float = 0.2) -> Split:
    events = events.sort_values(["user_id", "ts"], kind="stable")
    counts = events.groupby("user_id").size()
    n_test = np.maximum(2, (counts * test_frac).round().astype(int))
    rank_from_end = events.groupby("user_id").cumcount(ascending=False)
    is_test = rank_from_end.to_numpy() < events["user_id"].map(n_test).to_numpy()
    return Split(
        train=events[~is_test].reset_index(drop=True),
        test=events[is_test].reset_index(drop=True),
        user_ids=sorted(events["user_id"].unique()),
        item_index={it: i for i, it in enumerate(item_ids)},
    )


def validation_split(split: Split, val_frac: float = 0.2) -> Split:
    """For tuning only: carve each viewer's latest training events into a validation set.

    Same rule as the test split, applied to the training events, so the test set
    is never seen while choosing settings.
    """
    return temporal_split(split.train, list(split.item_index), val_frac)


def cold_start_split(events: pd.DataFrame, items: pd.DataFrame, frac: float = 0.1, seed: int = 0) -> tuple[Split, np.ndarray]:
    """Item cold start: hold out a share of titles, stratified by type, from training entirely.

    Train is every event on the other titles. Test is every event on the held-out
    titles, so no model has seen a single watch of them. Returns the split and a
    boolean mask over `items` marking the held-out ("new") titles.
    """
    rng = np.random.default_rng(seed)
    cold = np.zeros(len(items), dtype=bool)
    for positions in items.reset_index(drop=True).groupby("type").indices.values():
        size = max(1, int(round(len(positions) * frac)))
        cold[rng.choice(positions, size=size, replace=False)] = True
    is_cold = events["item_id"].isin(set(items["item_id"].to_numpy()[cold])).to_numpy()
    split = Split(
        train=events[~is_cold].reset_index(drop=True),
        test=events[is_cold].reset_index(drop=True),
        user_ids=sorted(events["user_id"].unique()),
        item_index={it: i for i, it in enumerate(items["item_id"])},
    )
    return split, cold


def top_k(scores: np.ndarray, history: sparse.csr_matrix, k: int) -> np.ndarray:
    scores = np.array(scores, dtype=np.float32, copy=True)
    scores[history.nonzero()] = -np.inf  # never recommend what was already watched
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def recommend(scorer: Scorer, history: sparse.csr_matrix, k: int = 10, block: int = 500) -> np.ndarray:
    """Top k item positions for every viewer, scored in blocks to bound memory."""
    out = [top_k(scorer(history[s : s + block]), history[s : s + block], k) for s in range(0, history.shape[0], block)]
    return np.vstack(out)


def per_user_metrics(
    recs: np.ndarray, test: sparse.csr_matrix, items: pd.DataFrame, k: int = 10, ratio: sparse.csr_matrix | None = None
) -> dict[str, np.ndarray]:
    non_en_item = (~items["is_english"].to_numpy()).astype(bool)
    tv_item = items["type"].eq("tv").to_numpy()
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    test = test.tocsr()
    test.sort_indices()
    if ratio is not None:
        ratio = ratio.tocsr()
        ratio.sort_indices()
        if not (np.array_equal(ratio.indptr, test.indptr) and np.array_equal(ratio.indices, test.indices)):
            raise ValueError("ratio matrix must hold the same held-out items as the test matrix")
    recall, ndcg, heldout, hit, recall_w = [], [], [], [], []
    held_counts = np.zeros((recs.shape[0], 4), dtype=np.int32)  # movie, movie non-EN, tv, tv non-EN
    for u in range(recs.shape[0]):
        relevant = test.indices[test.indptr[u] : test.indptr[u + 1]]
        hits = np.isin(recs[u, :k], relevant).astype(float)
        n_rel = max(len(relevant), 1)
        recall.append(hits.sum() / n_rel)
        ndcg.append((hits * discounts).sum() / discounts[: min(n_rel, k)].sum())
        hit.append(float(hits.sum() > 0))
        if ratio is not None:
            w = ratio.data[ratio.indptr[u] : ratio.indptr[u + 1]]
            found = np.isin(relevant, recs[u, :k])
            recall_w.append((w * found).sum() / w.sum() if w.sum() > 0 else np.nan)
        heldout.append(non_en_item[relevant].mean() if len(relevant) else np.nan)
        is_tv, is_non_en = tv_item[relevant], non_en_item[relevant]
        held_counts[u] = [(~is_tv).sum(), (~is_tv & is_non_en).sum(), is_tv.sum(), (is_tv & is_non_en).sum()]
    top_tv, top_non_en = tv_item[recs[:, :k]], non_en_item[recs[:, :k]]
    out = {
        "recall": np.asarray(recall),
        "ndcg": np.asarray(ndcg),
        "hit": np.asarray(hit),
        "non_en": top_non_en.mean(axis=1),
        "tv": top_tv.mean(axis=1),
        "heldout_non_en": np.asarray(heldout),
        "rec_movie": (~top_tv).sum(axis=1),
        "rec_movie_non_en": (~top_tv & top_non_en).sum(axis=1),
        "rec_tv": top_tv.sum(axis=1),
        "rec_tv_non_en": (top_tv & top_non_en).sum(axis=1),
        "heldout_movie": held_counts[:, 0],
        "heldout_movie_non_en": held_counts[:, 1],
        "heldout_tv": held_counts[:, 2],
        "heldout_tv_non_en": held_counts[:, 3],
    }
    if ratio is not None:
        out["recall_w"] = np.asarray(recall_w)
    return out


def summarise(per_user: dict[str, np.ndarray], recs: np.ndarray, n_items: int, k: int = 10) -> dict[str, float]:
    out = {
        f"recall@{k}": float(np.nanmean(per_user["recall"])),
        f"ndcg@{k}": float(np.nanmean(per_user["ndcg"])),
        f"hit@{k}": float(np.nanmean(per_user["hit"])),
        f"coverage@{k}": float(len(np.unique(recs[:, :k])) / n_items),
        f"non_en@{k}": float(np.nanmean(per_user["non_en"])),
        f"tv@{k}": float(np.nanmean(per_user["tv"])),
        "heldout_non_en": float(np.nanmean(per_user["heldout_non_en"])),
    }
    if "recall_w" in per_user:
        out[f"recall_w@{k}"] = float(np.nanmean(per_user["recall_w"]))
    return out


def evaluate(scorer: Scorer, split: Split, items: pd.DataFrame, k: int = 10) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    history = split.history_matrix()
    recs = recommend(scorer, history, k)
    per_user = per_user_metrics(recs, split.test_matrix(), items, k, ratio=split.test_ratio_matrix())
    return summarise(per_user, recs, len(items), k), per_user


def evaluate_cold(
    scorer: Scorer, split: Split, items: pd.DataFrame, cold: np.ndarray, k: int = 10
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Rank only the held-out titles ("new this week" row). Viewers with no held-out watch are left out."""

    def cold_only(history: sparse.csr_matrix) -> np.ndarray:
        scores = np.array(scorer(history), dtype=np.float32, copy=True)
        scores[:, ~cold] = -np.inf
        return scores

    recs = recommend(cold_only, split.history_matrix(), k)
    test = split.test_matrix()
    per_user = per_user_metrics(recs, test, items, k, ratio=split.test_ratio_matrix())
    has_test = np.diff(test.indptr) > 0
    per_user = {key: values[has_test] for key, values in per_user.items()}
    summary = {
        f"recall@{k}": float(np.nanmean(per_user["recall"])),
        f"ndcg@{k}": float(np.nanmean(per_user["ndcg"])),
        f"hit@{k}": float(np.nanmean(per_user["hit"])),
        f"cold_coverage@{k}": float(len(np.unique(recs[has_test, :k])) / max(int(cold.sum()), 1)),
        "viewers": int(has_test.sum()),
        "cold_items": int(cold.sum()),
        f"random_recall@{k}": float(k / max(int(cold.sum()), 1)),
    }
    return summary, per_user


def paired_bootstrap(diff: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of a per-viewer difference and its 95% percentile interval."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


# ------------------------------------------------------------------ baselines


def popularity_scorer(items: pd.DataFrame) -> Scorer:
    """Same list for everyone: weighted rating times popularity percentile (TMDb signals)."""
    base = (items["weighted_rating"] * items["popularity_pct"]).to_numpy(dtype=np.float32)
    return lambda history: np.tile(base, (history.shape[0], 1))


def popularity_by_type_scorer(items: pd.DataFrame) -> Scorer:
    """Exploratory check, not part of the H2 rule: rating as a percentile within type.

    EDA showed TV titles score about one point higher than movies, so the raw
    weighted rating pushes the popularity list towards TV. This version puts
    both types on the same scale.
    """
    rating_pct = items.groupby("type")["weighted_rating"].rank(pct=True)
    base = (rating_pct * items["popularity_pct"]).to_numpy(dtype=np.float32)
    return lambda history: np.tile(base, (history.shape[0], 1))


def most_watched_scorer(train_history: sparse.csr_matrix) -> Scorer:
    """Same list for everyone: how many simulated viewers watched each item in training."""
    counts = np.asarray(train_history.sum(axis=0)).ravel().astype(np.float32)
    return lambda history: np.tile(counts, (history.shape[0], 1))


def _mean_profile_scorer(table: np.ndarray) -> Scorer:
    table = table / np.maximum(np.linalg.norm(table, axis=1, keepdims=True), 1e-8)

    def score(history: sparse.csr_matrix) -> np.ndarray:
        counts = np.maximum(np.asarray(history.sum(axis=1)), 1)
        profile = np.asarray(history @ table) / counts
        profile /= np.maximum(np.linalg.norm(profile, axis=1, keepdims=True), 1e-8)
        return (profile @ table.T).astype(np.float32)

    return score


def content_knn_scorer(item_features: np.ndarray) -> Scorer:
    """No learning: a viewer is the mean of the feature rows they watched; rank by cosine."""
    return _mean_profile_scorer(item_features)


def two_tower_scorer(item_table: np.ndarray) -> Scorer:
    """numpy-only inference for the trained model: viewer = mean of watched item vectors."""
    return _mean_profile_scorer(item_table)
