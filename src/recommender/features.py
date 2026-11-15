"""Item feature engineering.

Builds one dense float32 row per recommendable item, with named feature groups
so the model can drop a group (the H1 ablation drops `text`).

Feature groups:
    genre     multi-hot over the harmonised genre vocabulary
    language  one-hot over the top languages plus "other"
    country   one-hot over the top first countries plus "other"
    type      movie or tv
    era       one-hot release half-decade
    numeric   popularity percentile, log vote count, weighted rating, has_financials
    text      TF-IDF over title and description, reduced by truncated SVD, L2 normalised

Features use item metadata only. There is no viewer data in them, so the user
split cannot leak through the features.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

TOP_LANGUAGES = 12
TOP_COUNTRIES = 12
TEXT_DIMS = 128


@dataclass
class ItemFeatures:
    matrix: np.ndarray
    groups: dict[str, slice] = field(default_factory=dict)
    item_ids: list[str] = field(default_factory=list)

    def mask_without(self, *drop: str) -> np.ndarray:
        """A 0/1 column mask with the named groups switched off."""
        mask = np.ones(self.matrix.shape[1], dtype=np.float32)
        for name in drop:
            mask[self.groups[name]] = 0.0
        return mask

    def columns(self, *keep: str) -> np.ndarray:
        """The matrix restricted to the named groups."""
        return np.hstack([self.matrix[:, self.groups[name]] for name in keep])

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            matrix=self.matrix,
            item_ids=np.array(self.item_ids),
            groups=json.dumps({k: [v.start, v.stop] for k, v in self.groups.items()}),
        )

    @classmethod
    def load(cls, path: Path) -> "ItemFeatures":
        npz = np.load(path, allow_pickle=False)
        groups = {k: slice(a, b) for k, (a, b) in json.loads(str(npz["groups"])).items()}
        return cls(matrix=npz["matrix"], groups=groups, item_ids=list(npz["item_ids"]))


def _one_hot(values: pd.Series, top: int) -> np.ndarray:
    vocab = list(values.value_counts().head(top).index)
    index = {v: i for i, v in enumerate(vocab)}
    out = np.zeros((len(values), len(vocab) + 1), dtype=np.float32)
    for row, value in enumerate(values):
        out[row, index.get(value, len(vocab))] = 1.0
    return out


def _multi_hot(lists: pd.Series) -> np.ndarray:
    vocab = sorted({g for items in lists for g in items})
    index = {v: i for i, v in enumerate(vocab)}
    out = np.zeros((len(lists), len(vocab)), dtype=np.float32)
    for row, items in enumerate(lists):
        for g in items:
            out[row, index[g]] = 1.0
    return out


def text_embeddings(items: pd.DataFrame, dims: int = TEXT_DIMS, seed: int = 42) -> np.ndarray:
    """TF-IDF over title and description, then truncated SVD.

    A pretrained sentence encoder is the planned upgrade. Descriptions are in
    English for every original language, so an English model is enough. Only
    the input size of the item tower would change.
    """
    docs = items["title"] + ". " + items["description"]
    tfidf = TfidfVectorizer(
        max_features=40_000, min_df=2, ngram_range=(1, 2), sublinear_tf=True, stop_words="english"
    ).fit_transform(docs)
    dims = min(dims, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    emb = TruncatedSVD(n_components=dims, random_state=seed).fit_transform(tfidf)
    return normalize(emb).astype(np.float32)


def build_item_features(items: pd.DataFrame, seed: int = 42) -> ItemFeatures:
    blocks: list[np.ndarray] = []
    groups: dict[str, slice] = {}
    cursor = 0

    def add(name: str, block: np.ndarray) -> None:
        nonlocal cursor
        groups[name] = slice(cursor, cursor + block.shape[1])
        blocks.append(block.astype(np.float32))
        cursor += block.shape[1]

    add("genre", _multi_hot(items["genres"]))
    add("language", _one_hot(items["language"], TOP_LANGUAGES))
    add("country", _one_hot(items["country"].map(lambda c: c[0] if c else "unknown"), TOP_COUNTRIES))
    add("type", np.column_stack([items["type"].eq("movie"), items["type"].eq("tv")]).astype(np.float32))
    add("era", _one_hot((items["release_year"] // 5 * 5).astype(str), 10))
    add(
        "numeric",
        np.column_stack(
            [
                items["popularity_pct"].to_numpy(dtype=float),
                np.log1p(items["vote_count"].to_numpy(dtype=float)) / 10.0,
                items["weighted_rating"].to_numpy(dtype=float) / 10.0,
                items["has_financials"].astype(float).to_numpy(),
            ]
        ),
    )
    add("text", text_embeddings(items, seed=seed))
    return ItemFeatures(matrix=np.hstack(blocks), groups=groups, item_ids=list(items["item_id"]))
