"""Item feature engineering.

Produces a dense float32 matrix, one row per catalogue item, plus a column
index so the model can ablate feature groups (used by hypothesis 1).

Feature groups:
    genre     multi-hot over the harmonised genre vocabulary
    language  one-hot over the top languages plus "other"
    country   one-hot over the top countries plus "other"
    type      movie / tv
    era       one-hot release half-decade
    numeric   popularity percentile, log1p vote count, weighted rating, has_financials
    text      TF-IDF (title + description) reduced by truncated SVD, L2 normalised

All vocabularies are fit on the catalogue only. There is no user data, so
there is nothing to leak from a future split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    matrix: np.ndarray  # (n_items, n_features) float32
    groups: dict[str, slice] = field(default_factory=dict)
    item_ids: list[str] = field(default_factory=list)
    genre_vocab: list[str] = field(default_factory=list)

    def without(self, *drop: str) -> np.ndarray:
        """Return the matrix with the named feature groups zeroed out (ablation)."""
        out = self.matrix.copy()
        for name in drop:
            out[:, self.groups[name]] = 0.0
        return out

    @property
    def dim(self) -> int:
        return self.matrix.shape[1]


def _one_hot(series: pd.Series, top: int) -> tuple[np.ndarray, list[str]]:
    vocab = list(series.value_counts().head(top).index)
    index = {v: i for i, v in enumerate(vocab)}
    out = np.zeros((len(series), len(vocab) + 1), dtype=np.float32)
    for row, value in enumerate(series):
        out[row, index.get(value, len(vocab))] = 1.0
    return out, vocab + ["other"]


def _multi_hot(lists: pd.Series) -> tuple[np.ndarray, list[str]]:
    vocab = sorted({g for items in lists for g in items})
    index = {v: i for i, v in enumerate(vocab)}
    out = np.zeros((len(lists), len(vocab)), dtype=np.float32)
    for row, items in enumerate(lists):
        for g in items:
            out[row, index[g]] = 1.0
    return out, vocab


def text_embeddings(catalogue: pd.DataFrame, dims: int = TEXT_DIMS, seed: int = 42) -> np.ndarray:
    """TF-IDF over title and description, reduced with truncated SVD.

    A pretrained sentence encoder would be the first upgrade in production. It is
    not used here so the project installs with scikit-learn only.
    """
    docs = catalogue["title"].fillna("") + ". " + catalogue["description"].fillna("")
    vec = TfidfVectorizer(max_features=40_000, min_df=2, ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    tfidf = vec.fit_transform(docs)
    svd = TruncatedSVD(n_components=dims, random_state=seed)
    emb = svd.fit_transform(tfidf).astype(np.float32)
    return normalize(emb).astype(np.float32)


def build_item_features(catalogue: pd.DataFrame, seed: int = 42) -> ItemFeatures:
    blocks: list[np.ndarray] = []
    groups: dict[str, slice] = {}
    cursor = 0

    def add(name: str, block: np.ndarray) -> None:
        nonlocal cursor
        groups[name] = slice(cursor, cursor + block.shape[1])
        blocks.append(block.astype(np.float32))
        cursor += block.shape[1]

    genre, genre_vocab = _multi_hot(catalogue["genres"])
    add("genre", genre)
    add("language", _one_hot(catalogue["language"], TOP_LANGUAGES)[0])
    first_country = catalogue["country"].map(lambda c: c[0] if c else "unknown")
    add("country", _one_hot(first_country, TOP_COUNTRIES)[0])
    add("type", _one_hot(catalogue["type"], 2)[0][:, :2])
    era = (catalogue["release_year"] // 5 * 5).astype(str)
    add("era", _one_hot(era, 10)[0])
    numeric = np.column_stack([
        catalogue["popularity_pct"].to_numpy(),
        np.log1p(catalogue["vote_count"].to_numpy()) / 10.0,
        catalogue["weighted_rating"].to_numpy() / 10.0,
        catalogue["has_financials"].astype(float).to_numpy(),
    ])
    add("numeric", numeric)
    add("text", text_embeddings(catalogue, seed=seed))

    return ItemFeatures(
        matrix=np.hstack(blocks),
        groups=groups,
        item_ids=list(catalogue["item_id"]),
        genre_vocab=genre_vocab,
    )
