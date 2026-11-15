from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from recommender.evaluation import paired_bootstrap, per_user_metrics, temporal_split, top_k


def _events() -> pd.DataFrame:
    rows = [("a", f"i{j}", pd.Timestamp("2024-01-01") + pd.Timedelta(days=j)) for j in range(10)]
    rows += [("b", f"i{j}", pd.Timestamp("2024-01-01") + pd.Timedelta(days=10 - j)) for j in range(3)]
    return pd.DataFrame(rows, columns=["user_id", "item_id", "ts"])


def test_split_holds_out_each_viewers_latest_events() -> None:
    split = temporal_split(_events(), [f"i{j}" for j in range(10)])
    test_a = split.test[split.test.user_id == "a"]
    assert set(test_a.item_id) == {"i8", "i9"}  # last 20% of 10 events
    test_b = split.test[split.test.user_id == "b"]
    assert set(test_b.item_id) == {"i0", "i1"}  # at least 2, latest timestamps
    for user in ("a", "b"):
        train_items = set(split.train[split.train.user_id == user].item_id)
        test_items = set(split.test[split.test.user_id == user].item_id)
        assert not train_items & test_items
        assert split.train[split.train.user_id == user].ts.max() <= split.test[split.test.user_id == user].ts.min()


def test_top_k_never_returns_watched_items() -> None:
    scores = np.array([[0.9, 0.8, 0.1, 0.2]], dtype=np.float32)
    history = sparse.csr_matrix(np.array([[1, 0, 0, 0]], dtype=np.float32))
    assert top_k(scores, history, 2).tolist() == [[1, 3]]


def test_metrics_on_a_known_example() -> None:
    items = pd.DataFrame({"is_english": [True, False, True, False, True, True], "type": ["movie", "tv"] * 3})
    recs = np.array([[0, 1, 2]])
    test = sparse.csr_matrix(np.array([[0, 1, 0, 0, 0, 1]], dtype=np.float32))
    m = per_user_metrics(recs, test, items, k=3)
    assert m["recall"][0] == pytest.approx(0.5)
    assert m["ndcg"][0] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))
    assert m["non_en"][0] == pytest.approx(1 / 3)
    assert m["heldout_non_en"][0] == pytest.approx(0.5)


def test_bootstrap_of_a_constant_is_that_constant() -> None:
    mean, lo, hi = paired_bootstrap(np.full(50, 0.1))
    assert mean == pytest.approx(0.1) and lo == pytest.approx(0.1) and hi == pytest.approx(0.1)


def test_bootstrap_interval_contains_the_mean() -> None:
    diff = np.random.default_rng(0).normal(0.02, 0.1, size=500)
    mean, lo, hi = paired_bootstrap(diff)
    assert lo < mean < hi
