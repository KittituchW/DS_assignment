from __future__ import annotations

import numpy as np
import pandas as pd

from recommender.data import build_catalogue, recommendable_items
from recommender.features import build_item_features
from recommender.simulate import SimConfig, simulate


def _items(raw: pd.DataFrame) -> pd.DataFrame:
    return recommendable_items(build_catalogue(raw))


def test_feature_groups_cover_the_matrix_and_the_text_mask_is_exact(raw: pd.DataFrame) -> None:
    feats = build_item_features(_items(raw))
    spans = sorted((s.start, s.stop) for s in feats.groups.values())
    assert spans[0][0] == 0 and spans[-1][1] == feats.matrix.shape[1]
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    mask = feats.mask_without("text")
    assert mask[feats.groups["text"]].sum() == 0
    assert mask.sum() == feats.matrix.shape[1] - (feats.groups["text"].stop - feats.groups["text"].start)


def test_simulation_is_deterministic_and_valid(raw: pd.DataFrame) -> None:
    items = _items(raw)
    cfg = SimConfig(users=20, pop_exponent=6, seed=7, min_events=3, max_events=10)
    a = simulate(items, cfg)
    b = simulate(items, cfg)
    pd.testing.assert_frame_equal(a, b)
    assert set(a.item_id) <= set(items.item_id)
    assert not a.duplicated(["user_id", "item_id"]).any()
    sizes = a.groupby("user_id").size()
    assert sizes.min() >= 3 and sizes.max() <= 10
    assert (a.ts <= pd.Timestamp(cfg.as_of)).all()


def test_higher_popularity_exponent_concentrates_on_popular_items(raw: pd.DataFrame) -> None:
    items = _items(raw)
    pop = items.set_index("item_id")["popularity_pct"]
    low = simulate(items, SimConfig(users=40, pop_exponent=0, seed=1, min_events=3, max_events=10))
    high = simulate(items, SimConfig(users=40, pop_exponent=12, seed=1, min_events=3, max_events=10))
    assert np.mean(pop[high.item_id]) > np.mean(pop[low.item_id])
