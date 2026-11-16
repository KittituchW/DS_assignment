from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

torch = pytest.importorskip("torch")

from recommender.two_tower import TrainConfig, train_two_tower  # noqa: E402


def _toy_problem(n_users: int = 30, n_items: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    feats = rng.normal(size=(n_items, 12)).astype(np.float32)
    history = sparse.random(n_users, n_items, density=0.2, random_state=seed, format="csr", dtype=np.float32)
    history.data[:] = 1.0
    return feats, history


def test_item_table_has_unit_rows_and_loss_falls() -> None:
    feats, history = _toy_problem()
    _, table, losses = train_two_tower(feats, history, TrainConfig(embed_dim=8, hidden=16, epochs=15, batch_size=64), log=False)
    assert table.shape == (feats.shape[0], 8)
    np.testing.assert_allclose(np.linalg.norm(table, axis=1), 1.0, atol=1e-5)
    assert losses[-1] < losses[0]


def test_same_seed_gives_the_same_table() -> None:
    feats, history = _toy_problem()
    cfg = TrainConfig(embed_dim=8, hidden=16, epochs=2, batch_size=64, seed=3)
    _, a, _ = train_two_tower(feats, history, cfg, log=False)
    _, b, _ = train_two_tower(feats, history, cfg, log=False)
    np.testing.assert_allclose(a, b)


def test_text_mask_removes_the_masked_columns() -> None:
    feats, history = _toy_problem()
    mask = np.ones(feats.shape[1], dtype=np.float32)
    mask[6:] = 0.0
    changed = feats.copy()
    changed[:, 6:] = 99.0  # masked columns must not matter
    cfg = TrainConfig(embed_dim=8, hidden=16, epochs=2, batch_size=64, seed=3)
    _, a, _ = train_two_tower(feats, history, cfg, feature_mask=mask, log=False)
    _, b, _ = train_two_tower(changed, history, cfg, feature_mask=mask, log=False)
    np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("activation", ["relu", "gelu", "silu"])
def test_every_activation_in_the_search_space_trains(activation: str) -> None:
    feats, history = _toy_problem()
    cfg = TrainConfig(embed_dim=8, hidden=16, epochs=10, batch_size=64, activation=activation)
    _, table, losses = train_two_tower(feats, history, cfg, log=False)
    np.testing.assert_allclose(np.linalg.norm(table, axis=1), 1.0, atol=1e-5)
    assert losses[-1] < losses[0]
