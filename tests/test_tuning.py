from __future__ import annotations

from recommender.tuning import DEFAULT, SEARCH_SPACE, pick_best, sample_configs


def test_search_starts_from_the_preregistered_default_and_is_repeatable() -> None:
    configs = sample_configs(12, seed=3)
    assert configs[0] == DEFAULT
    assert configs == sample_configs(12, seed=3)
    assert len({tuple(sorted(c.items())) for c in configs}) == 12
    for c in configs:
        assert all(c[k] in SEARCH_SPACE[k] for k in SEARCH_SPACE)


def test_pick_best_uses_validation_recall_then_ndcg() -> None:
    trials = [
        {"val_recall@10": 0.02, "val_ndcg@10": 0.03},
        {"val_recall@10": 0.03, "val_ndcg@10": 0.01},
        {"val_recall@10": 0.03, "val_ndcg@10": 0.02},
    ]
    assert pick_best(trials) == 2
