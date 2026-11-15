from __future__ import annotations

import numpy as np

from recommender.analysis import check_h1, check_h2, save_run

N = 400
RNG = np.random.default_rng(0)
HELDOUT = np.full(N, 0.4)


def _save(run_dir, model, p, seed, recall, non_en):
    per_user = {
        "recall": recall,
        "ndcg": recall * 0.8,
        "non_en": non_en,
        "tv": np.full(N, 0.3),
        "heldout_non_en": HELDOUT,
    }
    save_run(run_dir, model, p, seed, {"recall@10": float(recall.mean())}, per_user)


def test_h1_supported_when_text_helps_everyone(tmp_path) -> None:
    base = RNG.uniform(0, 0.1, N)
    for s in (1, 2):
        _save(tmp_path, "two_tower", 6, s, base + 0.02, HELDOUT)
        _save(tmp_path, "two_tower_no_text", 6, s, base, HELDOUT)
    assert check_h1(tmp_path, 6, [1, 2])["verdict"] == "supported"


def test_h1_inconclusive_when_arms_are_equal_on_average(tmp_path) -> None:
    base = RNG.uniform(0, 0.1, N)
    noise = RNG.normal(0, 0.02, N)
    noise -= noise.mean()
    for s in (1, 2):
        _save(tmp_path, "two_tower", 6, s, base + noise, HELDOUT)
        _save(tmp_path, "two_tower_no_text", 6, s, base, HELDOUT)
    assert check_h1(tmp_path, 6, [1, 2])["verdict"] == "inconclusive"


def test_h1_reports_missing_runs(tmp_path) -> None:
    assert check_h1(tmp_path, 6, [1])["verdict"] == "not run"


def test_h2_rules(tmp_path) -> None:
    zeros = np.zeros(N)
    for p in (2, 6):
        _save(tmp_path, "popularity", p, None, zeros, np.full(N, 0.01))
        _save(tmp_path, "most_watched", p, None, zeros, np.full(N, 0.3))
        _save(tmp_path, "content_knn", p, None, zeros, np.full(N, 0.15))
    _save(tmp_path, "two_tower", 2, 1, zeros, HELDOUT + RNG.normal(0, 0.05, N))
    for s in (1, 2):
        _save(tmp_path, "two_tower", 6, s, zeros, HELDOUT + RNG.normal(0, 0.05, N))
    result = check_h2(tmp_path, [2, 6], 6, [1, 2])
    assert result["h2a_verdict"] == "supported"
    assert result["h2b_verdict"] == "supported"

    _save(tmp_path, "two_tower", 2, 1, zeros, HELDOUT - 0.2)  # now far below the held-out share
    assert check_h2(tmp_path, [2, 6], 6, [1, 2])["h2b_verdict"] == "not supported"
