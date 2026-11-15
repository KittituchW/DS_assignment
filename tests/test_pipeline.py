"""End-to-end run of every stage on a tiny catalogue.

PyTorch is replaced by a stand-in that returns a random item table, so this
test checks the wiring (files, splits, analysis) without training a network.
`test_two_tower.py` covers the real model when PyTorch is installed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass

import numpy as np

from conftest import ROOT, make_raw
from recommender.data import MOVIES_FILE, TV_FILE


def _fake_two_tower() -> types.ModuleType:
    module = types.ModuleType("recommender.two_tower")

    @dataclass
    class TrainConfig:
        epochs: int = 1
        seed: int = 0

    def train_two_tower(item_features, history, cfg, feature_mask=None, log=True):
        table = np.random.default_rng(cfg.seed).normal(size=(item_features.shape[0], 8)).astype(np.float32)
        return None, table, [1.0]

    module.TrainConfig = TrainConfig
    module.train_two_tower = train_two_tower
    return module


def test_all_stages_run_and_write_their_outputs(tmp_path, monkeypatch) -> None:
    raw = make_raw(n_per_type=80)
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    raw[raw.type == "Movie"].to_csv(data_dir / MOVIES_FILE, index=False)
    raw[raw.type == "TV Show"].drop(columns=["budget", "revenue"]).to_csv(data_dir / TV_FILE, index=False)

    monkeypatch.setitem(sys.modules, "recommender.two_tower", _fake_two_tower())
    spec = importlib.util.spec_from_file_location("run_experiment", ROOT / "run_experiment.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    out = tmp_path / "artifacts"
    monkeypatch.setattr(
        sys, "argv",
        ["run_experiment.py", "--data-dir", str(data_dir), "--output", str(out), "--users", "40",
         "--exponents", "2", "6", "--seeds", "1", "2", "--epochs", "1"],
    )
    runner.main()

    for name in (
        "eda_summary.json", "eda_schema.csv", "metrics_all_runs.csv", "hypothesis_tests.json",
        "results_summary.md", "h2_non_english_gap.png", "h1_text_ablation.png", "sample_recommendations.md",
    ):
        assert (out / name).exists(), name
    tests = json.loads((out / "hypothesis_tests.json").read_text())
    assert tests["H1"]["verdict"] in {"supported", "rejected: text hurts", "inconclusive"}
    assert tests["H2"]["h2b_verdict"] in {"supported", "not supported"}
    runs = (out / "metrics_all_runs.csv").read_text()
    assert "two_tower_no_text" in runs and "content_knn_text_only" in runs
