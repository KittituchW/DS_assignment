"""Random search for the two tower settings, scored on a validation split.

Added after the main run (see the 2026-11-16 entries in HYPOTHESES.md). The
search trains on each viewer's earlier training events and scores Recall@10 on
their latest training events. The test set is used once, for the chosen
settings only. The pre-registered results are kept as they are; the tuned
model is reported next to them.
"""

from __future__ import annotations

import numpy as np

SEARCH_SPACE: dict[str, list] = {
    "activation": ["relu", "gelu", "silu"],
    "embed_dim": [32, 64, 128],
    "hidden": [128, 256, 512],
    "dropout": [0.0, 0.1, 0.2, 0.4],
    "lr": [5e-4, 1e-3, 3e-3],
    "temperature": [0.05, 0.07, 0.1],
}

# The pre-registered settings, always tried first so the search has a reference.
DEFAULT: dict = {"activation": "relu", "embed_dim": 64, "hidden": 256, "dropout": 0.2, "lr": 1e-3, "temperature": 0.07}


def sample_configs(n: int, seed: int = 0) -> list[dict]:
    """The default first, then distinct random draws from the search space."""
    rng = np.random.default_rng(seed)
    configs = [dict(DEFAULT)]
    seen = {tuple(sorted(DEFAULT.items()))}
    while len(configs) < n:
        config = {key: values[int(rng.integers(len(values)))] for key, values in SEARCH_SPACE.items()}
        key = tuple(sorted(config.items()))
        if key not in seen:
            seen.add(key)
            configs.append(config)
    return configs


def pick_best(trials: list[dict], metric: str = "val_recall@10", tie_break: str = "val_ndcg@10") -> int:
    """Index of the best trial by validation Recall@10, NDCG@10 breaking ties."""
    return max(range(len(trials)), key=lambda i: (trials[i][metric], trials[i][tie_break]))
