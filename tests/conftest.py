from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GENRES = ["Action", "Adventure", "Drama", "Romance", "Comedy", "Animation", "Documentary", "Horror", "Crime"]
LANGS = ["en", "en", "ko", "ja", "zh"]
WORDS = "space war love city family ghost detective ocean robot village secret island school dragon".split()


def make_raw(n_per_type: int = 60, seed: int = 0) -> pd.DataFrame:
    """A small raw table with the same columns and quirks as the real files."""
    rng = np.random.default_rng(seed)
    rows = []
    for kind in ("Movie", "TV Show"):
        for i in range(n_per_type):
            year = int(rng.integers(2015, 2025))
            g = rng.choice(GENRES, size=2, replace=False)
            genres = ", ".join(g)
            if kind == "TV Show" and i % 7 == 0:
                genres = "Action & Adventure, Drama"
            rows.append(
                {
                    "show_id": 1000 + i,  # same ids in both files: a deliberate collision
                    "type": kind,
                    "title": f"{kind} title {i}",
                    "director": "A Director" if kind == "Movie" else np.nan,
                    "cast": "Actor One, Actor Two",
                    "country": "United States of America",
                    "date_added": f"{year}-03-01",
                    "release_year": year,
                    "rating": 7.0,
                    "duration": np.nan if kind == "Movie" else "1 Seasons",
                    "genres": genres,
                    "language": LANGS[i % len(LANGS)],
                    "description": " ".join(rng.choice(WORDS, size=8)),
                    "popularity": float(rng.uniform(1, 100)),
                    "vote_count": int(rng.integers(5, 5000)),
                    "vote_average": 7.0,
                    "budget": 0.0 if kind == "Movie" else np.nan,
                    "revenue": 0.0 if kind == "Movie" else np.nan,
                }
            )
    raw = pd.DataFrame(rows)
    raw.loc[0, "title"] = "Inception"  # lets the demo viewer resolve one seed
    return raw


@pytest.fixture
def raw() -> pd.DataFrame:
    return make_raw()
