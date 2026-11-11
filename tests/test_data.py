from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommender.data import canonicalize_genres


def test_genre_taxonomy_is_canonicalized() -> None:
    assert canonicalize_genres("Action & Adventure, Sci-Fi & Fantasy") == [
        "Action",
        "Adventure",
        "Science Fiction",
        "Fantasy",
    ]


def test_missing_genres_are_empty() -> None:
    assert canonicalize_genres(pd.NA) == []
