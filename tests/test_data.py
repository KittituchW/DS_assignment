from __future__ import annotations

import numpy as np
import pandas as pd

from recommender.data import build_catalogue, harmonise_genres, load_catalogue, save_catalogue


def test_tv_genres_are_mapped_to_the_movie_vocabulary() -> None:
    assert harmonise_genres("Action & Adventure, Sci-Fi & Fantasy, Kids") == [
        "Action", "Adventure", "Family", "Fantasy", "Science Fiction",
    ]
    assert harmonise_genres(np.nan) == []


def test_ids_are_namespaced_by_type(raw: pd.DataFrame) -> None:
    cat = build_catalogue(raw)
    assert {"movie_1000", "tv_1000"} <= set(cat["item_id"])
    assert cat["item_id"].is_unique


def test_duplicate_id_keeps_the_most_complete_row(raw: pd.DataFrame) -> None:
    sparse_copy = raw.iloc[[61]].copy()  # a TV row
    sparse_copy[["director", "cast", "country", "description"]] = np.nan
    sparse_copy["title"] = "Sparse duplicate"
    cat = build_catalogue(pd.concat([sparse_copy, raw], ignore_index=True))
    row = cat.set_index("item_id").loc[f"tv_{raw.loc[61, 'show_id']}"]
    assert row["title"] == raw.loc[61, "title"]


def test_recommendable_rules(raw: pd.DataFrame) -> None:
    raw = raw.copy()
    raw.loc[1, "vote_count"] = 2  # too few votes
    raw.loc[2, "title"] = "Erotic Story"  # adult keyword
    raw.loc[3, ["genres", "description"]] = np.nan  # no content
    cat = build_catalogue(raw).set_index("item_id")
    for i in (1, 2, 3):
        assert not cat.loc[f"movie_{raw.loc[i, 'show_id']}", "recommendable"]
    assert cat.loc[f"movie_{raw.loc[4, 'show_id']}", "recommendable"]


def test_zero_financials_become_missing_and_redundant_columns_are_dropped(raw: pd.DataFrame) -> None:
    cat = build_catalogue(raw)
    assert cat["budget"].isna().all()
    assert "rating" not in cat.columns and "duration" not in cat.columns


def test_popularity_percentile_is_within_type(raw: pd.DataFrame) -> None:
    cat = build_catalogue(raw)
    for _, group in cat.groupby("type"):
        assert group["popularity_pct"].max() == 1.0
        assert group["popularity_pct"].min() > 0.0


def test_catalogue_round_trip_keeps_list_columns(raw: pd.DataFrame, tmp_path) -> None:
    cat = build_catalogue(raw)
    save_catalogue(cat, tmp_path / "catalogue.csv")
    back = load_catalogue(tmp_path / "catalogue.csv")
    assert back["genres"].tolist() == cat["genres"].tolist()
    assert back["cast"].tolist() == cat["cast"].tolist()
