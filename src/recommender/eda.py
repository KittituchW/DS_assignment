"""Exploration: schema, basic statistics, and the figures behind each design choice."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from recommender.data import CORE_METADATA

DESCRIPTIONS = {
    "show_id": "Source id. Unique per file only; collides across files",
    "type": "Movie or TV Show",
    "title": "Display title",
    "director": "Comma-separated names",
    "cast": "Comma-separated names",
    "country": "Comma-separated production countries",
    "date_added": "Same year as release_year for every row",
    "release_year": "2010 to 2025",
    "rating": "Exact copy of vote_average",
    "duration": "Missing for movies, constant for TV",
    "genres": "Comma-separated; TV and movie vocabularies differ",
    "language": "ISO 639-1 original language",
    "description": "Free-text synopsis",
    "popularity": "TMDb popularity snapshot; scale differs by type",
    "vote_count": "TMDb votes; aggregate, not per viewer",
    "vote_average": "TMDb average score",
    "budget": "Movies only; 0 means unknown",
    "revenue": "Movies only; 0 means unknown",
}


def schema_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per source column: type, missing share by content type, distinct values, meaning."""
    rows = []
    for col in raw.columns:
        series = raw[col]
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "missing_movie_pct": round(float(series[raw["type"].eq("Movie")].isna().mean() * 100), 1),
                "missing_tv_pct": round(float(series[raw["type"].eq("TV Show")].isna().mean() * 100), 1),
                "distinct": int(series.nunique()),
                "meaning": DESCRIPTIONS.get(col, ""),
            }
        )
    return pd.DataFrame(rows)


def english_share_by_popularity(items: pd.DataFrame, bins: int = 10) -> pd.Series:
    decile = pd.qcut(items["popularity_pct"].rank(method="first"), bins, labels=False) + 1
    return items.groupby(decile)["is_english"].mean().rename("english_share")


def eda_summary(raw: pd.DataFrame, catalogue: pd.DataFrame) -> dict:
    movies = raw[raw["type"].eq("Movie")]
    tv = raw[raw["type"].eq("TV Show")]
    items = catalogue[catalogue["recommendable"]]
    shared_ids = set(movies["show_id"]) & set(tv["show_id"])
    genres_per_title = catalogue["genres"].map(len)
    share = english_share_by_popularity(items)
    return {
        "raw_rows": {"movie": int(len(movies)), "tv": int(len(tv))},
        "rows_per_type_and_year": sorted(set(raw.groupby(["type", "release_year"]).size().tolist())),
        "show_id_shared_across_files": len(shared_ids),
        "tv_show_id_repeats": int(tv["show_id"].duplicated().sum()),
        "catalogue_items": int(len(catalogue)),
        "recommendable_items": int(len(items)),
        "rating_equals_vote_average": bool(
            np.allclose(raw["rating"].fillna(-1), raw["vote_average"].fillna(-1))
        ),
        "movie_duration_missing_pct": round(float(movies["duration"].isna().mean() * 100), 1),
        "tv_duration_values": sorted(tv["duration"].dropna().astype(str).unique().tolist()),
        "zero_vote_titles": int(raw["vote_count"].eq(0).sum()),
        "median_genres_per_title": float(genres_per_title.median()),
        "non_english_share_catalogue": round(float(1 - catalogue["is_english"].mean()), 3),
        "non_english_share_recommendable": round(float(1 - items["is_english"].mean()), 3),
        "english_share_least_popular_decile": round(float(share.iloc[0]), 3),
        "english_share_most_popular_decile": round(float(share.iloc[-1]), 3),
        "missing_pct_by_type": {
            kind: {col: round(float(raw.loc[raw["type"].eq(kind), col].isna().mean() * 100), 1) for col in CORE_METADATA}
            for kind in ("Movie", "TV Show")
        },
    }


def save_eda(raw: pd.DataFrame, catalogue: pd.DataFrame, output: Path) -> dict:
    """Write the schema table, summary JSON and three figures. Returns the summary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    schema_table(raw).to_csv(output / "eda_schema.csv", index=False)
    summary = eda_summary(raw, catalogue)
    (output / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    missing = pd.DataFrame(summary["missing_pct_by_type"])
    ax = missing.plot(kind="bar", figsize=(8, 4.5), color=["#1f4e79", "#4a90c2"])
    ax.set(title="TV metadata is much less complete than movie metadata", ylabel="Missing (%)", xlabel="")
    ax.legend(["Movie", "TV"], title="")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output / "eda_missingness.png", dpi=160)
    plt.close()

    items = catalogue[catalogue["recommendable"]]
    share = english_share_by_popularity(items)
    ax = (share * 100).plot(kind="bar", figsize=(8, 4.5), color="#1f4e79")
    ax.axhline(items["is_english"].mean() * 100, color="#d97706", linestyle="--", label="All recommendable titles")
    ax.set(
        title="English share rises with popularity",
        xlabel="Popularity decile (1 = least popular)",
        ylabel="English titles (%)",
    )
    ax.legend()
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output / "eda_english_share_by_popularity.png", dpi=160)
    plt.close()

    top = (catalogue["language"].value_counts(normalize=True).head(12) * 100).sort_values()
    ax = top.plot(kind="barh", figsize=(8, 4.5), color="#1f4e79")
    ax.set(title="Original language of catalogue titles", xlabel="Titles (%)", ylabel="")
    plt.tight_layout()
    plt.savefig(output / "eda_languages.png", dpi=160)
    plt.close()
    return summary
