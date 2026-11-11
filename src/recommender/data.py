from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd


CORE_METADATA = ["director", "cast", "country", "genres", "description"]

GENRE_EXPANSIONS = {
    "action & adventure": ["Action", "Adventure"],
    "sci-fi & fantasy": ["Science Fiction", "Fantasy"],
}


def _split_multivalue(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def canonicalize_genres(value: object) -> list[str]:
    canonical: list[str] = []
    for raw in _split_multivalue(value):
        expanded = GENRE_EXPANSIONS.get(raw.casefold(), [raw.strip().title()])
        canonical.extend(expanded)
    return list(dict.fromkeys(canonical))


def _token(value: str) -> str:
    value = value.casefold().strip()
    value = re.sub(r"[^\w]+", "_", value, flags=re.UNICODE)
    return value.strip("_")


def _prefixed_tokens(prefix: str, values: Iterable[str], limit: int | None = None) -> str:
    selected = list(values)[:limit]
    return " ".join(f"{prefix}_{_token(value)}" for value in selected if _token(value))


def _feature_document(row: pd.Series) -> str:
    genres = row["genre_list"]
    cast = _split_multivalue(row.get("cast"))
    directors = _split_multivalue(row.get("director"))
    countries = _split_multivalue(row.get("country"))
    title = str(row.get("title", ""))
    description = "" if pd.isna(row.get("description")) else str(row.get("description"))
    tokens = [
        title,
        title,
        description,
        _prefixed_tokens("genre", genres),
        _prefixed_tokens("cast", cast, limit=5),
        _prefixed_tokens("director", directors, limit=2),
        _prefixed_tokens("country", countries),
        f"language_{_token(str(row.get('language', 'unknown')))}",
        f"type_{_token(str(row.get('type', 'unknown')))}",
    ]
    return " ".join(part for part in tokens if part).strip()


def load_and_clean_catalog(movies_path: str, tv_path: str) -> pd.DataFrame:
    movies = pd.read_csv(movies_path)
    tv = pd.read_csv(tv_path)
    catalog = pd.concat([movies, tv], ignore_index=True, sort=False)

    catalog["item_id"] = np.where(catalog["type"].eq("Movie"), "M:", "T:") + catalog[
        "show_id"
    ].astype(str)
    catalog["metadata_completeness"] = catalog[CORE_METADATA].notna().sum(axis=1)

    # Keep the most complete representation for duplicate IDs. Stable sorting makes
    # this deterministic when completeness is tied.
    catalog = (
        catalog.sort_values(
            ["item_id", "metadata_completeness"], ascending=[True, False], kind="stable"
        )
        .drop_duplicates("item_id", keep="first")
        .reset_index(drop=True)
    )

    catalog["genre_list"] = catalog["genres"].map(canonicalize_genres)
    catalog["primary_genre"] = catalog["genre_list"].map(
        lambda genres: genres[0] if genres else "Unknown"
    )
    catalog["language"] = catalog["language"].fillna("unknown").astype(str)
    catalog["description_missing"] = catalog["description"].isna().astype("int8")
    catalog["genre_missing"] = catalog["genre_list"].map(lambda values: int(not values)).astype(
        "int8"
    )
    catalog["rating_observed"] = catalog["rating"].where(catalog["vote_count"].gt(0))
    catalog["log_popularity"] = np.log1p(catalog["popularity"].clip(lower=0))
    catalog["feature_document"] = catalog.apply(_feature_document, axis=1)

    # A confidence-adjusted quality measure. It is used only by the popularity
    # baseline and optional reranking, never as temporal ground truth.
    observed = catalog["rating_observed"].dropna()
    global_mean = float(observed.mean()) if len(observed) else 0.0
    prior_votes = float(catalog.loc[catalog["vote_count"].gt(0), "vote_count"].quantile(0.75))
    votes = catalog["vote_count"].clip(lower=0).astype(float)
    rating = catalog["rating_observed"].fillna(global_mean)
    catalog["bayesian_rating"] = (votes / (votes + prior_votes)) * rating + (
        prior_votes / (votes + prior_votes)
    ) * global_mean
    return catalog


def build_eda_summary(catalog: pd.DataFrame) -> dict:
    movies = catalog[catalog["type"].eq("Movie")]
    tv = catalog[catalog["type"].eq("TV Show")]
    year_counts = (
        catalog.groupby(["release_year", "type"]).size().unstack(fill_value=0).to_dict()
    )
    missing = (
        catalog.groupby("type")[CORE_METADATA].apply(lambda frame: frame.isna().mean() * 100)
    )
    return {
        "rows_after_deduplication": int(len(catalog)),
        "movies": int(len(movies)),
        "tv_shows": int(len(tv)),
        "year_range": [int(catalog.release_year.min()), int(catalog.release_year.max())],
        "year_counts_by_type": {
            key: {str(year): int(value) for year, value in values.items()}
            for key, values in year_counts.items()
        },
        "missing_pct_by_type": {
            content_type: {column: round(float(value), 2) for column, value in row.items()}
            for content_type, row in missing.iterrows()
        },
        "zero_vote_items": int(catalog.vote_count.eq(0).sum()),
        "low_vote_items_lt_20": int(catalog.vote_count.lt(20).sum()),
        "rating_duplicates_vote_average": bool(
            np.allclose(catalog.rating.fillna(-1), catalog.vote_average.fillna(-1))
        ),
        "movie_duration_missing_pct": round(float(movies.duration.isna().mean() * 100), 2),
        "tv_duration_unique_values": sorted(tv.duration.dropna().astype(str).unique().tolist()),
        "date_year_equals_release_year_pct": round(
            float(
                (
                    pd.to_datetime(catalog.date_added, errors="coerce").dt.year
                    == catalog.release_year
                ).mean()
                * 100
            ),
            2,
        ),
    }
