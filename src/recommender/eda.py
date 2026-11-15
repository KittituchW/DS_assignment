"""Exploration: schema, basic statistics, and one figure per design decision.

Each figure title states the finding. The decision it drives is in the README.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from recommender import plotting
from recommender.data import CORE_METADATA, MIN_VOTES
from recommender.simulate import PERSONAS

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
TYPES = ("movie", "tv")


# ------------------------------------------------------------------ tables and statistics


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


def funnel(raw: pd.DataFrame, catalogue: pd.DataFrame) -> pd.DataFrame:
    """Rows per type after each cleaning step, applied in order."""
    raw_type = raw["type"].map({"Movie": "movie", "TV Show": "tv"})
    keep = pd.Series(True, index=catalogue.index)
    stages = {"raw rows": raw_type.value_counts(), "after de-duplication": catalogue["type"].value_counts()}
    for label, col in (
        ("has genre or description", "excluded_no_content"),
        (f"at least {MIN_VOTES} votes", "excluded_low_votes"),
        ("adult keyword filter", "excluded_adult_keyword"),
    ):
        keep &= ~catalogue[col]
        stages[label] = catalogue.loc[keep, "type"].value_counts()
    return pd.DataFrame(stages).T.reindex(columns=list(TYPES)).fillna(0).astype(int)


def english_share_by_popularity(items: pd.DataFrame, bins: int = 10) -> pd.Series:
    decile = pd.qcut(items["popularity_pct"].rank(method="first"), bins, labels=False) + 1
    return items.groupby(decile)["is_english"].mean().rename("english_share")


def persona_pools(items: pd.DataFrame) -> pd.DataFrame:
    """Recommendable titles carrying each persona's main genre, and its language when it has one."""
    rows = []
    for name, spec in PERSONAS.items():
        genre = max(spec["genres"], key=spec["genres"].get)
        has_genre = items["genres"].map(lambda gs: genre in gs)
        lang = next(iter(spec["language"]), None)
        rows.append(
            {
                "persona": name,
                "main_genre": genre,
                "titles": int(has_genre.sum()),
                "language": lang,
                "titles_in_language": int((has_genre & items["language"].eq(lang)).sum()) if lang else None,
            }
        )
    return pd.DataFrame(rows)


def cast_link_share(items: pd.DataFrame, top: int = 5) -> float:
    """Share of titles that share at least one of their first `top` cast members with another title."""
    by_actor: dict[str, set[str]] = defaultdict(set)
    for item, cast in zip(items["item_id"], items["cast"]):
        for actor in cast[:top]:
            by_actor[actor].add(item)
    linked: set[str] = set()
    for titles in by_actor.values():
        if len(titles) > 1:
            linked |= titles
    return len(linked) / max(len(items), 1)


def eda_summary(raw: pd.DataFrame, catalogue: pd.DataFrame) -> dict:
    movies = raw[raw["type"].eq("Movie")]
    tv = raw[raw["type"].eq("TV Show")]
    items = catalogue[catalogue["recommendable"]]
    voted = catalogue[catalogue["vote_count"] >= MIN_VOTES]
    share = english_share_by_popularity(items)
    non_latin = items["description"].map(lambda s: sum(ch > "\u007f" for ch in s) / max(len(s), 1) > 0.3)
    return {
        "raw_rows": {"movie": int(len(movies)), "tv": int(len(tv))},
        "rows_per_type_and_year": sorted(set(raw.groupby(["type", "release_year"]).size().tolist())),
        "show_id_shared_across_files": len(set(movies["show_id"]) & set(tv["show_id"])),
        "tv_show_id_repeats": int(tv["show_id"].duplicated().sum()),
        "catalogue_items": int(len(catalogue)),
        "recommendable_items": int(len(items)),
        "funnel": funnel(raw, catalogue).to_dict(orient="index"),
        "recommendable_movie_share": round(float(items["type"].eq("movie").mean()), 3),
        "rating_equals_vote_average": bool(np.allclose(raw["rating"].fillna(-1), raw["vote_average"].fillna(-1))),
        "movie_duration_missing_pct": round(float(movies["duration"].isna().mean() * 100), 1),
        "tv_duration_values": sorted(tv["duration"].dropna().astype(str).unique().tolist()),
        "votes_by_type": {
            t: {
                "zero_share": round(float(g["vote_count"].eq(0).mean()), 3),
                "below_min_votes_share": round(float(g["vote_count"].lt(MIN_VOTES).mean()), 3),
                "median": float(g["vote_count"].median()),
            }
            for t, g in catalogue.groupby("type")
        },
        "mean_vote_average_by_type_min_votes": {
            t: round(float(g["vote_average"].mean()), 2) for t, g in voted.groupby("type")
        },
        "median_genres_per_title": float(catalogue["genres"].map(len).median()),
        "non_english_share_catalogue": round(float(1 - catalogue["is_english"].mean()), 3),
        "non_english_share_recommendable": round(float(1 - items["is_english"].mean()), 3),
        "non_english_share_by_type_recommendable": {
            t: round(float(1 - g["is_english"].mean()), 3) for t, g in items.groupby("type")
        },
        "english_share_least_popular_decile": round(float(share.iloc[0]), 3),
        "english_share_most_popular_decile": round(float(share.iloc[-1]), 3),
        "popularity_pct_median_by_year": {
            int(y): round(float(v), 2) for y, v in items.groupby("release_year")["popularity_pct"].median().items()
        },
        "description_mostly_non_latin_share": round(float(non_latin.mean()), 3),
        "cast_link_share": round(cast_link_share(items), 3),
        "persona_pools": persona_pools(items).to_dict(orient="records"),
        "missing_pct_by_type": {
            kind: {col: round(float(raw.loc[raw["type"].eq(kind), col].isna().mean() * 100), 1) for col in CORE_METADATA}
            for kind in ("Movie", "TV Show")
        },
    }


# ------------------------------------------------------------------ figures


def _grouped_bars(ax, frame: pd.DataFrame, fmt: str = "{:.0f}") -> None:
    """Bars for each column of `frame` (movie, tv) grouped by its index."""
    x = np.arange(len(frame))
    width = 0.38
    for i, col in enumerate(frame.columns):
        bars = ax.bar(x + (i - 0.5) * width, frame[col], width, color=plotting.TYPE_COLOURS[col], label=col.upper() if col == "tv" else "Movie", **plotting.BAR)
        plotting.label_bars(ax, bars, fmt)
    ax.set_xticks(x, frame.index)
    ax.legend()


def save_eda(raw: pd.DataFrame, catalogue: pd.DataFrame, output: Path) -> dict:
    """Write the schema table, summary JSON and the EDA figures. Returns the summary."""
    import matplotlib.pyplot as plt

    plotting.apply_style()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    schema_table(raw).to_csv(output / "eda_schema.csv", index=False)
    summary = eda_summary(raw, catalogue)
    (output / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    items = catalogue[catalogue["recommendable"]]

    # 1. Missing metadata by type
    missing = pd.DataFrame(summary["missing_pct_by_type"]).rename(columns={"Movie": "movie", "TV Show": "tv"})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _grouped_bars(ax, missing)
    ax.set(title="TV metadata is much less complete than movie metadata", ylabel="Missing (%)")
    plotting.save(fig, output / "eda_missingness.png")

    # 2. Cleaning funnel
    steps = funnel(raw, catalogue)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _grouped_bars(ax, steps)
    ax.set(title="The vote filter removes half of TV; recommendable titles are mostly movies", ylabel="Titles")
    ax.tick_params(axis="x", labelsize=8)
    plotting.save(fig, output / "eda_cleaning_funnel.png")

    # 3. Vote counts by type
    bins = np.logspace(0, 5.5, 34)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for t in TYPES:
        votes = catalogue.loc[catalogue["type"].eq(t), "vote_count"] + 1
        ax.hist(votes, bins=bins, histtype="step", linewidth=2, color=plotting.TYPE_COLOURS[t], label="TV" if t == "tv" else "Movie")
    ax.axvline(MIN_VOTES + 1, color=plotting.INK_SECONDARY, linewidth=1)
    ax.annotate(f"minimum {MIN_VOTES} votes", (MIN_VOTES + 1, ax.get_ylim()[1] * 0.92), xytext=(5, 0), textcoords="offset points", fontsize=8, color=plotting.INK_SECONDARY)
    ax.set_xscale("log")
    tv_below = summary["votes_by_type"]["tv"]["below_min_votes_share"]
    ax.set(title=f"{tv_below:.0%} of TV titles have fewer than {MIN_VOTES} votes", xlabel="Vote count + 1 (log scale)", ylabel="Titles")
    ax.legend()
    plotting.save(fig, output / "eda_vote_counts.png")

    # 4. Vote average by type
    voted = catalogue[catalogue["vote_count"] >= MIN_VOTES]
    means = summary["mean_vote_average_by_type_min_votes"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for t in TYPES:
        ax.hist(voted.loc[voted["type"].eq(t), "vote_average"], bins=np.arange(0, 10.25, 0.25), histtype="step", linewidth=2, color=plotting.TYPE_COLOURS[t], label=f"{'TV' if t == 'tv' else 'Movie'} (mean {means[t]:.1f})")
        ax.axvline(means[t], color=plotting.TYPE_COLOURS[t], linewidth=1)
    ax.set(title="TV titles score higher than movies, so rating-based popularity favours TV", xlabel=f"Vote average (titles with at least {MIN_VOTES} votes)", ylabel="Titles")
    ax.legend()
    plotting.save(fig, output / "eda_vote_average_by_type.png")

    # 5. Language by type
    top = items["language"].value_counts().head(8).index
    lang = (pd.crosstab(items["language"], items["type"], normalize="columns").loc[top] * 100).reindex(columns=list(TYPES))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _grouped_bars(ax, lang)
    nen = summary["non_english_share_by_type_recommendable"]
    ax.set(title=f"Non-English: {nen['tv']:.0%} of TV, {nen['movie']:.0%} of movies (recommendable titles)", ylabel="Titles of that type (%)")
    plotting.save(fig, output / "eda_language_by_type.png")

    # 6. English share by popularity decile
    share = english_share_by_popularity(items) * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(share.index, share.values, 0.6, color=plotting.SERIES[0], **plotting.BAR)
    plotting.label_bars(ax, bars)
    ax.axhline(items["is_english"].mean() * 100, color=plotting.INK_SECONDARY, linewidth=1)
    ax.annotate("all recommendable titles", (0.6, items["is_english"].mean() * 100), xytext=(0, 4), textcoords="offset points", fontsize=8, color=plotting.INK_SECONDARY)
    ax.set_xticks(share.index)
    ax.set(title="English share rises with popularity", xlabel="Popularity decile (1 = least popular)", ylabel="English titles (%)")
    plotting.save(fig, output / "eda_english_share_by_popularity.png")

    # 7. Popularity by release year
    by_year = items.groupby(["release_year", "type"])["popularity_pct"].median().unstack()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for t in TYPES:
        ax.plot(by_year.index, by_year[t], color=plotting.TYPE_COLOURS[t], linewidth=2, marker="o", markersize=5, label="TV" if t == "tv" else "Movie")
    ax.set(title="Popularity favours recent movies; TV is flatter until 2025", xlabel="Release year", ylabel="Median popularity percentile")
    ax.legend()
    plotting.save(fig, output / "eda_popularity_by_year.png")

    # 8. Persona pool sizes
    pools = persona_pools(items).set_index("persona").sort_values("titles")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(pools))
    bars = ax.barh(y, pools["titles"], 0.6, color=plotting.SERIES[0], label="titles with the main genre", **plotting.BAR)
    with_lang = pools["titles_in_language"].fillna(0)
    ax.barh(y[with_lang > 0], with_lang[with_lang > 0], 0.6, color=plotting.SERIES[1], label="of which in the persona's language", **plotting.BAR)
    for bar, (_, row) in zip(bars, pools.iterrows()):
        text = f"{row['titles']:,}"
        if isinstance(row["language"], str):
            text += f" ({int(row['titles_in_language']):,} in '{row['language']}')"
        ax.annotate(text, (bar.get_width(), bar.get_y() + bar.get_height() / 2), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8, color=plotting.INK_SECONDARY)
    ax.set_yticks(y, [f"{p} ({g})" for p, g in zip(pools.index, pools["main_genre"])])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set(title="Simulated personas draw from very different pool sizes", xlabel="Recommendable titles")
    ax.set_xlim(0, pools["titles"].max() * 1.22)
    ax.legend(loc="lower right")
    plotting.save(fig, output / "eda_persona_pools.png")

    # 9. Languages overall
    top_all = (catalogue["language"].value_counts(normalize=True).head(12) * 100).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(top_all.index, top_all.values, 0.6, color=plotting.SERIES[0], **plotting.BAR)
    for bar in bars:
        ax.annotate(f"{bar.get_width():.1f}", (bar.get_width(), bar.get_y() + bar.get_height() / 2), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8, color=plotting.INK_SECONDARY)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set(title="Original language of catalogue titles", xlabel="Titles (%)")
    plotting.save(fig, output / "eda_languages.png")
    return summary
