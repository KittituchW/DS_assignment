"""Structured exploration of the raw files and the clean catalogue.

Writes `reports/eda_summary.md` and four figures under `reports/figures/`.
Run: python scripts/eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import MOVIES_FILE, TV_FILE, build_catalogue, save_catalogue  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "reports"
FIG = OUT / "figures"


def schema_section(movies: pd.DataFrame, tv: pd.DataFrame) -> list[str]:
    lines = ["## 1. Schema and relationships", ""]
    lines.append(f"Movies file: {movies.shape[0]:,} rows x {movies.shape[1]} columns.")
    lines.append(f"TV file: {tv.shape[0]:,} rows x {tv.shape[1]} columns.")
    only_movies = sorted(set(movies.columns) - set(tv.columns))
    lines.append(f"Columns only in movies: {only_movies}.")
    overlap = len(set(movies.show_id) & set(tv.show_id))
    lines.append(f"`show_id` values shared by both files: {overlap} (so `show_id` alone is not a key).")
    lines.append(f"Duplicate `show_id` inside TV file: {tv.show_id.duplicated().sum()}.")
    lines.append(f"`rating` vs `vote_average` correlation (movies): {movies[['rating', 'vote_average']].corr().iloc[0, 1]:.3f} (identical column).")
    lines.append(f"`duration` null in movies: {movies.duration.isna().mean():.0%}; unique values in TV: {tv.duration.nunique()} (constant).")
    lines.append(f"Rows per release year, movies: {sorted(int(v) for v in movies.release_year.value_counts().unique())} (a stratified sample, not the full catalogue).")
    lines.append("")
    lines.append("Relationships: there is no user table and no interaction table. Every row is one title. The only")
    lines.append("cross-row structure comes from shared genres, cast, director, country, and language. `popularity`,")
    lines.append("`vote_count`, and `vote_average` are global TMDb aggregates, not per-user signals.")
    lines.append("")
    lines.append("### Missing values (raw)")
    lines.append("")
    lines.append("| column | movies | tv |")
    lines.append("|---|---|---|")
    for col in ["director", "cast", "country", "genres", "description"]:
        lines.append(f"| {col} | {movies[col].isna().mean():.1%} | {tv[col].isna().mean():.1%} |")
    lines.append(f"| budget == 0 | {(movies.budget == 0).mean():.1%} | n/a |")
    lines.append(f"| revenue == 0 | {(movies.revenue == 0).mean():.1%} | n/a |")
    lines.append(f"| vote_count == 0 | {(movies.vote_count == 0).mean():.1%} | {(tv.vote_count == 0).mean():.1%} |")
    return lines + [""]


def stats_section(cat: pd.DataFrame) -> list[str]:
    lines = ["## 2. Clean catalogue statistics", ""]
    lines.append(f"Items after cleaning: {len(cat):,} ({cat.type.value_counts().to_dict()}).")
    lines.append(f"Recommendable (has genres or description): {cat.recommendable.sum():,}.")
    lines.append(f"Non-English share: {1 - cat.is_english.mean():.1%}.")
    lines.append(f"Median votes: movies {cat[cat.type == 'movie'].vote_count.median():.0f}, tv {cat[cat.type == 'tv'].vote_count.median():.0f}.")
    top1 = cat.popularity.quantile(0.99)
    lines.append(f"Top 1% popularity threshold: {top1:.1f}; median: {cat.popularity.median():.1f} (heavy tail).")
    lines.append("")
    lines.append("### Top genres (harmonised)")
    genres = cat.genres.explode().value_counts().head(10)
    lines.append("")
    lines.append("| genre | items |")
    lines.append("|---|---|")
    lines += [f"| {g} | {n:,} |" for g, n in genres.items()]
    lines.append("")
    lines.append("### Top languages")
    langs = cat.language.value_counts().head(8)
    lines.append("")
    lines.append("| language | items | share |")
    lines.append("|---|---|---|")
    lines += [f"| {l} | {n:,} | {n / len(cat):.1%} |" for l, n in langs.items()]
    lines.append("")
    lines.append("### Observations that shape the design")
    lines.append("")
    lines.append("1. No behaviour data exists. Any recommender must be content based or must simulate users and say so.")
    lines.append("2. Popularity has a heavy tail and skews towards recent titles. A popularity baseline will be narrow.")
    dec = pd.qcut(cat.popularity_pct, 10, labels=False)
    en_low = cat[dec == 0].is_english.mean()
    en_high = cat[dec == 9].is_english.mean()
    mv = cat[cat.type == "movie"]
    mdec = pd.qcut(mv.popularity_pct, 10, labels=False)
    top200 = cat.nlargest(200, "popularity")
    lines.append(
        f"3. English share rises with popularity: {en_low:.0%} in the least popular decile vs {en_high:.0%} in the most popular "
        f"(movies: {mv[mdec == 0].is_english.mean():.0%} to {mv[mdec == 9].is_english.mean():.0%}). "
        f"The raw popularity top 200 is {top200.type.eq('tv').mean():.0%} TV. A popularity recommender will over-serve English and TV titles relative to the catalogue."
    )
    lines.append("4. Text (description) is the richest per-item field. Genre tags are coarse (median 2 per title).")
    lines.append("5. Vote counts are sparse for TV. Ranking by raw vote average would reward titles with 3 votes.")
    return lines + [""]


def figures(cat: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    cat.genres.explode().value_counts().head(12).sort_values().plot.barh(ax=ax, color="#4C72B0")
    ax.set_title("Items per genre (harmonised)")
    ax.set_xlabel("items")
    fig.tight_layout()
    fig.savefig(FIG / "genres.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for t, color in (("movie", "#4C72B0"), ("tv", "#DD8452")):
        cat[cat.type == t].popularity.clip(upper=200).plot.hist(bins=60, alpha=0.6, ax=ax, label=t, color=color)
    ax.set_title("Popularity distribution (clipped at 200)")
    ax.set_xlabel("TMDb popularity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "popularity.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    top = cat.language.value_counts().head(8).index
    share = cat.assign(lang=cat.language.where(cat.language.isin(top), "other"))
    share = share.groupby("lang").is_english.size().sort_values(ascending=False)
    share.plot.bar(ax=ax, color="#55A868")
    ax.set_title("Items per language")
    ax.set_ylabel("items")
    fig.tight_layout()
    fig.savefig(FIG / "languages.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = pd.qcut(cat.popularity_pct, 10, labels=[f"d{i}" for i in range(1, 11)])
    cat.assign(decile=bins).groupby("decile", observed=True).is_english.mean().plot.bar(ax=ax, color="#C44E52")
    ax.set_title("English share by popularity decile (d10 = most popular)")
    ax.set_ylabel("share English")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / "english_share_by_popularity.png", dpi=150)
    plt.close(fig)


def main() -> None:
    movies = pd.read_csv(RAW / MOVIES_FILE)
    tv = pd.read_csv(RAW / TV_FILE)
    cat = build_catalogue(RAW)
    save_catalogue(cat, ROOT / "data" / "processed" / "catalogue.csv")

    lines = ["# Data exploration summary", "", "Generated by `scripts/eda.py`.", ""]
    lines += schema_section(movies, tv)
    lines += stats_section(cat)
    figures(cat)
    lines += ["## 3. Figures", "", "See `reports/figures/`: genres.png, popularity.png, languages.png, english_share_by_popularity.png.", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eda_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
