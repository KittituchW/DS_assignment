"""Data loading and cleaning.

Turns the two raw TMDb CSV files into one clean catalogue table.
Every cleaning decision is listed in the module docstring of `build_catalogue`
and mirrored in the README so a reviewer can audit it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MOVIES_FILE = "netflix_movies_detailed_up_to_2025.csv"
TV_FILE = "netflix_tv_shows_detailed_up_to_2025.csv"

# TV genre labels that do not exist in the movie vocabulary.
# They are mapped onto the movie vocabulary so both types share one genre space.
GENRE_MAP: dict[str, list[str]] = {
    "Action & Adventure": ["Action", "Adventure"],
    "Sci-Fi & Fantasy": ["Science Fiction", "Fantasy"],
    "War & Politics": ["War", "Politics"],
    "Kids": ["Family"],
    "TV Movie": [],
    "Unknown": [],
}

# A title with fewer votes than this has no evidence of ever being watched.
MIN_VOTES = 5

# The dataset has no maturity rating. A cluster of low-vote adult softcore titles
# would otherwise dominate Romance-persona histories. This is a bounded keyword
# filter on titles only, and it is known to be incomplete. Production needs a
# licensed maturity feed, not keywords.
ADULT_TITLE_KEYWORDS = [
    "sister-in-law", "mom's friend", "housemate's wife", "older sister", "seduction",
    "an affair", "secretary wants", "taste of a", "sex education for", "young mother",
    "stepmom", "step mom", "erotic", "female hostel", "boarding house", "temptation",
    "porn", "aunt's", "mom's", "russian mom", "young mom", "friend's mom", "wife's",
    "nude", "sexy", "affair", "swapping", "massage", "delivery girl", "hot girl",
    r"\bsex\b", "desire", "bosomy", "x-rated", "clitoris", "nasty", "adult", "lesson",
    "neighbor's wife", "neighbour's wife", "lust", "pleasure", "virgin", "lover",
]


def load_raw(data_dir: Path) -> pd.DataFrame:
    """Load both files and union them with a `type` column already present."""
    movies = pd.read_csv(data_dir / MOVIES_FILE)
    tv = pd.read_csv(data_dir / TV_FILE)
    # Movies carry budget and revenue, TV does not. Align columns before the union.
    for col in ("budget", "revenue"):
        if col not in tv.columns:
            tv[col] = np.nan
    return pd.concat([movies, tv], ignore_index=True)


def _split_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def harmonise_genres(value: object) -> list[str]:
    out: list[str] = []
    for genre in _split_list(value):
        out.extend(GENRE_MAP.get(genre, [genre]))
    return sorted(set(out))


def bayesian_rating(vote_average: pd.Series, vote_count: pd.Series, prior_votes: int = 50) -> pd.Series:
    """Shrink each title's score towards the global mean when it has few votes.

    Raw TMDb averages are noisy below a few dozen votes. Using a hard vote floor
    would drop most TV and recent titles, so we shrink instead of filter.
    """
    rated = vote_count > 0
    global_mean = float(vote_average[rated].mean())
    return (vote_count * vote_average + prior_votes * global_mean) / (vote_count + prior_votes)


def build_catalogue(data_dir: Path) -> pd.DataFrame:
    """Return one clean item table.

    Cleaning decisions:
    1. `show_id` collides across the two files (397 ids appear in both), so the
       item key is `movie_<id>` / `tv_<id>`.
    2. Nine TV `show_id`s are duplicated. Keep the row with the highest popularity.
    3. Titles duplicated within type and release year are collapsed the same way.
    4. `rating` is an exact copy of `vote_average`; it is dropped.
    5. `duration` is null for every movie and "1 Seasons" for every TV show; dropped.
    6. Budget and revenue of 0 mean "unknown", not zero. They become null and a
       `has_financials` flag is kept.
    7. TV genre labels are mapped onto the movie vocabulary (see GENRE_MAP).
    8. Items with no genres and no description carry no usable content signal,
       items with fewer than MIN_VOTES votes have no evidence of being watched,
       and a bounded keyword list removes an adult softcore cluster (the data
       has no maturity rating). These stay in the table flagged
       `recommendable = False`.
    9. Popularity is converted to a percentile within type, because TV and movie
       popularity scales differ.
    10. A Bayesian weighted rating replaces the raw vote average for ranking.
    """
    raw = load_raw(data_dir)
    df = raw.copy()

    df["type"] = df["type"].map({"Movie": "movie", "TV Show": "tv"})
    df["item_id"] = df["type"] + "_" + df["show_id"].astype(str)

    # 2 and 3: duplicates
    df = df.sort_values("popularity", ascending=False)
    df = df.drop_duplicates("item_id", keep="first")
    df["_title_key"] = df["title"].str.lower().str.strip()
    df = df.drop_duplicates(["type", "_title_key", "release_year"], keep="first")
    df = df.drop(columns="_title_key")

    # 4 and 5: redundant columns
    df = df.drop(columns=["rating", "duration"])

    # 6: zero financials mean unknown
    for col in ("budget", "revenue"):
        df[col] = df[col].where(df[col] > 0, np.nan)
    df["has_financials"] = df["budget"].notna() & df["revenue"].notna()

    # 7: list columns
    df["genres"] = df["genres"].map(harmonise_genres)
    df["cast"] = df["cast"].map(_split_list)
    df["country"] = df["country"].map(_split_list)
    df["director"] = df["director"].fillna("").astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df["language"] = df["language"].fillna("unknown").astype(str)

    # 8: recommendable flag
    has_content = (df["genres"].str.len() > 0) | (df["description"].str.len() > 0)
    has_votes = df["vote_count"] >= MIN_VOTES
    pattern = "|".join(ADULT_TITLE_KEYWORDS)
    adult_flag = df["title"].str.lower().str.contains(pattern, regex=True, na=False)
    df["recommendable"] = has_content & has_votes & ~adult_flag

    # 9 and 10: ranking signals
    df["popularity_pct"] = df.groupby("type")["popularity"].rank(pct=True)
    df["weighted_rating"] = bayesian_rating(df["vote_average"], df["vote_count"])

    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["is_english"] = df["language"].eq("en")

    cols = [
        "item_id", "type", "show_id", "title", "director", "cast", "country", "language",
        "is_english", "date_added", "release_year", "genres", "description", "popularity",
        "popularity_pct", "vote_count", "vote_average", "weighted_rating", "budget", "revenue",
        "has_financials", "recommendable",
    ]
    return df[cols].sort_values("item_id").reset_index(drop=True)


def save_catalogue(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in ("genres", "cast", "country"):
        out[col] = out[col].map("|".join)
    out.to_csv(path, index=False)


def load_catalogue(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date_added"])
    for col in ("genres", "cast", "country"):
        df[col] = df[col].fillna("").map(lambda s: [p for p in s.split("|") if p])
    df["description"] = df["description"].fillna("")
    df["director"] = df["director"].fillna("")
    return df
