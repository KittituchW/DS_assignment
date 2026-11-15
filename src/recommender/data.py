"""Load the two catalogue files and clean them into one item table.

Data processing only. This module never imports a modelling library.

Cleaning decisions (each one comes from an EDA finding, see `eda.py`):

1. `show_id` collides across the movie and TV files, so the item key is
   `movie_<id>` or `tv_<id>`.
2. Some TV `show_id`s repeat. Keep the row with the most complete metadata,
   then the most popular row.
3. Titles repeated within the same type and release year are collapsed the
   same way.
4. `rating` is an exact copy of `vote_average`, and `duration` has no useful
   variation. Both are dropped.
5. Budget and revenue of 0 mean "unknown". They become missing values.
6. TV genre labels are mapped onto the movie genre vocabulary.
7. Titles with no genre and no description, titles with fewer than
   `MIN_VOTES` votes, and low-vote titles that match an adult keyword list are
   kept in the table but flagged `recommendable = False`, with one column per
   reason. Keywords match whole words only, and only titles with fewer than
   `ADULT_MAX_VOTES` votes are checked: EDA showed that substring matching
   removed mainstream titles ("lover" inside "10 Cloverfield Lane", "lust"
   inside "Wanderlust"), while the adult cluster sits in low-vote titles. The
   data has no maturity rating, so this filter is known to be incomplete.
8. Line breaks and repeated spaces inside text fields become single spaces.
   Some descriptions contain carriage returns, which break CSV round trips
   in pandas 2.
9. Popularity becomes a percentile within type, because movie and TV
   popularity use different scales.
10. A Bayesian weighted rating shrinks low-vote averages towards the mean.
"""

from __future__ import annotations

import hashlib
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

MOVIES_FILE = "netflix_movies_detailed_up_to_2025.csv"
TV_FILE = "netflix_tv_shows_detailed_up_to_2025.csv"

# Pinned source: the commit and file hashes the reported results were built on.
SOURCE_COMMIT = "7e74398ef4d9fec262bd7f64573d4d2692db2ead"
SOURCE_URL = f"https://raw.githubusercontent.com/pimphornm-pim/DSAssignmentDataSet/{SOURCE_COMMIT}/"
EXPECTED_SHA256 = {
    MOVIES_FILE: "5439bcc05ccc4c170265a88e115891adecef64a330060dbd55266ac7fea14613",
    TV_FILE: "16156fb21d0139a396b3a72cd58d66f86d272f7607df9281c2abfe68f59f80d7",
}

CORE_METADATA = ["director", "cast", "country", "genres", "description"]

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

# Bounded keyword list for a low-vote adult cluster. Not a maturity system.
# Titles with at least ADULT_MAX_VOTES votes are treated as mainstream and kept.
ADULT_MAX_VOTES = 100
ADULT_TITLE_KEYWORDS = [
    "sister-in-law", "mom's friend", "housemate's wife", "older sister", "seduction",
    "an affair", "secretary wants", "taste of a", "sex education for", "young mother",
    "stepmom", "step mom", "erotic", "female hostel", "boarding house", "temptation",
    "porn", "aunt's", "mom's", "russian mom", "young mom", "friend's mom", "wife's",
    "nude", "sexy", "affair", "swapping", "massage", "delivery girl", "hot girl",
    r"\bsex\b", "desire", "bosomy", "x-rated", "clitoris", "nasty", "adult", "lesson",
    "neighbor's wife", "neighbour's wife", "lust", "pleasure", "virgin", "lover",
]
ADULT_PATTERN = "|".join(k if k.startswith(r"\b") else rf"\b{re.escape(k)}\b" for k in ADULT_TITLE_KEYWORDS)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_raw(data_dir: Path) -> None:
    """Download any missing input file from the pinned commit."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in (MOVIES_FILE, TV_FILE):
        target = data_dir / name
        if not target.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(SOURCE_URL + name, target)


def check_inputs(data_dir: Path) -> dict[str, str]:
    """Return the SHA-256 of each input and warn when it differs from the pinned file."""
    hashes = {}
    for name in (MOVIES_FILE, TV_FILE):
        path = Path(data_dir) / name
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Pass --download or copy the file there.")
        hashes[name] = file_sha256(path)
        if hashes[name] != EXPECTED_SHA256[name]:
            print(f"warning: {name} differs from the pinned version; results may not match RESULTS.md")
    return hashes


def load_raw(data_dir: Path) -> pd.DataFrame:
    """Load both files and union them. Movie-only columns become missing for TV."""
    movies = pd.read_csv(Path(data_dir) / MOVIES_FILE)
    tv = pd.read_csv(Path(data_dir) / TV_FILE)
    return pd.concat([movies, tv], ignore_index=True, sort=False)


def split_list(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def clean_text(values: pd.Series) -> pd.Series:
    """Missing becomes empty; any run of whitespace (including line breaks) becomes one space."""
    return values.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def harmonise_genres(value: object) -> list[str]:
    out: list[str] = []
    for genre in split_list(value):
        out.extend(GENRE_MAP.get(genre, [genre]))
    return sorted(set(out))


def bayesian_rating(vote_average: pd.Series, vote_count: pd.Series, prior_votes: int = 50) -> pd.Series:
    """Shrink each average towards the global mean when it has few votes."""
    rated = vote_count > 0
    global_mean = float(vote_average[rated].mean()) if rated.any() else 0.0
    average = vote_average.fillna(global_mean)
    return (vote_count * average + prior_votes * global_mean) / (vote_count + prior_votes)


def build_catalogue(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one clean item table. See the module docstring for each rule."""
    df = raw.copy()
    df["type"] = df["type"].map({"Movie": "movie", "TV Show": "tv"})
    df["item_id"] = df["type"] + "_" + df["show_id"].astype(str)
    df["metadata_completeness"] = df[CORE_METADATA].notna().sum(axis=1)

    # 2 and 3: duplicates, most complete row first, then most popular.
    df = df.sort_values(
        ["metadata_completeness", "popularity"], ascending=[False, False], kind="stable"
    )
    df = df.drop_duplicates("item_id", keep="first")
    title_key = df["title"].fillna("").astype(str).str.lower().str.strip()
    df = df.loc[~pd.DataFrame({"t": df["type"], "k": title_key, "y": df["release_year"]}).duplicated()]

    # 4 and 5
    df = df.drop(columns=[c for c in ("rating", "duration") if c in df.columns])
    for col in ("budget", "revenue"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].where(df[col] > 0, np.nan)
    df["has_financials"] = df["budget"].notna() & df["revenue"].notna()

    # 6: list columns
    df["genres"] = df["genres"].map(harmonise_genres)
    df["cast"] = df["cast"].map(split_list)
    df["country"] = df["country"].map(split_list)
    for col in ("title", "director", "description"):
        df[col] = clean_text(df[col])
    df["language"] = df["language"].fillna("unknown").astype(str)

    # 7: recommendable flag, one column per reason
    df["vote_count"] = df["vote_count"].fillna(0).astype(int)
    df["excluded_no_content"] = df["genres"].map(len).eq(0) & df["description"].str.len().eq(0)
    df["excluded_low_votes"] = df["vote_count"] < MIN_VOTES
    keyword = df["title"].str.lower().str.contains(ADULT_PATTERN, regex=True)
    df["excluded_adult_keyword"] = keyword & (df["vote_count"] < ADULT_MAX_VOTES)
    df["recommendable"] = ~(df["excluded_no_content"] | df["excluded_low_votes"] | df["excluded_adult_keyword"])

    # 9 and 10
    df["popularity"] = df["popularity"].fillna(0.0)
    df["popularity_pct"] = df.groupby("type")["popularity"].rank(pct=True)
    df["weighted_rating"] = bayesian_rating(df["vote_average"], df["vote_count"])
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["is_english"] = df["language"].eq("en")

    cols = [
        "item_id", "type", "show_id", "title", "director", "cast", "country", "language",
        "is_english", "date_added", "release_year", "genres", "description", "popularity",
        "popularity_pct", "vote_count", "vote_average", "weighted_rating", "budget", "revenue",
        "has_financials", "metadata_completeness", "excluded_no_content", "excluded_low_votes",
        "excluded_adult_keyword", "recommendable",
    ]
    return df[cols].sort_values("item_id").reset_index(drop=True)


def recommendable_items(catalogue: pd.DataFrame) -> pd.DataFrame:
    """The items a model may recommend, in a fixed order shared by every stage."""
    return catalogue[catalogue["recommendable"]].sort_values("item_id").reset_index(drop=True)


def save_catalogue(catalogue: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = catalogue.copy()
    for col in ("genres", "cast", "country"):
        out[col] = out[col].map("|".join)
    out.to_csv(path, index=False)


def load_catalogue(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date_added"])
    for col in ("genres", "cast", "country"):
        df[col] = df[col].fillna("").astype(str).map(lambda s: [p for p in s.split("|") if p])
    for col in ("description", "director", "title"):
        df[col] = df[col].fillna("").astype(str)
    return df
