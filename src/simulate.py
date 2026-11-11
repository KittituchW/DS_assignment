"""Synthetic user interactions.

The catalogue has no users. To train and evaluate a recommender at all, we
simulate viewing events from an explicit, documented choice model. The model
is a research instrument, not a claim about real viewers, and its rules are
kept simple so every result can be traced back to a rule.

Choice model for one user:
    p(item) ∝ (genre_affinity + eps)^2        persona-driven genre taste
              * (text_similarity_to_anchor + eps)  a latent "taste anchor" title
              * popularity_pct ** pop_exponent    how herd-driven the user is
              * language_boost                   persona-driven language taste

`pop_exponent` is the bias dial used by hypothesis 2. Higher values make
behaviour more popularity driven, which is what a real platform with strong
"Top 10" surfaces looks like.

The text term uses sparse TF-IDF cosine to one anchor title per user. It is a
different text representation from the SVD features the model sees, so the
model cannot simply read the rule back out of its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PERSONAS: dict[str, dict] = {
    "action": {"genres": {"Action": 5, "Adventure": 3, "Thriller": 2}, "language": {}},
    "kdrama": {"genres": {"Romance": 4, "Drama": 3, "Comedy": 1}, "language": {"ko": 4}},
    "anime": {"genres": {"Animation": 5, "Fantasy": 2, "Action": 1}, "language": {"ja": 4}},
    "family": {"genres": {"Family": 5, "Animation": 3, "Comedy": 2}, "language": {}},
    "horror": {"genres": {"Horror": 6, "Thriller": 3, "Mystery": 2}, "language": {}},
    "documentary": {"genres": {"Documentary": 6, "History": 2}, "language": {}},
    "scifi": {"genres": {"Science Fiction": 5, "Fantasy": 3, "Adventure": 2}, "language": {}},
    "crime": {"genres": {"Crime": 5, "Thriller": 3, "Mystery": 3}, "language": {}},
    "comedy": {"genres": {"Comedy": 6, "Reality": 2, "Talk": 1}, "language": {}},
    "cdrama": {"genres": {"Drama": 5, "Romance": 2, "Fantasy": 1}, "language": {"zh": 4}},
}

# Three fixed demo users, used for the sample recommendations section.
# Their histories are hand picked, well known titles so a reader can judge the
# recommendations without knowing the simulator. They are excluded from metrics.
DEMO_USERS: dict[str, dict] = {
    "user_A": {
        "persona": "action",
        "titles": ["Inception", "Mad Max: Fury Road", "John Wick", "Extraction", "Edge of Tomorrow", "The Dark Knight"],
    },
    "user_B": {
        "persona": "kdrama",
        "titles": ["Crash Landing on You", "Squid Game", "Itaewon Class", "Vincenzo", "Parasite", "Our Beloved Summer"],
    },
    "user_C": {
        "persona": "documentary",
        "titles": ["Our Planet", "The Social Dilemma", "Chef's Table", "Making a Murderer", "My Octopus Teacher", "Free Solo"],
    },
}


def demo_histories(catalogue: pd.DataFrame) -> pd.DataFrame:
    """Resolve the demo users' seed titles to item ids (most voted match per title)."""
    rows = []
    cat = catalogue[catalogue.recommendable]
    for user, spec in DEMO_USERS.items():
        for title in spec["titles"]:
            match = cat[cat.title.str.lower() == title.lower()].sort_values("vote_count", ascending=False)
            if len(match):
                rows.append((user, match.item_id.iloc[0], spec["persona"], pd.Timestamp("2025-06-01")))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "persona", "ts"])


@dataclass
class SimConfig:
    users: int = 3000
    pop_exponent: float = 6.0
    seed: int = 42
    as_of: str = "2025-06-30"
    min_events: int = 8
    max_events: int = 80
    lognormal_mean: float = 3.2  # median ~ 25 events
    lognormal_sigma: float = 0.5


def _genre_matrix(catalogue: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    vocab = sorted({g for gs in catalogue.genres for g in gs})
    index = {g: i for i, g in enumerate(vocab)}
    m = np.zeros((len(catalogue), len(vocab)), dtype=np.float32)
    for row, gs in enumerate(catalogue.genres):
        for g in gs:
            m[row, index[g]] = 1.0
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, norms, out=np.zeros_like(m), where=norms > 0), index


def simulate(catalogue: pd.DataFrame, cfg: SimConfig = SimConfig()) -> pd.DataFrame:
    """Return events with columns user_id, item_id, persona, ts."""
    rng = np.random.default_rng(cfg.seed)
    items = catalogue[catalogue.recommendable].reset_index(drop=True)
    as_of = pd.Timestamp(cfg.as_of)
    released = items.date_added.le(as_of).to_numpy()

    genre_m, genre_index = _genre_matrix(items)
    pop = items.popularity_pct.to_numpy(dtype=np.float64) ** cfg.pop_exponent
    langs = items.language.to_numpy()

    docs = items.title.fillna("") + ". " + items.description.fillna("")
    tfidf = TfidfVectorizer(max_features=30_000, min_df=2, sublinear_tf=True, stop_words="english").fit_transform(docs)

    names = list(PERSONAS)
    user_ids = [f"u{i:05d}" for i in range(cfg.users)]
    personas = [names[rng.integers(len(names))] for _ in user_ids]

    rows: list[tuple[str, str, str, pd.Timestamp]] = []
    for user, persona_name in zip(user_ids, personas):
        persona = PERSONAS[persona_name]

        # genre taste: Dirichlet around the persona's declared weights
        alpha = np.full(len(genre_index), 0.05)
        total = sum(persona["genres"].values())
        for g, w in persona["genres"].items():
            if g in genre_index:
                alpha[genre_index[g]] += 30.0 * w / total
        taste = rng.dirichlet(alpha)
        affinity = genre_m @ taste
        prob = (affinity + 0.003) ** 2

        # taste anchor: one title drawn from the genre taste; text similarity to it
        anchor = rng.choice(len(items), p=prob * released / (prob * released).sum())
        text_sim = np.asarray(tfidf @ tfidf[anchor].T.toarray()).ravel()
        prob = prob * (text_sim + 0.02)

        prob = prob * pop
        for lang, boost in persona["language"].items():
            prob[langs == lang] *= boost
        prob[~released] = 0.0
        prob /= prob.sum()

        n = int(np.clip(rng.lognormal(cfg.lognormal_mean, cfg.lognormal_sigma), cfg.min_events, cfg.max_events))
        n = min(n, int((prob > 0).sum()))
        chosen = rng.choice(len(items), size=n, replace=False, p=prob)
        for idx in chosen:
            start = max(items.date_added.iloc[idx], as_of - pd.Timedelta(days=730))
            span = max(1, int((as_of - start).total_seconds()))
            ts = start + pd.Timedelta(seconds=int(rng.integers(span)))
            rows.append((user, items.item_id.iloc[idx], persona_name, ts))

    events = pd.DataFrame(rows, columns=["user_id", "item_id", "persona", "ts"])
    return events.sort_values(["user_id", "ts"]).reset_index(drop=True)
