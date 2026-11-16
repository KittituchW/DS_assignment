"""Simulated viewers.

The catalogue has no viewers. To train and evaluate a personalised model at
all, viewing events are simulated from an explicit choice model. It is a
research instrument, not a claim about real viewers. The rules are simple so
every result can be traced back to a rule.

Choice model for one viewer:

    p(item) ∝ (genre_affinity + eps)^2              persona genre taste
              * (text_similarity_to_anchor + eps)    one "taste anchor" title
              * popularity_pct ** pop_exponent       how herd-driven the viewer is
              * language_boost                       persona language taste

`pop_exponent` is the bias dial used by H2.

Watch ratio (added after the main run, exploratory only): each event also gets
the share of the title the viewer watched, drawn as

    watch_ratio ~ Beta(1 + 5 * fit, 2)

where `fit` is the title's rank by watch chance among that viewer's own picks
(0 = weakest fit, 1 = best fit). A better fit is more likely to be finished.
It feeds one exploratory metric only. Training never sees it, and it draws
from its own random stream, so the watch events are identical with or without it.

The text term uses sparse TF-IDF cosine to the anchor title. The model sees a
different representation (an SVD of a separate TF-IDF), so it cannot read the
rule straight out of its inputs.
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

# Three hand-seeded viewers for the sample recommendations. Their histories are
# well-known titles so a reader can judge the output. They are not in the metrics.
DEMO_USERS: dict[str, list[str]] = {
    "user_A_action": [
        "Inception", "Mad Max: Fury Road", "John Wick", "Extraction", "Edge of Tomorrow", "The Dark Knight",
    ],
    "user_B_kdrama": [
        "Crash Landing on You", "Squid Game", "Itaewon Class", "Vincenzo", "Parasite", "Our Beloved Summer",
    ],
    "user_C_documentary": [
        "Our Planet", "The Social Dilemma", "Chef's Table", "Making a Murderer", "My Octopus Teacher", "Free Solo",
    ],
}


@dataclass
class SimConfig:
    users: int = 3000
    pop_exponent: float = 6.0
    seed: int = 42
    as_of: str = "2025-06-30"
    min_events: int = 8
    max_events: int = 80
    lognormal_mean: float = 3.2  # median about 25 events
    lognormal_sigma: float = 0.5


def demo_histories(items: pd.DataFrame) -> pd.DataFrame:
    """Resolve the demo seed titles to item ids. Ties go to the most-voted title."""
    rows = []
    lowered = items["title"].str.lower()
    for user, titles in DEMO_USERS.items():
        for title in titles:
            match = items[lowered.eq(title.lower())].sort_values("vote_count", ascending=False)
            if len(match):
                rows.append((user, match["item_id"].iloc[0], match["title"].iloc[0]))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "title"])


def _genre_matrix(items: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    vocab = sorted({g for gs in items["genres"] for g in gs})
    index = {g: i for i, g in enumerate(vocab)}
    m = np.zeros((len(items), len(vocab)), dtype=np.float32)
    for row, gs in enumerate(items["genres"]):
        for g in gs:
            m[row, index[g]] = 1.0
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, norms, out=np.zeros_like(m), where=norms > 0), index


def simulate(items: pd.DataFrame, cfg: SimConfig = SimConfig()) -> pd.DataFrame:
    """Return events with columns user_id, item_id, persona, ts. `items` are the recommendable items."""
    rng = np.random.default_rng(cfg.seed)
    items = items.reset_index(drop=True)
    as_of = pd.Timestamp(cfg.as_of)
    released = items["date_added"].le(as_of).to_numpy()

    genre_m, genre_index = _genre_matrix(items)
    pop = items["popularity_pct"].to_numpy(dtype=np.float64) ** cfg.pop_exponent
    langs = items["language"].to_numpy()
    docs = items["title"] + ". " + items["description"]
    tfidf = TfidfVectorizer(max_features=30_000, min_df=2, sublinear_tf=True, stop_words="english").fit_transform(docs)

    names = list(PERSONAS)
    ratio_rng = np.random.default_rng([cfg.seed, 1])  # separate stream: the events do not change
    rows: list[tuple[str, str, str, pd.Timestamp, float]] = []
    for u in range(cfg.users):
        user = f"u{u:05d}"
        persona_name = names[rng.integers(len(names))]
        persona = PERSONAS[persona_name]

        # Genre taste: Dirichlet around the persona's weights.
        alpha = np.full(len(genre_index), 0.05)
        total = sum(persona["genres"].values())
        for g, w in persona["genres"].items():
            if g in genre_index:
                alpha[genre_index[g]] += 30.0 * w / total
        prob = (genre_m @ rng.dirichlet(alpha) + 0.003) ** 2

        # Taste anchor: one released title drawn from the genre taste.
        anchor_p = prob * released
        anchor = rng.choice(len(items), p=anchor_p / anchor_p.sum())
        text_sim = np.asarray((tfidf @ tfidf[anchor].T).toarray()).ravel()
        prob = prob * (text_sim + 0.02) * pop
        for lang, boost in persona["language"].items():
            prob[langs == lang] *= boost
        prob[~released] = 0.0
        prob /= prob.sum()

        n = int(np.clip(rng.lognormal(cfg.lognormal_mean, cfg.lognormal_sigma), cfg.min_events, cfg.max_events))
        n = min(n, int((prob > 0).sum()))
        picked = rng.choice(len(items), size=n, replace=False, p=prob)
        fit = np.argsort(np.argsort(prob[picked], kind="stable"), kind="stable") / max(n - 1, 1)
        ratios = ratio_rng.beta(1.0 + 5.0 * fit, 2.0)
        for idx, ratio in zip(picked, ratios):
            start = max(items["date_added"].iloc[idx], as_of - pd.Timedelta(days=730))
            span = max(1, int((as_of - start).total_seconds()))
            ts = start + pd.Timedelta(seconds=int(rng.integers(span)))
            rows.append((user, items["item_id"].iloc[idx], persona_name, ts, round(float(ratio), 4)))

    events = pd.DataFrame(rows, columns=["user_id", "item_id", "persona", "ts", "watch_ratio"])
    return events.sort_values(["user_id", "ts"], kind="stable").reset_index(drop=True)
