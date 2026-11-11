from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class Episode:
    target_index: int
    seed_indices: tuple[int, ...]


def _as_set(value: object) -> set[str]:
    if isinstance(value, list):
        return set(value)
    if pd.isna(value):
        return set()
    return {part.strip() for part in str(value).split(",") if part.strip()}


def _metadata_similarity(target: pd.Series, candidate: pd.Series) -> float:
    target_genres = set(target["genre_list"])
    candidate_genres = set(candidate["genre_list"])
    union = target_genres | candidate_genres
    genre_score = len(target_genres & candidate_genres) / len(union) if union else 0.0
    language = float(target["language"] == candidate["language"])
    cast_overlap = bool(_as_set(target.get("cast")) & _as_set(candidate.get("cast")))
    director_overlap = bool(_as_set(target.get("director")) & _as_set(candidate.get("director")))
    year_distance = abs(int(target["release_year"]) - int(candidate["release_year"]))
    recency = max(0.0, 1.0 - year_distance / 15.0)
    return 0.55 * genre_score + 0.15 * language + 0.15 * cast_overlap + 0.1 * director_overlap + 0.05 * recency


def build_cold_start_episodes(
    catalog: pd.DataFrame,
    target_year: int = 2025,
    max_episodes: int = 500,
    seeds_per_episode: int = 3,
    seed: int = 42,
) -> list[Episode]:
    rng = np.random.default_rng(seed)
    targets = catalog[
        catalog.release_year.eq(target_year)
        & catalog.genre_list.map(bool)
        & catalog.description.notna()
    ]
    target_indices = targets.index.to_numpy()
    rng.shuffle(target_indices)
    seed_pool = catalog[catalog.release_year.lt(target_year)]
    episodes: list[Episode] = []

    for target_index in target_indices:
        target = catalog.loc[target_index]
        candidates = seed_pool[
            seed_pool["type"].eq(target["type"])
            & seed_pool["primary_genre"].eq(target["primary_genre"])
        ]
        if len(candidates) < seeds_per_episode:
            continue
        scores = candidates.apply(lambda row: _metadata_similarity(target, row), axis=1)
        best = scores.nlargest(seeds_per_episode)
        if len(best) < seeds_per_episode or float(best.mean()) < 0.35:
            continue
        episodes.append(Episode(int(target_index), tuple(int(i) for i in best.index)))
        if len(episodes) >= max_episodes:
            break
    return episodes


def _ranks_to_metrics(ranks: list[int], k: int = 10) -> dict[str, float]:
    array = np.asarray(ranks)
    hits = array <= k
    return {
        f"recall@{k}": float(hits.mean()),
        f"ndcg@{k}": float(np.where(hits, 1.0 / np.log2(array + 1), 0.0).mean()),
        f"mrr@{k}": float(np.where(hits, 1.0 / array, 0.0).mean()),
        "mean_rank": float(array.mean()),
        "median_rank": float(np.median(array)),
    }


def _mean_pairwise_distance(embeddings: np.ndarray) -> float:
    if len(embeddings) < 2:
        return 0.0
    similarities = embeddings @ embeddings.T
    upper = np.triu_indices(len(embeddings), k=1)
    return float((1.0 - similarities[upper]).mean())


def _mmr_top_k(
    relevance: np.ndarray,
    candidate_embeddings: np.ndarray,
    k: int = 10,
    lambda_relevance: float = 0.8,
    pool_size: int = 100,
) -> np.ndarray:
    pool = np.argsort(-relevance)[:pool_size]
    selected: list[int] = []
    while len(selected) < min(k, len(pool)):
        if not selected:
            choice = int(pool[0])
        else:
            remaining = np.asarray([idx for idx in pool if idx not in selected])
            redundancy = (candidate_embeddings[remaining] @ candidate_embeddings[selected].T).max(axis=1)
            values = lambda_relevance * relevance[remaining] - (1.0 - lambda_relevance) * redundancy
            choice = int(remaining[np.argmax(values)])
        selected.append(choice)
    return np.asarray(selected, dtype=int)


def evaluate_models(
    catalog: pd.DataFrame,
    episodes: list[Episode],
    tfidf: sparse.csr_matrix,
    svd_embeddings: np.ndarray,
    neural_embeddings: np.ndarray,
    target_year: int = 2025,
    k: int = 10,
) -> tuple[pd.DataFrame, dict[str, list[np.ndarray]]]:
    candidate_indices = catalog.index[catalog.release_year.eq(target_year)].to_numpy()
    candidate_lookup = {index: position for position, index in enumerate(candidate_indices)}
    candidate_tfidf = tfidf[candidate_indices]
    candidate_svd = svd_embeddings[candidate_indices]
    candidate_neural = neural_embeddings[candidate_indices]
    popularity = (
        0.85 * catalog.loc[candidate_indices, "log_popularity"].to_numpy()
        + 0.15 * catalog.loc[candidate_indices, "bayesian_rating"].to_numpy()
    )
    popularity = (popularity - popularity.mean()) / (popularity.std() + 1e-8)

    ranks: dict[str, list[int]] = {name: [] for name in ["popularity", "tfidf", "svd", "neural", "neural_mmr"]}
    top_lists: dict[str, list[np.ndarray]] = {name: [] for name in ranks}

    for episode in episodes:
        target_position = candidate_lookup[episode.target_index]
        seed_indices = np.asarray(episode.seed_indices)

        tfidf_profile = normalize(
            sparse.csr_matrix(tfidf[seed_indices].sum(axis=0) / len(seed_indices))
        )
        tfidf_scores = (candidate_tfidf @ tfidf_profile.T).toarray().ravel()

        svd_profile = svd_embeddings[seed_indices].mean(axis=0)
        svd_profile /= np.linalg.norm(svd_profile) + 1e-8
        svd_scores = candidate_svd @ svd_profile

        neural_profile = neural_embeddings[seed_indices].mean(axis=0)
        neural_profile /= np.linalg.norm(neural_profile) + 1e-8
        neural_scores = candidate_neural @ neural_profile

        score_map = {
            "popularity": popularity,
            "tfidf": tfidf_scores,
            "svd": svd_scores,
            "neural": neural_scores,
        }
        for name, scores in score_map.items():
            order = np.argsort(-scores)
            rank = int(np.flatnonzero(order == target_position)[0] + 1)
            ranks[name].append(rank)
            top_lists[name].append(order[:k])

        mmr = _mmr_top_k(neural_scores, candidate_neural, k=k)
        neural_order = np.argsort(-neural_scores)
        mmr_set = set(mmr.tolist())
        mmr_full_order = np.concatenate(
            [mmr, np.asarray([idx for idx in neural_order if idx not in mmr_set], dtype=int)]
        )
        ranks["neural_mmr"].append(
            int(np.flatnonzero(mmr_full_order == target_position)[0] + 1)
        )
        top_lists["neural_mmr"].append(mmr)

    rows: list[dict[str, float | str]] = []
    for name in ranks:
        row: dict[str, float | str] = {"model": name, **_ranks_to_metrics(ranks[name], k=k)}
        flattened = np.concatenate(top_lists[name])
        row["catalog_coverage@10"] = float(len(np.unique(flattened)) / len(candidate_indices))
        row["intra_list_diversity@10"] = float(
            np.mean([_mean_pairwise_distance(candidate_svd[top]) for top in top_lists[name]])
        )
        rows.append(row)
    return pd.DataFrame(rows), top_lists


def recommend_for_personas(
    catalog: pd.DataFrame,
    embeddings: np.ndarray,
    persona_seeds: dict[str, list[str]],
    top_k: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    normalized_titles = catalog.title.str.casefold()
    for persona, titles in persona_seeds.items():
        seed_indices: list[int] = []
        for title in titles:
            matches = catalog.index[normalized_titles.eq(title.casefold())]
            if len(matches):
                # Titles can collide across movies and TV. The most-voted match is
                # the least ambiguous default for a named, well-known seed.
                best_match = catalog.loc[matches, "vote_count"].idxmax()
                seed_indices.append(int(best_match))
        if not seed_indices:
            continue
        profile = embeddings[seed_indices].mean(axis=0)
        profile /= np.linalg.norm(profile) + 1e-8
        scores = embeddings @ profile
        scores[seed_indices] = -np.inf
        recommendations = np.argsort(-scores)[:top_k]
        seed_genres = set().union(*(set(catalog.at[index, "genre_list"]) for index in seed_indices))
        for rank, index in enumerate(recommendations, start=1):
            rec_genres = set(catalog.at[index, "genre_list"])
            shared = sorted(seed_genres & rec_genres)
            rows.append(
                {
                    "persona": persona,
                    "seed_titles": " | ".join(catalog.loc[seed_indices, "title"]),
                    "rank": rank,
                    "recommended_title": catalog.at[index, "title"],
                    "type": catalog.at[index, "type"],
                    "release_year": int(catalog.at[index, "release_year"]),
                    "score": float(scores[index]),
                    "shared_genres": ", ".join(shared) if shared else "semantic similarity",
                }
            )
    return pd.DataFrame(rows)
