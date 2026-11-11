from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recommender.data import CORE_METADATA, build_eda_summary, load_and_clean_catalog
from recommender.evaluation import (
    build_cold_start_episodes,
    evaluate_models,
    recommend_for_personas,
)
from recommender.features import fit_content_features
from recommender.neural import encode, train_projection_network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare catalog recommendation models")
    parser.add_argument("--movies", required=True, help="Path to the movie CSV")
    parser.add_argument("--tv", required=True, help="Path to the TV CSV")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=15)
    return parser.parse_args()


def save_eda_figures(catalog: pd.DataFrame, output: Path) -> None:
    missing = catalog.groupby("type")[CORE_METADATA].apply(lambda x: x.isna().mean() * 100)
    ax = missing.T.plot(kind="bar", figsize=(9, 5), color=["#2563eb", "#f97316"])
    ax.set_ylabel("Missing values (%)")
    ax.set_title("Metadata missingness differs sharply by content type")
    ax.legend(title="")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output / "eda_missingness.png", dpi=180)
    plt.close()

    counts = catalog.groupby(["release_year", "type"]).size().unstack(fill_value=0)
    ax = counts.plot(figsize=(9, 4), marker="o", color=["#2563eb", "#f97316"])
    ax.set_ylabel("Catalog rows")
    ax.set_title("The dataset is uniformly sampled by year and type")
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(output / "eda_year_counts.png", dpi=180)
    plt.close()


def save_metrics_figure(metrics: pd.DataFrame, output: Path) -> None:
    plot = metrics.set_index("model")[["recall@10", "ndcg@10", "mrr@10"]]
    ax = plot.plot(kind="bar", figsize=(9, 5), color=["#2563eb", "#14b8a6", "#f97316"])
    ax.set_ylim(0, max(0.05, float(plot.to_numpy().max()) * 1.2))
    ax.set_ylabel("Score")
    ax.set_title("Cold-start proxy evaluation on 2025 titles")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output / "model_comparison.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    catalog = load_and_clean_catalog(args.movies, args.tv)
    with (output / "eda_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(build_eda_summary(catalog), handle, indent=2, ensure_ascii=False)
    save_eda_figures(catalog, output)

    train_mask = catalog.release_year.le(2023)
    features = fit_content_features(
        catalog.loc[train_mask, "feature_document"].tolist(),
        catalog["feature_document"].tolist(),
        seed=args.seed,
    )
    train_indices = catalog.index[train_mask].to_numpy()
    training = train_projection_network(
        features.dense[train_indices],
        catalog.loc[train_indices].reset_index(drop=True),
        epochs=args.epochs,
        seed=args.seed,
    )
    neural_embeddings = encode(training.model, features.dense)

    episodes = build_cold_start_episodes(
        catalog, target_year=2025, max_episodes=args.episodes, seed=args.seed
    )
    metrics, _ = evaluate_models(
        catalog,
        episodes,
        features.tfidf,
        features.dense,
        neural_embeddings,
        target_year=2025,
    )
    metrics.to_csv(output / "model_metrics.csv", index=False)
    save_metrics_figure(metrics, output)

    persona_seeds = {
        "Mind-bending science fiction": ["Inception", "Black Mirror", "Stranger Things"],
        "Korean drama and suspense": ["Crash Landing on You", "Squid Game"],
        "Family animation": ["Coco", "Shrek Forever After"],
    }
    recommendations = recommend_for_personas(
        catalog, neural_embeddings, persona_seeds, top_k=5
    )
    recommendations.to_csv(output / "sample_recommendations.csv", index=False)

    catalog_export = catalog[
        [
            "item_id",
            "type",
            "title",
            "release_year",
            "language",
            "genres",
            "metadata_completeness",
        ]
    ]
    catalog_export.to_csv(output / "clean_catalog_index.csv", index=False)
    np.save(output / "neural_item_embeddings.npy", neural_embeddings)
    torch.save(training.model.state_dict(), output / "projection_network.pt")
    joblib.dump(features.vectorizer, output / "tfidf_vectorizer.joblib")
    joblib.dump(features.svd, output / "svd.joblib")

    run_summary = {
        "seed": args.seed,
        "train_rows": int(train_mask.sum()),
        "validation_rows_2024": int(catalog.release_year.eq(2024).sum()),
        "test_rows_2025": int(catalog.release_year.eq(2025).sum()),
        "evaluation_episodes": len(episodes),
        "triplets": training.triplet_count,
        "loss_history": training.loss_history,
        "tfidf_features": len(features.vectorizer.get_feature_names_out()),
        "svd_dimensions": int(features.dense.shape[1]),
        "neural_dimensions": int(neural_embeddings.shape[1]),
    }
    with (output / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)
    print(json.dumps(run_summary, indent=2))
    print("\nMetrics\n", metrics.to_string(index=False))
    print("\nSample recommendations\n", recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
