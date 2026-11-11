"""Stage 4: sample recommendations for the three demo users.

numpy only. Uses the trained item table if it exists, and always shows the
content kNN baseline next to it for comparison.

Run: python scripts/recommend.py [--exponent 6]
Writes reports/sample_recommendations.md
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from baselines import load_items_and_features  # noqa: E402
from src.evaluate import content_knn_scorer, top_k  # noqa: E402
from src.evaluate import two_tower_scorer  # noqa: E402

DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
K = 10


def describe(items: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    rows = items.iloc[idx][["title", "type", "language", "release_year", "genres", "vote_count"]].copy()
    rows["genres"] = rows.genres.map(lambda g: ", ".join(g[:3]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exponent", type=float, default=6.0)
    args = parser.parse_args()

    items, feats, _ = load_items_and_features()
    index = {it: i for i, it in enumerate(items.item_id)}
    demo = pd.read_csv(DATA / "demo_histories.csv")
    users = list(dict.fromkeys(demo.user_id))
    rows = demo.user_id.map({u: i for i, u in enumerate(users)}).to_numpy()
    cols = demo.item_id.map(index).to_numpy()
    history = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(users), len(items)))

    scorers = {"content_knn": content_knn_scorer(feats)}
    table_path = MODELS / f"item_table_p{args.exponent:g}.npy"
    if table_path.exists():
        scorers["two_tower"] = two_tower_scorer(np.load(table_path))
    else:
        print(f"no trained model at {table_path}; showing the baseline only")

    lines = ["# Sample recommendations", "", f"Top {K} for three hand seeded users. Already watched titles are excluded.", ""]
    for name, scorer in scorers.items():
        recs = top_k(scorer(history), history, K)
        for u, user in enumerate(users):
            seeds = items.iloc[history[u].indices]
            seed_genres = Counter(g for gs in seeds.genres for g in gs)
            rec = describe(items, recs[u])
            rec_genres = Counter(g for gs in items.iloc[recs[u]].genres for g in gs)
            overlap = sum(min(seed_genres[g], rec_genres[g]) for g in seed_genres) / max(1, sum(rec_genres.values()))
            lines.append(f"## {name} / {user}")
            lines.append("")
            lines.append("Seeds: " + "; ".join(seeds.title) + ".")
            lines.append("")
            lines.append(rec.to_markdown(index=False))
            lines.append("")
            lines.append(
                f"Non-English in top {K}: {int((rec.language != 'en').sum())}. TV: {int((rec.type == 'tv').sum())}. "
                f"Median votes: {int(rec.vote_count.median())}. Genre overlap with seeds: {overlap:.0%}."
            )
            lines.append("")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "sample_recommendations.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
