"""Stage 2: evaluate the two non-neural baselines on every simulated dataset.

Run: python scripts/baselines.py
Writes reports/results_baselines.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_catalogue  # noqa: E402
from src.evaluate import content_knn_scorer, evaluate, most_watched_scorer, popularity_scorer, temporal_split  # noqa: E402

DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def load_items_and_features() -> tuple[pd.DataFrame, np.ndarray, dict[str, slice]]:
    cat = load_catalogue(DATA / "catalogue.csv")
    items = cat[cat.recommendable].reset_index(drop=True)
    npz = np.load(DATA / "item_features.npz", allow_pickle=False)
    assert list(npz["item_ids"]) == list(items.item_id), "feature rows must match catalogue order"
    groups = {k: slice(a, b) for k, (a, b) in json.loads(str(npz["groups"])).items()}
    return items, npz["matrix"], groups


def main() -> None:
    items, feats, groups = load_items_and_features()
    rows = []
    for path in sorted(DATA.glob("events_p*.csv")):
        exponent = float(path.stem.split("_p")[1])
        events = pd.read_csv(path, parse_dates=["ts"])
        split = temporal_split(events, list(items.item_id))
        scorers = {
            "popularity": popularity_scorer(items),
            "most_watched": most_watched_scorer(),
            "content_knn": content_knn_scorer(feats),
            "content_knn_text_only": content_knn_scorer(feats[:, groups["text"]]),
            "content_knn_no_text": content_knn_scorer(np.delete(feats, np.r_[groups["text"]], axis=1)),
        }
        for name, scorer in scorers.items():
            m = evaluate(scorer, split, items, k=10)
            rows.append({"model": name, "pop_exponent": exponent, **m})
            print(f"p={exponent:g} {name:24s} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))
    REPORTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORTS / "results_baselines.csv", index=False)


if __name__ == "__main__":
    main()
