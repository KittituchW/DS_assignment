"""Stage 3: train and evaluate the two tower model, including the hypothesis runs.

Runs (default):
    H2 bias sweep   full model on events_p2, events_p6, events_p12
    H1 ablation     full vs text-ablated model on events_p6, two seeds each

Run: python scripts/train.py [--main-exponent 6] [--seeds 42 43] [--epochs 10]
Writes:
    reports/results_two_tower.csv
    models/item_table_p{e}.npy     item embedding table of the full model per exponent (seed 0)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from baselines import load_items_and_features  # noqa: E402
from src.evaluate import evaluate, temporal_split, two_tower_scorer  # noqa: E402
from src.model import TrainConfig, train_two_tower  # noqa: E402

DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-exponent", type=float, default=6.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    items, feats, groups = load_items_and_features()
    item_ids = list(items.item_id)
    no_text = np.ones(feats.shape[1], dtype=np.float32)
    no_text[groups["text"]] = 0.0

    runs: list[tuple[float, str, int]] = []
    for path in sorted(DATA.glob("events_p*.csv")):
        exponent = float(path.stem.split("_p")[1])
        if exponent == args.main_exponent:
            for seed in args.seeds:
                runs.append((exponent, "two_tower", seed))
                runs.append((exponent, "two_tower_no_text", seed))
        else:
            runs.append((exponent, "two_tower", args.seeds[0]))

    rows = []
    cache: dict[float, object] = {}
    for exponent, variant, seed in runs:
        if exponent not in cache:
            events = pd.read_csv(DATA / f"events_p{exponent:g}.csv", parse_dates=["ts"])
            cache[exponent] = temporal_split(events, item_ids)
        split = cache[exponent]
        print(f"== p={exponent:g} {variant} seed={seed}")
        t0 = time.time()
        cfg = TrainConfig(epochs=args.epochs, seed=seed)
        mask = no_text if variant.endswith("no_text") else None
        _, table = train_two_tower(feats, split.history_matrix(), cfg, feature_mask=mask)
        m = evaluate(two_tower_scorer(table), split, items, k=10)
        rows.append({"model": variant, "pop_exponent": exponent, "seed": seed, "train_seconds": round(time.time() - t0), **m})
        print("   " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))
        if variant == "two_tower" and seed == args.seeds[0]:
            np.save(MODELS / f"item_table_p{exponent:g}.npy", table)
        pd.DataFrame(rows).to_csv(REPORTS / "results_two_tower.csv", index=False)


if __name__ == "__main__":
    main()
