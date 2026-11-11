"""Stage 1: clean catalogue, item features, simulated events.

Run: python scripts/prepare.py [--users 3000] [--exponents 2 6 12]
Outputs under data/processed/:
    catalogue.csv        clean item table (all items, with `recommendable` flag)
    item_features.npz    feature matrix for recommendable items + group slices
    events_p{e}.csv      simulated events for each popularity exponent e
    demo_histories.csv   hand picked seed histories for user_A/B/C
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_catalogue, load_catalogue, save_catalogue  # noqa: E402
from src.features import build_item_features  # noqa: E402
from src.simulate import SimConfig, demo_histories, simulate  # noqa: E402

OUT = ROOT / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=3000)
    parser.add_argument("--exponents", type=float, nargs="+", default=[2.0, 6.0, 12.0])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cat = build_catalogue(ROOT / "data" / "raw")
    save_catalogue(cat, OUT / "catalogue.csv")
    cat = load_catalogue(OUT / "catalogue.csv")
    items = cat[cat.recommendable].reset_index(drop=True)
    print(f"catalogue: {len(cat):,} items, {len(items):,} recommendable")

    feats = build_item_features(items, seed=args.seed)
    np.savez_compressed(
        OUT / "item_features.npz",
        matrix=feats.matrix,
        item_ids=np.array(feats.item_ids),
        groups=json.dumps({k: [v.start, v.stop] for k, v in feats.groups.items()}),
    )
    print(f"features: {feats.matrix.shape}, groups {list(feats.groups)}")

    demo_histories(cat).to_csv(OUT / "demo_histories.csv", index=False)

    for exponent in args.exponents:
        t0 = time.time()
        events = simulate(cat, SimConfig(users=args.users, pop_exponent=exponent, seed=args.seed))
        tag = f"{exponent:g}"
        events.to_csv(OUT / f"events_p{tag}.csv", index=False)
        top = events.item_id.value_counts()
        head = top.head(max(1, int(len(top) * 0.01))).sum() / len(events)
        print(f"events p={tag}: {len(events):,} events, {events.user_id.nunique():,} users, "
              f"{events.item_id.nunique():,} items, top-1% items take {head:.1%} of events ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
