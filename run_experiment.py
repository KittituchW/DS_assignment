"""Run the full experiment, stage by stage.

Stages (default: all five, in order):
    prepare    check inputs, clean catalogue, EDA, item features, simulated viewers
    baselines  popularity (+ type-balanced check), most watched, content kNN (+ text-only and no-text)
    train      two tower: H1 ablation (5 seeds at p = 6) and H2 sweep (p = 2 and 12). Needs PyTorch.
    tune       random search on a validation split, then the chosen settings on the test set
               (exploratory, added after the main run). Needs PyTorch.
    coldstart  hold 10% of titles out of training entirely, then rank only those titles
               (exploratory item cold start). Needs PyTorch.
    analyse    H1 and H2 decision rules, result tables, figures, sample recommendations

Examples:
    python run_experiment.py --download                 # everything, about 30 minutes on a laptop CPU
    python run_experiment.py --stages tune analyse      # only the search, after prepare and train
    python run_experiment.py --min-votes 5 --output artifacts/sensitivity_min5 --stages prepare baselines train analyse
    python run_experiment.py --stages prepare baselines analyse   # no PyTorch needed
    python run_experiment.py --stages analyse           # re-run the analysis from saved runs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from recommender import analysis  # noqa: E402
from recommender.data import (  # noqa: E402
    MIN_VOTES,
    build_catalogue,
    check_inputs,
    download_raw,
    load_catalogue,
    load_raw,
    recommendable_items,
    save_catalogue,
)
from recommender.eda import save_eda  # noqa: E402
from recommender.evaluation import (  # noqa: E402
    cold_start_split,
    content_knn_scorer,
    evaluate,
    evaluate_cold,
    most_watched_scorer,
    popularity_by_type_scorer,
    popularity_scorer,
    temporal_split,
    two_tower_scorer,
    validation_split,
)
from recommender.features import ItemFeatures, build_item_features  # noqa: E402
from recommender.simulate import SimConfig, demo_histories, simulate  # noqa: E402
from recommender.tuning import pick_best, sample_configs  # noqa: E402

STAGES = ["prepare", "baselines", "train", "tune", "coldstart", "analyse"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "raw"), help="folder with the two CSV files")
    parser.add_argument("--download", action="store_true", help="download missing CSV files from the pinned commit")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "artifacts"))
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--users", type=int, default=3000)
    parser.add_argument("--exponents", type=float, nargs="+", default=[2.0, 6.0, 12.0])
    parser.add_argument("--main-exponent", type=float, default=6.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sim-seed", type=int, default=42)
    parser.add_argument("--min-votes", type=int, default=MIN_VOTES, help="vote floor for recommendable titles (5 = sensitivity run)")
    parser.add_argument("--tune-trials", type=int, default=20, help="settings tried in the random search, default first")
    parser.add_argument("--tune-seed", type=int, default=0)
    parser.add_argument("--cold-frac", type=float, default=0.1, help="share of titles hidden from training in the cold-start test")
    parser.add_argument("--cold-seed", type=int, default=0)
    return parser.parse_args()


class Paths:
    def __init__(self, output: str) -> None:
        self.out = Path(output)
        self.cache = self.out / "cache"
        self.runs = self.cache / "runs"
        self.models = self.cache / "models"
        for p in (self.out, self.cache, self.runs, self.models):
            p.mkdir(parents=True, exist_ok=True)

    def events(self, exponent: float) -> Path:
        return self.cache / f"events_p{exponent:g}.csv"

    def item_table(self, exponent: float, seed: int) -> Path:
        return self.models / f"item_table_p{exponent:g}_s{seed}.npy"


def load_items(paths: Paths) -> tuple[pd.DataFrame, ItemFeatures]:
    items = recommendable_items(load_catalogue(paths.cache / "catalogue.csv"))
    feats = ItemFeatures.load(paths.cache / "item_features.npz")
    if list(feats.item_ids) != list(items["item_id"]):
        raise RuntimeError("feature rows do not match the catalogue order; re-run the prepare stage")
    return items, feats


def load_split(paths: Paths, exponent: float, items: pd.DataFrame):
    events = pd.read_csv(paths.events(exponent), parse_dates=["ts"])
    return temporal_split(events, list(items["item_id"]))


# ------------------------------------------------------------------ stages


def stage_prepare(args: argparse.Namespace, paths: Paths) -> dict:
    data_dir = Path(args.data_dir)
    if args.download:
        download_raw(data_dir)
    hashes = check_inputs(data_dir)
    raw = load_raw(data_dir)
    catalogue = build_catalogue(raw, args.min_votes)
    save_catalogue(catalogue, paths.cache / "catalogue.csv")
    catalogue = load_catalogue(paths.cache / "catalogue.csv")
    summary = save_eda(raw, catalogue, paths.out, args.min_votes)
    items = recommendable_items(catalogue)
    print(f"catalogue: {len(catalogue):,} items, {len(items):,} recommendable")

    feats = build_item_features(items, seed=args.sim_seed)
    feats.save(paths.cache / "item_features.npz")
    print(f"features: {feats.matrix.shape}, groups {list(feats.groups)}")

    demo_histories(items).to_csv(paths.cache / "demo_histories.csv", index=False)
    sim = {}
    for p in args.exponents:
        t0 = time.time()
        events = simulate(items, SimConfig(users=args.users, pop_exponent=p, seed=args.sim_seed))
        events.to_csv(paths.events(p), index=False)
        counts = events["item_id"].value_counts()
        head = counts.head(max(1, int(len(counts) * 0.01))).sum() / len(events)
        sim[f"{p:g}"] = {"events": len(events), "items_watched": int(len(counts)), "top_1pct_item_share": round(float(head), 3)}
        print(f"events p={p:g}: {len(events):,} events, top 1% of items take {head:.1%} ({time.time() - t0:.0f}s)")
    return {"input_sha256": hashes, "eda": summary, "simulation": sim}


def stage_baselines(args: argparse.Namespace, paths: Paths) -> None:
    items, feats = load_items(paths)
    for p in args.exponents:
        split = load_split(paths, p, items)
        scorers = {
            "popularity": popularity_scorer(items),
            "popularity_by_type": popularity_by_type_scorer(items),
            "most_watched": most_watched_scorer(split.history_matrix()),
            "content_knn": content_knn_scorer(feats.matrix),
            "content_knn_text_only": content_knn_scorer(feats.columns("text")),
            "content_knn_no_text": content_knn_scorer(feats.matrix * feats.mask_without("text")),
        }
        for name, scorer in scorers.items():
            summary, per_user = evaluate(scorer, split, items)
            analysis.save_run(paths.runs, name, p, None, summary, per_user)
            print(f"p={p:g} {name:22s} " + " ".join(f"{k}={v:.3f}" for k, v in summary.items()))


def stage_train(args: argparse.Namespace, paths: Paths) -> None:
    try:
        from recommender.two_tower import TrainConfig, train_two_tower
    except ImportError as exc:
        raise SystemExit(f"PyTorch is needed for the train stage ({exc}). Install requirements.txt or skip this stage.")

    items, feats = load_items(paths)
    no_text = feats.mask_without("text")
    runs = []
    for p in args.exponents:
        for seed in args.seeds if p == args.main_exponent else args.seeds[:1]:
            runs.append((p, "two_tower", seed))
            if p == args.main_exponent:
                runs.append((p, "two_tower_no_text", seed))

    splits: dict[float, object] = {}
    for p, variant, seed in runs:
        split = splits.setdefault(p, load_split(paths, p, items))
        print(f"== p={p:g} {variant} seed={seed}")
        t0 = time.time()
        mask = no_text if variant == "two_tower_no_text" else None
        _, table, losses = train_two_tower(
            feats.matrix, split.history_matrix(), TrainConfig(epochs=args.epochs, seed=seed), feature_mask=mask
        )
        summary, per_user = evaluate(two_tower_scorer(table), split, items)
        analysis.save_run(
            paths.runs, variant, p, seed, summary, per_user,
            train_seconds=round(time.time() - t0), loss_history=[round(x, 4) for x in losses],
        )
        if variant == "two_tower":
            np.save(paths.item_table(p, seed), table)
        print("   " + " ".join(f"{k}={v:.3f}" for k, v in summary.items()))


def stage_tune(args: argparse.Namespace, paths: Paths) -> dict:
    """Choose settings on a validation split, then run them once on the test set.

    Validation = each viewer's latest training events, by the same rule as the test
    split. The test set plays no part in the choice.
    """
    try:
        from recommender.two_tower import TrainConfig, train_two_tower
    except ImportError as exc:
        raise SystemExit(f"PyTorch is needed for the tune stage ({exc}). Install requirements.txt or skip this stage.")

    items, feats = load_items(paths)
    p = args.main_exponent
    split = load_split(paths, p, items)
    val = validation_split(split)
    trials = []
    for i, config in enumerate(sample_configs(args.tune_trials, args.tune_seed)):
        t0 = time.time()
        _, table, _ = train_two_tower(
            feats.matrix, val.history_matrix(), TrainConfig(epochs=args.epochs, seed=args.seeds[0], **config), log=False
        )
        summary, _ = evaluate(two_tower_scorer(table), val, items)
        trials.append({"trial": i, **config, **{f"val_{k}": v for k, v in summary.items() if k in ("recall@10", "ndcg@10", "hit@10")},
                       "seconds": round(time.time() - t0)})
        print(f"  trial {i:2d} {config} val recall@10={summary['recall@10']:.4f} ({trials[-1]['seconds']}s)")
    best_i = pick_best(trials)
    best = {k: trials[best_i][k] for k in sample_configs(1)[0]}
    search = {
        "exponent": p,
        "seed": args.seeds[0],
        "trials": trials,
        "best": best,
        "best_val_recall@10": trials[best_i]["val_recall@10"],
        "default_val_recall@10": trials[0]["val_recall@10"],
    }
    (paths.out / "tuning.json").write_text(json.dumps(search, indent=2), encoding="utf-8")
    pd.DataFrame(trials).to_csv(paths.out / "tuning_trials.csv", index=False)
    print(f"  chosen: {best}")

    no_text = feats.mask_without("text")
    for q in args.exponents:
        q_split = split if q == p else load_split(paths, q, items)
        for seed in args.seeds if q == p else args.seeds[:1]:
            variants = [("two_tower_tuned", None)] + ([("two_tower_tuned_no_text", no_text)] if q == p else [])
            for variant, mask in variants:
                t0 = time.time()
                _, table, losses = train_two_tower(
                    feats.matrix, q_split.history_matrix(), TrainConfig(epochs=args.epochs, seed=seed, **best),
                    feature_mask=mask, log=False,
                )
                summary, per_user = evaluate(two_tower_scorer(table), q_split, items)
                analysis.save_run(
                    paths.runs, variant, q, seed, summary, per_user, config=best,
                    train_seconds=round(time.time() - t0), loss_history=[round(x, 4) for x in losses],
                )
                print(f"  p={q:g} {variant} seed={seed} recall@10={summary['recall@10']:.4f}")
    return {"best": best, "best_val_recall@10": search["best_val_recall@10"], "default_val_recall@10": search["default_val_recall@10"]}


def stage_coldstart(args: argparse.Namespace, paths: Paths) -> dict:
    """Item cold start (exploratory): titles hidden from training, then ranked from their metadata alone."""
    try:
        from recommender.two_tower import TrainConfig, embed_items, train_two_tower
    except ImportError as exc:
        raise SystemExit(f"PyTorch is needed for the coldstart stage ({exc}). Install requirements.txt or skip this stage.")

    items, feats = load_items(paths)
    p = args.main_exponent
    events = pd.read_csv(paths.events(p), parse_dates=["ts"])
    split, cold = cold_start_split(events, items, args.cold_frac, args.cold_seed)
    warm = np.flatnonzero(~cold)
    history = split.history_matrix()
    print(f"  {int(cold.sum()):,} titles held out of training, {len(split.test):,} of their watches held out")
    for name, scorer in (("cold_content_knn", content_knn_scorer(feats.matrix)), ("cold_popularity", popularity_scorer(items))):
        summary, per_user = evaluate_cold(scorer, split, items, cold)
        analysis.save_run(paths.runs, name, p, None, summary, per_user)
        print(f"  {name:18s} recall@10={summary['recall@10']:.4f} hit@10={summary['hit@10']:.4f}")
    for seed in args.seeds:
        t0 = time.time()
        # Train on the other titles only: held-out titles are not even negatives in the softmax.
        tower, _, losses = train_two_tower(
            feats.matrix[warm], history[:, warm], TrainConfig(epochs=args.epochs, seed=seed), log=False
        )
        table = embed_items(tower, feats.matrix)
        summary, per_user = evaluate_cold(two_tower_scorer(table), split, items, cold)
        analysis.save_run(
            paths.runs, "cold_two_tower", p, seed, summary, per_user,
            train_seconds=round(time.time() - t0), loss_history=[round(x, 4) for x in losses],
        )
        print(f"  cold_two_tower seed={seed} recall@10={summary['recall@10']:.4f} hit@10={summary['hit@10']:.4f}")
    return {"cold_items": int(cold.sum()), "viewers_with_held_out_watches": summary["viewers"]}


def stage_analyse(args: argparse.Namespace, paths: Paths) -> dict:
    runs = analysis.all_runs_table(paths.runs)
    runs.to_csv(paths.out / "metrics_all_runs.csv", index=False)
    h1 = analysis.check_h1(paths.runs, args.main_exponent, args.seeds)
    h2 = analysis.check_h2(paths.runs, args.exponents, args.main_exponent, args.seeds)
    comparison = analysis.compare_two_tower(paths.runs, args.exponents, args.main_exponent, args.seeds)
    tuning_path = paths.out / "tuning.json"
    search = json.loads(tuning_path.read_text()) if tuning_path.exists() else None
    tuned = analysis.tuned_summary(paths.runs, args.exponents, args.main_exponent, args.seeds, search)
    cold = analysis.cold_start_summary(paths.runs, args.main_exponent, args.seeds)
    (paths.out / "hypothesis_tests.json").write_text(
        json.dumps(
            {"H1": h1, "H2": h2, "exploratory_two_tower_vs_content_knn": comparison, "exploratory_tuned_two_tower": tuned,
             "exploratory_cold_start": cold, "min_votes": args.min_votes},
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    (paths.out / "results_summary.md").write_text(analysis.results_summary(runs, h1, h2, comparison, tuned, cold), encoding="utf-8")
    analysis.save_figures(h1, h2, paths.out)

    items, feats = load_items(paths)
    demo = pd.read_csv(paths.cache / "demo_histories.csv")
    scorers = {"content_knn": content_knn_scorer(feats.matrix)}
    table_path = paths.item_table(args.main_exponent, args.seeds[0])
    if table_path.exists():
        scorers["two_tower"] = two_tower_scorer(np.load(table_path))
    records, text = analysis.sample_recommendations(items, demo, scorers)
    records.to_csv(paths.out / "sample_recommendations.csv", index=False)
    (paths.out / "sample_recommendations.md").write_text(text, encoding="utf-8")
    print(f"H1: {h1['verdict']}. H2a: {h2['h2a_verdict']}. H2b: {h2['h2b_verdict']}.")
    return {"H1": h1["verdict"], "H2a": h2["h2a_verdict"], "H2b": h2["h2b_verdict"]}


def main() -> None:
    args = parse_args()
    paths = Paths(args.output)
    summary_path = paths.out / "run_summary.json"
    run_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    run_summary["config"] = {k: v for k, v in vars(args).items() if k not in ("data_dir", "output", "stages")}

    for stage in STAGES:
        if stage not in args.stages:
            continue
        print(f"\n### stage: {stage}")
        t0 = time.time()
        result = {
            "prepare": stage_prepare,
            "baselines": stage_baselines,
            "train": stage_train,
            "tune": stage_tune,
            "coldstart": stage_coldstart,
            "analyse": stage_analyse,
        }[stage](args, paths)
        if result:
            run_summary[stage] = result
        run_summary.setdefault("stage_seconds", {})[stage] = round(time.time() - t0)
        summary_path.write_text(json.dumps(run_summary, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
