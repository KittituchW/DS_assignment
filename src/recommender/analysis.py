"""Run storage, the H1 and H2 decision rules, and the result tables and figures.

Every model run saves its per-viewer metrics to one `.npz` file. The analysis
reads those files only, so it can be re-run without retraining.
The decision rules mirror HYPOTHESES.md exactly.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from recommender.evaluation import Scorer, paired_bootstrap, top_k

H2A_MAX_GAP = -0.10  # popularity gap must sit below -10 points
H2B_BAND = 0.05  # two tower gap must sit inside +/- 5 points
BOOTSTRAP_SAMPLES = 2000


# ------------------------------------------------------------------ run storage


def run_path(run_dir: Path, model: str, exponent: float, seed: int | None) -> Path:
    return Path(run_dir) / f"{model}__p{exponent:g}__s{'na' if seed is None else seed}.npz"


def save_run(
    run_dir: Path, model: str, exponent: float, seed: int | None, summary: dict, per_user: dict[str, np.ndarray], **extra
) -> None:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    meta = {"model": model, "pop_exponent": exponent, "seed": seed, **extra, **summary}
    np.savez_compressed(run_path(run_dir, model, exponent, seed), meta=json.dumps(meta), **per_user)


def load_run(run_dir: Path, model: str, exponent: float, seed: int | None) -> dict[str, np.ndarray] | None:
    path = run_path(run_dir, model, exponent, seed)
    if not path.exists():
        return None
    npz = np.load(path, allow_pickle=False)
    return {key: npz[key] for key in npz.files if key != "meta"}


def all_runs_table(run_dir: Path) -> pd.DataFrame:
    rows = [json.loads(str(np.load(p, allow_pickle=False)["meta"])) for p in sorted(Path(run_dir).glob("*.npz"))]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["pop_exponent", "model", "seed"], na_position="first").reset_index(drop=True)


def _seed_mean(run_dir: Path, model: str, exponent: float, seeds: list[int], key: str) -> np.ndarray | None:
    arrays = [load_run(run_dir, model, exponent, s) for s in seeds]
    if any(a is None for a in arrays):
        return None
    return np.mean([a[key] for a in arrays], axis=0)


def _interval(values: tuple[float, float, float]) -> dict[str, float]:
    return {"mean": values[0], "ci_low": values[1], "ci_high": values[2]}


# ------------------------------------------------------------------ hypotheses


def check_h1(run_dir: Path, exponent: float, seeds: list[int]) -> dict:
    """Full model versus the text-ablated model, per-viewer metrics averaged over seeds."""
    result: dict = {"exponent": exponent, "seeds": seeds}
    for metric in ("recall", "ndcg"):
        full = _seed_mean(run_dir, "two_tower", exponent, seeds, metric)
        no_text = _seed_mean(run_dir, "two_tower_no_text", exponent, seeds, metric)
        if full is None or no_text is None:
            return {**result, "verdict": "not run", "reason": "two tower runs missing"}
        result[f"{metric}@10_full"] = float(full.mean())
        result[f"{metric}@10_no_text"] = float(no_text.mean())
        result[f"{metric}@10_difference"] = _interval(paired_bootstrap(full - no_text, BOOTSTRAP_SAMPLES))
    per_seed = []
    for s in seeds:
        per_seed.append(
            {
                "seed": s,
                "recall@10_full": float(load_run(run_dir, "two_tower", exponent, s)["recall"].mean()),
                "recall@10_no_text": float(load_run(run_dir, "two_tower_no_text", exponent, s)["recall"].mean()),
            }
        )
    result["per_seed"] = per_seed
    diff = result["recall@10_difference"]
    if diff["ci_low"] > 0:
        result["verdict"] = "supported"
    elif diff["ci_high"] < 0:
        result["verdict"] = "rejected: text hurts"
    else:
        result["verdict"] = "inconclusive"
    return result


def _gap(run_dir: Path, model: str, exponent: float, seeds: list[int | None]) -> dict | None:
    non_en = _seed_mean(run_dir, model, exponent, seeds, "non_en")
    if non_en is None:
        return None
    heldout = load_run(run_dir, model, exponent, seeds[0])["heldout_non_en"]
    return {
        "non_en@10": float(np.nanmean(non_en)),
        "heldout_non_en": float(np.nanmean(heldout)),
        "gap": _interval(paired_bootstrap(non_en - heldout, BOOTSTRAP_SAMPLES)),
    }


def check_h2(run_dir: Path, exponents: list[float], main_exponent: float, seeds: list[int]) -> dict:
    rows = []
    for p in exponents:
        tt_seeds = seeds if p == main_exponent else seeds[:1]
        row = {"exponent": p}
        for model, model_seeds in (
            ("popularity", [None]),
            ("most_watched", [None]),
            ("content_knn", [None]),
            ("two_tower", tt_seeds),
        ):
            row[model] = _gap(run_dir, model, p, model_seeds)
        row["two_tower_seeds"] = tt_seeds
        rows.append(row)

    pop_ok = [r["popularity"] is not None and r["popularity"]["gap"]["ci_high"] < H2A_MAX_GAP for r in rows]
    h2a = "supported" if all(pop_ok) else "not supported"

    if any(r["two_tower"] is None for r in rows):
        h2b = "not run"
        trend = None
    else:
        tt_ok = [abs(r["two_tower"]["gap"]["ci_low"]) < H2B_BAND and abs(r["two_tower"]["gap"]["ci_high"]) < H2B_BAND for r in rows]
        h2b = "supported" if all(tt_ok) else "not supported"
        gaps = [r["two_tower"]["gap"]["mean"] for r in sorted(rows, key=lambda r: r["exponent"])]
        trend = "falls as p rises (learned the popularity bias)" if all(np.diff(gaps) < 0) else "does not fall steadily as p rises"
    return {
        "rule_h2a": f"popularity gap upper bound below {H2A_MAX_GAP * 100:+.0f} points at every p",
        "rule_h2b": f"two tower gap interval inside +/-{H2B_BAND * 100:.0f} points at every p",
        "by_exponent": rows,
        "h2a_verdict": h2a,
        "h2b_verdict": h2b,
        "two_tower_gap_trend": trend,
    }


# ------------------------------------------------------------------ sample recommendations


def sample_recommendations(
    items: pd.DataFrame, demo: pd.DataFrame, scorers: dict[str, Scorer], k: int = 10
) -> tuple[pd.DataFrame, str]:
    index = {it: i for i, it in enumerate(items["item_id"])}
    users = list(dict.fromkeys(demo["user_id"]))
    rows_idx = demo["user_id"].map({u: i for i, u in enumerate(users)}).to_numpy()
    cols_idx = demo["item_id"].map(index).to_numpy()
    history = sparse.csr_matrix(
        (np.ones(len(rows_idx), dtype=np.float32), (rows_idx, cols_idx)), shape=(len(users), len(items))
    )

    records, lines = [], ["# Sample recommendations", "", f"Top {k} per demo viewer. Watched titles are excluded.", ""]
    for name, scorer in scorers.items():
        recs = top_k(scorer(history), history, k)
        for u, user in enumerate(users):
            seeds = items.iloc[history[u].indices]
            picked = items.iloc[recs[u]]
            seed_genres = Counter(g for gs in seeds["genres"] for g in gs)
            rec_genres = Counter(g for gs in picked["genres"] for g in gs)
            overlap = sum(min(seed_genres[g], rec_genres[g]) for g in seed_genres) / max(1, sum(rec_genres.values()))
            lines += [f"## {name} / {user}", "", "Seeds: " + "; ".join(seeds["title"]) + ".", ""]
            table = picked[["title", "type", "language", "release_year", "vote_count"]].copy()
            table.insert(0, "rank", range(1, len(table) + 1))
            table["genres"] = picked["genres"].map(lambda g: ", ".join(g[:3]))
            lines += [markdown_table(table), ""]
            lines += [
                f"Non-English: {int((picked['language'] != 'en').sum())}/{k}. TV: {int((picked['type'] == 'tv').sum())}/{k}. "
                f"Median votes: {int(picked['vote_count'].median())}. Genre overlap with seeds: {overlap:.0%}.",
                "",
            ]
            for rank, (_, row) in enumerate(table.iterrows(), start=1):
                records.append({"model": name, "user": user, **row.to_dict(), "rank": rank})
    return pd.DataFrame(records), "\n".join(lines)


# ------------------------------------------------------------------ reporting


def markdown_table(frame: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else floatfmt.format(value)
        return str(value)

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def _pts(interval: dict | None) -> str:
    if interval is None:
        return "not run"
    return f"{interval['mean'] * 100:+.1f} pts [{interval['ci_low'] * 100:+.1f}, {interval['ci_high'] * 100:+.1f}]"


def results_summary(runs: pd.DataFrame, h1: dict, h2: dict) -> str:
    lines = [
        "# Results summary (generated)",
        "",
        "Generated by `run_experiment.py`. All numbers come from simulated viewers and measure how well a",
        "model recovers the simulator's rules, not quality for real viewers. Interpretation is in RESULTS.md.",
        "",
        "## All runs",
        "",
        markdown_table(runs.drop(columns=[c for c in ("loss_history",) if c in runs.columns])),
        "",
        "## H1: text beyond genre",
        "",
        f"Verdict: **{h1['verdict']}**",
        "",
    ]
    if "recall@10_difference" in h1:
        lines += [
            f"Recall@10 full {h1['recall@10_full']:.4f}, no text {h1['recall@10_no_text']:.4f}. "
            f"Difference {_pts(h1['recall@10_difference'])}.",
            f"NDCG@10 difference {_pts(h1['ndcg@10_difference'])}.",
            "",
            markdown_table(pd.DataFrame(h1["per_seed"]), "{:.4f}"),
            "",
        ]
    lines += [
        "## H2: non-English gap against each viewer's held-out watches",
        "",
        f"H2a ({h2['rule_h2a']}): **{h2['h2a_verdict']}**",
        f"H2b ({h2['rule_h2b']}): **{h2['h2b_verdict']}**",
        f"Two tower gap trend: {h2['two_tower_gap_trend']}",
        "",
    ]
    table = []
    for row in h2["by_exponent"]:
        ref = next((row[m]["heldout_non_en"] for m in ("popularity", "two_tower") if row[m]), float("nan"))
        table.append(
            {
                "p": f"{row['exponent']:g}",
                "held-out non-English": f"{ref:.1%}",
                **{m: _pts(row[m]["gap"] if row[m] else None) for m in ("popularity", "most_watched", "content_knn", "two_tower")},
            }
        )
    lines += [markdown_table(pd.DataFrame(table)), ""]
    return "\n".join(lines)


def save_figures(h1: dict, h2: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"popularity": "#9ca3af", "most_watched": "#d97706", "content_knn": "#4a90c2", "two_tower": "#1f4e79"}
    rows = sorted(h2["by_exponent"], key=lambda r: r["exponent"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.2
    for i, model in enumerate(colours):
        xs, ys, errs = [], [], []
        for j, row in enumerate(rows):
            if row[model] is None:
                continue
            gap = row[model]["gap"]
            xs.append(j + (i - 1.5) * width)
            ys.append(gap["mean"] * 100)
            errs.append([[(gap["mean"] - gap["ci_low"]) * 100], [(gap["ci_high"] - gap["mean"]) * 100]])
        if xs:
            yerr = np.hstack(errs) if errs else None
            ax.bar(xs, ys, width, yerr=yerr, color=colours[model], label=model.replace("_", " "), capsize=3)
    ax.axhspan(-H2B_BAND * 100, H2B_BAND * 100, color="#16a34a", alpha=0.12, label="H2b band")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(rows)), [f"p = {r['exponent']:g}" for r in rows])
    ax.set(ylabel="Non-English share of top 10 minus held-out share (pts)", title="H2: who under-serves non-English titles")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(Path(output) / "h2_non_english_gap.png", dpi=160)
    plt.close()

    if "per_seed" not in h1:
        return
    seeds = pd.DataFrame(h1["per_seed"])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for col, colour, label in (("recall@10_full", "#1f4e79", "with text"), ("recall@10_no_text", "#d97706", "no text")):
        ax.scatter([label] * len(seeds), seeds[col], color=colour, s=40, zorder=3)
        ax.hlines(seeds[col].mean(), -0.2 if label == "with text" else 0.8, 0.2 if label == "with text" else 1.2, color=colour)
    diff = h1["recall@10_difference"]
    ax.set(
        ylabel="Recall@10 per seed",
        title=f"H1: difference {diff['mean'] * 100:+.2f} pts, 95% CI [{diff['ci_low'] * 100:+.2f}, {diff['ci_high'] * 100:+.2f}]",
    )
    plt.tight_layout()
    plt.savefig(Path(output) / "h1_text_ablation.png", dpi=160)
    plt.close()
