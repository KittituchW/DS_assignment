"""Stage 5: hypothesis tables and figures from the result CSVs.

Run: python scripts/analyse.py
Writes reports/results.md and reports/figures/h2_non_english_share.png
Works with baselines alone; adds the two tower rows when results_two_tower.csv exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_catalogue  # noqa: E402

REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
K = 10


def load_results() -> pd.DataFrame:
    frames = [pd.read_csv(REPORTS / "results_baselines.csv")]
    path = REPORTS / "results_two_tower.csv"
    if path.exists():
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    if "seed" not in df:
        df["seed"] = 42
    df["seed"] = df["seed"].fillna(42).astype(int)
    return df


def h1_table(df: pd.DataFrame, exponent: float) -> tuple[str, str]:
    sub = df[(df.pop_exponent == exponent) & df.model.isin(["two_tower", "two_tower_no_text", "content_knn", "content_knn_no_text"])]
    if sub.empty:
        return "Two tower results not available yet.", "pending"
    cols = [f"recall@{K}", f"ndcg@{K}", f"coverage@{K}"]
    agg = sub.groupby("model")[cols].agg(["mean", "min", "max"])
    lines = ["| model | " + " | ".join(f"{c} mean (min to max)" for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for model in ["two_tower", "two_tower_no_text", "content_knn", "content_knn_no_text"]:
        if model not in agg.index:
            continue
        cells = [f"{agg.loc[model, (c, 'mean')]:.3f} ({agg.loc[model, (c, 'min')]:.3f} to {agg.loc[model, (c, 'max')]:.3f})" for c in cols]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    verdict = "pending"
    if {"two_tower", "two_tower_no_text"} <= set(agg.index):
        full = sub[sub.model == "two_tower"][f"recall@{K}"]
        abl = sub[sub.model == "two_tower_no_text"][f"recall@{K}"]
        spread = max(full.max() - full.min(), abl.max() - abl.min())
        gap = full.mean() - abl.mean()
        if gap > spread:
            verdict = f"supported: text adds {gap:.3f} recall@{K} ({gap / abl.mean():+.0%}), larger than the seed spread of {spread:.3f}"
        elif gap < -spread:
            verdict = f"reversed: removing text improves recall@{K} by {-gap:.3f}, larger than the seed spread of {spread:.3f}"
        else:
            verdict = f"not supported: gap of {gap:+.3f} recall@{K} sits inside the seed spread of {spread:.3f}"
    return "\n".join(lines), verdict


def h2_table(df: pd.DataFrame, non_en_ref: float) -> tuple[str, pd.DataFrame]:
    sub = df[df.model.isin(["popularity", "most_watched", "content_knn", "two_tower"])]
    piv = sub.groupby(["model", "pop_exponent"])[[f"non_en_share@{K}", f"coverage@{K}", f"tv_share@{K}", f"recall@{K}"]].mean()
    lines = [f"Reference: {non_en_ref:.0%} of recommendable titles are non-English.", "",
             "| model | p | non-English share@10 | coverage@10 | TV share@10 | recall@10 |", "|---|---|---|---|---|---|"]
    for (model, p), row in piv.iterrows():
        lines.append(f"| {model} | {p:g} | {row[f'non_en_share@{K}']:.1%} | {row[f'coverage@{K}']:.1%} | {row[f'tv_share@{K}']:.0%} | {row[f'recall@{K}']:.3f} |")
    return "\n".join(lines), piv.reset_index()


def h2_figure(piv: pd.DataFrame, non_en_ref: float) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = {"popularity": "#C44E52", "most_watched": "#8172B2", "content_knn": "#DD8452", "two_tower": "#4C72B0"}
    for model, grp in piv.groupby("model"):
        grp = grp.sort_values("pop_exponent")
        ax.plot(grp.pop_exponent, grp[f"non_en_share@{K}"], marker="o", label=model, color=colors.get(model))
    ax.axhline(non_en_ref, ls="--", color="grey", label="catalogue share")
    ax.set_xlabel("popularity exponent p (how herd-driven simulated viewers are)")
    ax.set_ylabel("non-English share of top 10 slots")
    ax.set_ylim(0, 0.8)
    ax.set_title("H2: who gets to see non-English titles")
    ax.legend(loc="upper center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "h2_non_english_share.png", dpi=150)
    plt.close(fig)


def main() -> None:
    cat = load_catalogue(ROOT / "data" / "processed" / "catalogue.csv")
    non_en_ref = 1 - cat[cat.recommendable].is_english.mean()
    df = load_results()
    main_p = 6.0

    h1, verdict = h1_table(df, main_p)
    h2, piv = h2_table(df, non_en_ref)
    h2_figure(piv, non_en_ref)

    lines = ["# Results", "",
             "All metrics are measured on simulated viewers. They show how well each model recovers the simulator's rules, not real viewer taste.", "",
             f"## H1: text beyond genre (p = {main_p:g}, two seeds)", "", h1, "", f"Verdict: {verdict}.", "",
             "## H2: popularity bias against non-English titles", "", h2, "",
             "Figure: `reports/figures/h2_non_english_share.png`.", ""]
    full = df.sort_values(["pop_exponent", "model", "seed"])
    lines += ["## All runs", "", full.round(4).to_markdown(index=False), ""]
    (REPORTS / "results.md").write_text("\n".join(lines))
    print("\n".join(lines[:25]))


if __name__ == "__main__":
    main()
