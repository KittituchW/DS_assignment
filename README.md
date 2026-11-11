# Telecom content recommender assignment

This project implements a reproducible catalog-only recommendation experiment for the
provided Netflix/TMDb metadata. The source data has no customer interactions, so the
experiment deliberately avoids claiming that it is collaborative filtering.

## What is compared

1. Global popularity and confidence-adjusted rating
2. TF-IDF content similarity
3. Truncated-SVD semantic content similarity
4. A PyTorch neural projection model trained with triplet metric learning
5. The neural model with Maximal Marginal Relevance diversity reranking

The cold-start proxy evaluation uses titles released in 2025 as candidates. Synthetic
profiles are formed from related pre-2025 titles, and the experiment measures whether
the held-out 2025 title is retrieved. These are weak metadata labels, not customer
preference labels.

## Data split

- Training: 2010-2023
- Validation period: 2024
- Cold-start test candidates: 2025

Vocabulary and SVD transformations are fitted on training rows only. Current popularity,
vote counts and ratings are excluded from the learned retrieval embeddings because they
would leak future information into a temporal evaluation.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py \
  --movies /path/to/netflix_movies_detailed_up_to_2025.csv \
  --tv /path/to/netflix_tv_shows_detailed_up_to_2025.csv \
  --output artifacts
```

The run produces:

- `eda_summary.json`
- `eda_missingness.png`
- `eda_year_counts.png`
- `model_metrics.csv`
- `model_comparison.png`
- `sample_recommendations.csv`
- the cleaned catalog index, fitted feature transforms, neural weights and item embeddings

The checked-in `artifacts/` directory contains a completed seed-42 run. See
`RESULTS.md` for the comparison and research interpretation.

## Notebook and presentation

- `notebooks/01_eda_and_recommender.ipynb` is an executed walkthrough of the
  cleaning decisions, EDA, temporal proxy design, model comparison and sample
  recommendations. Set `RUN_TRAINING = True` to rerun the full experiment.
- The submission package places the final eight-slide PowerPoint deck under
  `slide/`. Its quantitative charts remain editable PowerPoint charts.

## Interpretation boundary

The experiment compares content-retrieval approaches under a transparent proxy task.
Production telecom personalization requires impression, play, completion, abandonment,
search and entitlement events. Those signals would support a learned user tower and an
online ranking model; they are not present in this assignment dataset.
