# Netflix catalogue recommender: a research-first take

A content-aware two tower recommender built on the Netflix movies and TV shows
dataset (TMDb metadata up to 2025). The dataset has no users, so the project
treats that gap as the central design problem: it simulates viewers from an
explicit choice model with a tunable popularity bias, states two hypotheses up
front, and tests them with a deep learning model against two baselines.

Author: Kittituch Wongwatcharapaiboon (Top). Catalogue cleaning decisions build
on my earlier exploration of this dataset for the ML engineering assignment;
the code, simulator, model, and analysis here are new.

## How to run

Python 3.10 or newer. CPU only. About 10 minutes end to end on a laptop; training is 6 runs of about one minute each.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/prepare.py        # clean catalogue, item features, simulated events  (~1 min)
python scripts/baselines.py      # popularity and content kNN baselines              (~10 s)
python scripts/train.py          # two tower model, H1 ablation, H2 bias sweep       (~6 min)
python scripts/recommend.py      # sample recommendations for three demo users
python scripts/analyse.py        # results tables and figures for the report
```

The two raw CSV files must be downloaded into `data/raw/`. They are not stored
in this repository; the source is
<https://github.com/pimphornm-pim/DSAssignmentDataSet>.

## Layout

```
src/data.py        load both files, clean, one catalogue table   (data processing)
src/features.py    item feature matrix with named feature groups  (feature engineering)
src/simulate.py    persona based viewer simulator with a bias dial
src/model.py       two tower model and training loop (PyTorch)   (model training)
src/evaluate.py    temporal split, metrics, baselines, numpy inference (evaluation)
scripts/           one script per stage, in run order
reports/           generated tables, figures, and sample recommendations
```

Data processing and feature engineering never import torch. Training is the
only stage that does. Inference uses a saved item embedding table and numpy.

## 1. Data and exploration

See `reports/eda_summary.md` for the full output. The findings that drove the
design:

1. Two files, 16,000 rows each, one row per title. There is no user, event, or
   rating table. `popularity`, `vote_count`, and `vote_average` are global TMDb
   aggregates.
2. `show_id` collides across files (397 shared ids) and repeats inside the TV
   file (9 ids). Item key is `movie_<id>` / `tv_<id>`.
3. `rating` is a copy of `vote_average`. `duration` is null for every movie and
   constant for TV. Both dropped. Budget and revenue of 0 mean unknown (70% of
   movies).
4. The TV and movie genre vocabularies differ. TV labels such as
   `Action & Adventure` and `Sci-Fi & Fantasy` are mapped onto the movie set.
5. Exactly 1,000 titles per type and year: this is a stratified sample, not the
   catalogue.
6. 56% of titles are non-English, but English share rises with popularity (40%
   in the least popular decile, 54% in the most popular; 53% to 74% for
   movies). The raw popularity top 200 is 82% TV.
7. There is no maturity rating. A cluster of low-vote adult titles dominates
   Romance. Items with fewer than 5 votes and a bounded title keyword list are
   excluded from recommendation. The filter is known to be incomplete.

Clean catalogue: 31,915 items, 21,916 recommendable.

## 2. Research design

### Why simulate, and how

Without behaviour data there is nothing to train or score a personalised
model on. Options were (a) item to item similarity only, with no notion of a
user, (b) borrow a public interaction dataset and join on title, or (c)
simulate. (a) cannot answer any question about users. (b) joins badly: MovieLens
has few of these titles and no TV. (c) is chosen because the rules are explicit,
so every result can be traced to a rule and the limits are known.

Each simulated viewer has a persona (10, such as action, K-drama, anime,
documentary), a Dirichlet genre taste around that persona, an optional language
boost, and one "taste anchor" title. The probability of watching a title is

    genre affinity^2  x  text similarity to the anchor  x  popularity^p  x  language boost

`p` is the popularity exponent. It is the dial for hypothesis 2. Three datasets
are generated, p = 2, 6, 12, with 3,000 users and about 81,000 events each.
The text term uses sparse TF-IDF cosine, a different representation from the
SVD text features the model sees, so the model cannot read the rule back out
of its inputs.

What the metrics mean: they measure how well a model recovers the simulator's
rules from a user's past events. They are not evidence of quality for real
viewers. This caveat applies to every reported result.

### Hypothesis 1: text carries signal beyond genre tags

Why it matters: genre tags are coarse (median 2 per title) and the description
is the richest per-item field. If text adds nothing, the item tower can be a
cheap lookup table of categorical features; if it does, text encoding is where
to invest (a pretrained sentence encoder is the obvious next step).

Test: train the two tower model with and without the text feature group,
two seeds each, on the p = 6 dataset.

Decision rule: H1 holds if recall@10 and ndcg@10 of the full model beat the
text-ablated model by more than the seed to seed spread. H1 is rejected if
the gap is inside that spread.

### Hypothesis 2: popularity ranking suppresses non-English content, and a
### content-aware model narrows the gap

Why it matters: 56% of the catalogue is non-English, yet popularity rises with
English share. A platform whose home screen is driven by popularity will
under-serve more than half its catalogue, and the effect should get worse as
viewer behaviour becomes more herd driven.

Test: non_en_share@10 and coverage@10 for the popularity baseline, content kNN,
and the two tower model at p = 2, 6, 12. Reference point: 45% non-English among
recommendable titles.

Decision rule: H2a holds if the popularity baseline's non-English share sits
well below 45% at every p. H2b holds if the two tower model's share is closer
to 45% than the popularity baseline's at every p. If the two tower model's share
also collapses as p rises, the model has learned the bias instead of the
content, and that is reported as a weakness.

## 3. Model

Two tower retrieval model (PyTorch, `src/model.py`).

1. Item tower: 188 input features (genre multi-hot, language, country, type,
   era, four numeric signals, 128-d TF-IDF/SVD text) through a 256 unit MLP to a
   64-d unit vector.
2. User tower: the mean of the item vectors in the user's history, no
   parameters, no user ID table. A new user with one watched title gets a vector
   at once, and serving needs only the item table and numpy.
3. Loss: full softmax over the recommendable catalogue with temperature 0.07.
   The positive is removed from its own user's history before the query is
   formed (leave one out), so it never leaks into its own query.
4. Split: per user, the last 20% of events by time are held out.

Baselines: popularity (Bayesian weighted rating x popularity percentile, same
list for everyone) and content kNN (no learning, user = mean feature row,
cosine).

## 4. Results

All numbers come from simulated viewers and measure recovery of the
simulator's rules. Full tables in `reports/results.md`, every run in
`reports/results_two_tower.csv` and `reports/results_baselines.csv`.

### Overall (p = 6, mean of two seeds for the two tower)

| model | recall@10 mean (min to max) | ndcg@10 mean (min to max) | coverage@10 mean (min to max) |
|---|---|---|---|
| two_tower | 0.030 (0.029 to 0.031) | 0.026 (0.026 to 0.026) | 0.056 (0.055 to 0.056) |
| two_tower_no_text | 0.034 (0.033 to 0.035) | 0.027 (0.026 to 0.029) | 0.061 (0.061 to 0.061) |
| content_knn | 0.017 (0.017 to 0.017) | 0.014 (0.014 to 0.014) | 0.101 (0.101 to 0.101) |
| content_knn_no_text | 0.019 (0.019 to 0.019) | 0.018 (0.018 to 0.018) | 0.098 (0.098 to 0.098) |

The two tower model roughly doubles content kNN on recall@10 (0.030 vs 0.017)
and is 15x the popularity baseline (0.002). It trades catalogue coverage for
it: 5.6% of items appear in anyone's top 10 against 10.1% for content kNN.

### H1: text beyond genre. Result: reversed

Removing the text block improves recall@10 by 0.003 (0.034 vs 0.030), and the
seed spread is 0.002, so the gap is outside noise. The 128-d TF-IDF/SVD text
features add noise, not signal, even though the simulator does use text
similarity to pick titles. Two readings, both testable next:

1. The text representation is too weak. Sparse TF-IDF cosine (what the
   simulator uses) and a 128-d SVD of it (what the model sees) agree poorly on
   a catalogue where 20% of TV rows have no description. A pretrained
   multilingual sentence encoder is the next experiment, and the only change
   is the item tower input size.
2. The text signal is real but small next to genre, language, and popularity,
   so the optimiser spends capacity on those and text becomes a regulariser
   it would rather not have. A larger text weight in the simulator would
   separate these two readings.

Either way the decision for the next iteration is the same: do not invest in
more TF-IDF features; test an encoder.

### H2: popularity bias against non-English titles. Result: H2a and H2b both hold

| model | p | non-English share@10 | coverage@10 | TV share@10 | recall@10 |
|---|---|---|---|---|---|
| content_knn | 2 | 13.4% | 12.1% | 32% | 0.011 |
| content_knn | 6 | 12.4% | 10.1% | 31% | 0.017 |
| content_knn | 12 | 11.5% | 9.0% | 31% | 0.023 |
| most_watched | 2 | 30.2% | 0.1% | 50% | 0.003 |
| most_watched | 6 | 50.0% | 0.1% | 40% | 0.009 |
| most_watched | 12 | 31.6% | 0.1% | 68% | 0.020 |
| popularity | 2 | 0.3% | 0.1% | 90% | 0.002 |
| popularity | 6 | 0.6% | 0.1% | 90% | 0.002 |
| popularity | 12 | 1.0% | 0.1% | 90% | 0.004 |
| two_tower | 2 | 41.1% | 8.2% | 38% | 0.013 |
| two_tower | 6 | 43.7% | 5.6% | 38% | 0.030 |
| two_tower | 12 | 41.1% | 4.4% | 41% | 0.051 |

1. H2a: the TMDb popularity ranking fills 0.3% to 1% of slots with non-English
   titles and 90% with TV, at every bias level. Confirmed.
2. H2b: the two tower serves 41% to 44% non-English at every p, within four
   points of the 45% catalogue share, and the share does not fall as viewers
   become more herd driven. It does not inherit the TMDb bias through the
   popularity feature. Confirmed.
3. The cost shows up in coverage, not language: 8.2% at p = 2 down to 4.4% at
   p = 12. The model learns that herd driven viewers want the head of the
   catalogue, which is correct for recall and bad for discovery.
4. A side finding: "most watched" by simulated viewers serves 30% to 50%
   non-English, so the language bias lives in the TMDb popularity signal
   itself, not in viewing behaviour. Content kNN with no learning also drifts
   to English (12%): mean pooling a one-hot language block does that.

Figure: `reports/figures/h2_non_english_share.png`.

### Sample recommendations

Three hand seeded viewers, full lists in `reports/sample_recommendations.md`.

1. User A (Inception, Mad Max, John Wick, Extraction, Edge of Tomorrow): the
   two tower returns Jurassic World, Terminator: Dark Fate, Upgrade,
   Spider-Man: No Way Home, The Avengers. All English, all movies, median
   about 10,000 votes. Coherent, and it chases the head of the catalogue.
2. User B (Crash Landing on You, Squid Game, Itaewon Class, Vincenzo,
   Parasite, Our Beloved Summer): True Beauty, The Heirs, King the Land, The
   Good Bad Mother, Sweet & Sour. All Korean, nine TV and one film. The model
   has learned language plus genre as one taste.
3. User C (Our Planet, The Social Dilemma, Chef's Table, Making a Murderer,
   Free Solo): Drain the Oceans, Living with Leopards, Hamilton's
   Pharmacopeia, Making The Witcher. All documentary, median 16 votes. Here
   the model reaches the tail, because the documentary neighbourhood is
   small and popularity carries less weight in it. Good for discovery, but
   some of these are thin titles a real product would want to re-rank.


## 5. Strengths, weaknesses, and production notes

Strengths:
1. Every number is reproducible from two seeds and one config.
2. The user tower needs no user table, so cold start users and new titles are
   served the same way as everyone else.
3. Ablations are one flag away because feature groups are named.

Weaknesses:
1. The simulator defines the ground truth. Any model that learns genre and
   popularity will look good on it. The text ablation is the only test that
   is not circular, and even it shares a text source with the simulator.
2. No real behaviour, so no estimate of real click or watch lift.
3. The adult content filter is a keyword list. Production needs a licensed
   maturity feed.
4. Mean pooling over history has no notion of recency or sequence.

Production path:
1. Replace the simulator with logged impressions and watches behind the same
   `history matrix` interface; nothing downstream changes.
2. Swap the TF-IDF text block for a pretrained multilingual sentence encoder
   (the item tower input size is the only change).
3. Serve by approximate nearest neighbour over the item table; the user vector
   is a mean, so it can be updated online per event.
4. Evaluate online with an interleaving test on non-English exposure and
   watch completion, because offline metrics here cannot measure either.
