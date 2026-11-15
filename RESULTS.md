# Results

Final run: 15 November 2026, laptop CPU. 3,000 simulated viewers per dataset,
popularity bias p = 2, 6 and 12, two tower trained for 10 epochs (5 seeds at
p = 6, seed 42 elsewhere). The input files matched the pinned SHA-256 hashes.
The run took about 8 minutes, almost all of it training.

All numbers come from simulated viewers. They measure how well a model
recovers the simulator's rules, not quality for real viewers. Generated tables
are in `artifacts/results_summary.md`; the decision rules are in
[`HYPOTHESES.md`](HYPOTHESES.md).

## Summary

| Hypothesis | Pre-registered rule | Result | Verdict |
|---|---|---|---|
| H1: text adds signal beyond genre | 95% CI of the Recall@10 difference (with text minus without) above 0 | −0.12 pts, CI [−0.30, +0.08] | **Inconclusive** |
| H2a: popularity hides non-English titles | Popularity gap below −10 pts at every p | −41, −39 and −38 pts | **Supported** |
| H2b: two tower matches what viewers watch | Two tower gap CI inside ±5 pts at every p | p = 2: −3.9 [−5.2, −2.7]; p = 6: +1.4 [+0.2, +2.5]; p = 12: +0.9 [−0.2, +2.0] | **Not supported** |

## H1: text beyond genre

Removing the text features did not lower Recall@10. The difference is −0.12
points with a 95% interval of −0.30 to +0.08, so the test cannot tell the two
models apart. NDCG@10 gives the same picture (−0.12 points, interval −0.27 to
+0.05). The non-learning baselines agree: content kNN scores 0.017 with text
and 0.019 without, and text alone scores 0.003.

| Seed | Recall@10 with text | Recall@10 without text |
|---|---|---|
| 42 | 0.0294 | 0.0292 |
| 43 | 0.0292 | 0.0294 |
| 44 | 0.0304 | 0.0334 |
| 45 | 0.0308 | 0.0312 |
| 46 | 0.0300 | 0.0325 |

The pilot used two seeds and reported that text hurts. With five seeds and an
interval over viewers, that effect is not distinguishable from zero. A
two-seed spread was not enough evidence.

There are two readings, and this experiment cannot separate them:

1. The text signal in the simulator is small next to the genre and popularity
   terms, so there is little for the model to find.
2. A 128-dimension SVD of TF-IDF is too weak a text representation, and 20%
   of TV titles have no description.

**Conclusion.** The evidence does not support further investment in TF-IDF
text features. The next step is a positive control: rerun H1 with a stronger
text term in the simulator. If the model still shows no text effect, the
representation is the bottleneck, and a pretrained sentence encoder is the
next experiment.

## H2: non-English titles

**H2a is supported.** The popularity ranking shows 38 to 41 points fewer
non-English titles than viewers watch, at every level of popularity bias. Two
checks show this is a language effect, not only a TV effect:

1. Within TV titles, the popularity list is 0.7% non-English, while viewers'
   held-out TV watches are 55.6% non-English (p = 6).
2. The exploratory type-balanced popularity baseline still sits 29 to 31
   points below what viewers watch.

**H2b is not supported under the pre-registered rule.** At p = 6 and p = 12 the
two tower interval sits inside ±5 points. At p = 2 the mean gap is −3.9 points,
but the interval reaches −5.2 points, just outside the band. The rule is not
relaxed after the fact.

The two tower gap stays within 4 points at every p, and it does not fall as p
rises. The model did not learn the popularity signal's bias against
non-English titles, even though popularity is one of its input features. For
comparison, popularity and content kNN miss by 26 to 41 points at every p.
Most watched misses by 21 and 29 points at p = 2 and 6, but sits at +2 points
at p = 12, where herd behaviour makes the most watched titles mirror the
viewers' own mix.

Reading the result within each content type (p = 6, exploratory):

| | TV share of top 10 | Non-English within movies | Non-English within TV |
|---|---|---|---|
| Two tower | 39.6% | 27.8% | 60.2% |
| Held-out watches | 32.5% | 31.3% | 55.6% |
| Popularity | 90.0% | 0.4% | 0.7% |

The two tower model serves 7 points more TV than viewers watch. Part of its
near-zero total gap at p = 6 therefore comes from type mix: it under-serves
non-English movies a little and over-serves non-English TV a little.

**Conclusion.** Popularity-driven ranking would hide most non-English titles.
The two tower model largely removes that gap, but not within the
pre-registered band when viewers are least herd-driven.

## Retrieval quality against the baselines

Secondary results, not part of a pre-registered rule. Two tower at p = 6 is the
mean of 5 seeds.

| Model | p | Recall@10 | NDCG@10 | Coverage@10 | Non-English@10 | TV@10 |
|---|---|---|---|---|---|---|
| Two tower | 2 | 0.013 | 0.010 | 8.0% | 36.9% | 34.7% |
| Content kNN | 2 | 0.011 | 0.010 | 11.7% | 13.1% | 31.5% |
| Two tower | 6 | 0.030 | 0.025 | 5.8% | 40.6% | 39.6% |
| Content kNN | 6 | 0.017 | 0.015 | 10.2% | 12.8% | 31.8% |
| Most watched | 6 | 0.012 | 0.009 | 0.1% | 10.6% | 50.3% |
| Popularity | 6 | 0.002 | 0.002 | 0.1% | 0.7% | 90.0% |
| Two tower | 12 | 0.051 | 0.044 | 4.4% | 39.8% | 39.8% |
| Content kNN | 12 | 0.023 | 0.019 | 9.1% | 12.1% | 31.7% |

Paired bootstrap of the Recall@10 difference, two tower minus content kNN:

1. p = 2: +0.23 pts, CI [−0.03, +0.47]. Not distinguishable.
2. p = 6: +1.26 pts, CI [+0.96, +1.57].
3. p = 12: +2.77 pts, CI [+2.32, +3.20].

The gain grows with herd behaviour. One reading is that the model learns how
much weight to give the popularity features, which content kNN cannot do. The
cost is
catalogue coverage: 5.8% of titles appear in anyone's top 10 against 10.2% for
content kNN, falling to 4.4% at p = 12.

## Sample recommendations

Three hand-seeded viewers. Full lists in `artifacts/sample_recommendations.md`.

1. **Action and science fiction** (Inception, John Wick, Mad Max: Fury Road,
   Extraction, Edge of Tomorrow): Tenet, Battleship, Predators, Maze Runner:
   The Scorch Trials, Captain America: The Winter Soldier. All English
   movies, all action and science fiction. Coherent, and close to the head of
   the catalogue (median 7,000 votes).
2. **Korean drama** (Crash Landing on You, Squid Game, Itaewon Class,
   Vincenzo, Parasite, Our Beloved Summer): ten Korean comedy-dramas,
   including low-vote titles such as Sweet Revenge (36 votes). The model has
   learned language and genre as one taste. The thriller side of the seeds
   (Parasite, Squid Game) is lost, because averaging the seeds collapses mixed
   tastes into one point.
3. **Documentary** (Our Planet, Free Solo, The Social Dilemma, Chef's Table,
   My Octopus Teacher, Making a Murderer): Drain the Oceans, What on Earth?,
   An Inconvenient Sequel and other documentaries, 8 of them TV, median 15
   votes. The documentary pool is small, so the model reaches the tail. Good
   for discovery; some titles are thin.

The content kNN baseline recommended two adult-industry documentaries to the
documentary viewer. They pass the keyword filter because they have more than
100 votes. This confirms that a keyword list is not a maturity system.

## Strengths

1. Decision rules were committed before training, and every later change is
   logged with its reason.
2. Five seeds and viewer-level intervals turned a pilot claim ("text hurts")
   into an honest "no detectable effect".
3. The two tower model serves non-English titles within 4 points of what
   viewers watch, where popularity and content kNN under-serve them by 26 to
   41 points.
4. It beats content kNN when viewers are moderately or strongly herd-driven.
5. The user tower has no parameters, so a new viewer is served after one watch.
6. One command reproduces every number from pinned, hash-checked inputs in
   about 10 minutes on a laptop.

## Weaknesses

1. Text features add nothing measurable (H1).
2. Catalogue coverage is about half that of content kNN and falls as herd
   behaviour rises.
3. Mean pooling collapses mixed tastes into one vector.
4. The simulator defines the ground truth, its persona pools differ in size,
   and nothing here estimates real viewer lift.
5. The keyword filter is not a maturity system.

## Next steps

1. Positive control for H1: a stronger text term in the simulator.
2. Pretrained sentence encoder for the text block.
3. Coverage: a diversity re-rank such as MMR, or one vector per interest
   cluster.
4. Simulate watch completion (a watch ratio per event) and evaluate on
   completed watches, which is closer to a business metric.
5. With real logs: replace the simulator, then run an online A/B test on watch
   completion with non-English exposure as a guardrail.
