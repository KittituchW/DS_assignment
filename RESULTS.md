# Results

Final run: 16 November 2026, laptop CPU. 3,000 simulated viewers per dataset,
popularity bias p = 2, 6 and 12, two tower trained for 10 epochs (5 seeds at
p = 6, seed 42 elsewhere). The input files matched the pinned SHA-256 hashes.
The run took about 27 minutes: under a minute to prepare, 7.5 minutes for the
main training and 19 minutes for the settings search.

All numbers come from simulated viewers. They measure how well a model
recovers the simulator's rules, not quality for real viewers. Generated tables
are in `artifacts/results_summary.md`; the decision rules are in
[`HYPOTHESES.md`](HYPOTHESES.md).

**Repeat runs.** The simulated events, the baselines and the EDA are exact
from run to run. Two tower training is not bit-for-bit repeatable on this CPU:
the same seed and settings gave Recall@10 up to 0.09 points apart between
runs. The previous run (15 November) gave H1 −0.12 pts [−0.30, +0.08] and an
H2b gap at p = 2 of −3.9 pts [−5.2, −2.7]. The verdicts were the same in every
run.

## Summary

| Hypothesis | Pre-registered rule | Result | Verdict |
|---|---|---|---|
| H1: text adds signal beyond genre | 95% CI of the Recall@10 difference (with text minus without) above 0 | −0.07 pts, CI [−0.26, +0.12] | **Inconclusive** |
| H2a: popularity hides non-English titles | Popularity gap below −10 pts at every p | −41, −39 and −38 pts | **Supported** |
| H2b: two tower matches what viewers watch | Two tower gap CI inside ±5 pts at every p | p = 2: −4.3 [−5.6, −3.1]; p = 6: +1.4 [+0.2, +2.5]; p = 12: +0.4 [−0.7, +1.5] | **Not supported** |

## H1: text beyond genre

Removing the text features did not lower Recall@10. The difference is −0.07
points with a 95% interval of −0.26 to +0.12, so the test cannot tell the two
models apart. NDCG@10 gives the same picture (−0.08 points, interval −0.23 to
+0.08). The non-learning baselines agree: content kNN scores 0.017 with text
and 0.019 without, and text alone scores 0.003.

| Seed | Recall@10 with text | Recall@10 without text |
|---|---|---|
| 42 | 0.0297 | 0.0297 |
| 43 | 0.0294 | 0.0285 |
| 44 | 0.0302 | 0.0333 |
| 45 | 0.0312 | 0.0308 |
| 46 | 0.0306 | 0.0323 |

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
two tower interval sits inside ±5 points. At p = 2 the mean gap is −4.3 points,
and the interval reaches −5.6 points, outside the band. The rule is not
relaxed after the fact.

The two tower mean gap is −4.3, +1.4 and +0.4 points at p = 2, 6 and 12. It
does not fall as p rises: the model did not learn the popularity signal's bias
against non-English titles, even though popularity is one of its input
features. For comparison, popularity and content kNN miss by 26 to 41 points at
every p. Most watched misses by 21 and 29 points at p = 2 and 6, but sits at +2
points at p = 12, where herd behaviour makes the most watched titles mirror the
viewers' own mix.

Reading the result within each content type (p = 6, exploratory):

| | TV share of top 10 | Non-English within movies | Non-English within TV |
|---|---|---|---|
| Two tower | 39.5% | 27.7% | 60.5% |
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
mean of 5 seeds. Hit@10 is the share of viewers with at least one held-out
title in their top 10 (exploratory, added after the main run).

| Model | p | Recall@10 | NDCG@10 | Hit@10 | Coverage@10 | Non-English@10 | TV@10 |
|---|---|---|---|---|---|---|---|
| Two tower | 2 | 0.013 | 0.010 | 7.4% | 8.0% | 36.5% | 34.5% |
| Content kNN | 2 | 0.011 | 0.009 | 5.8% | 11.7% | 13.1% | 31.5% |
| Two tower | 6 | 0.030 | 0.025 | 14.6% | 5.8% | 40.6% | 39.5% |
| Content kNN | 6 | 0.017 | 0.015 | 8.8% | 10.2% | 12.8% | 31.8% |
| Most watched | 6 | 0.012 | 0.009 | 4.8% | 0.1% | 10.6% | 50.2% |
| Popularity | 6 | 0.002 | 0.001 | 1.1% | 0.1% | 0.7% | 90.0% |
| Two tower | 12 | 0.051 | 0.043 | 21.8% | 4.4% | 39.4% | 39.5% |
| Content kNN | 12 | 0.023 | 0.019 | 11.6% | 9.1% | 12.1% | 31.7% |

Paired bootstrap, two tower minus content kNN:

| p | Recall@10 | Hit@10 | Watch-weighted Recall@10 |
|---|---|---|---|
| 2 | +0.25 pts [+0.00, +0.51] | +1.6 pts [+0.4, +2.8] | +0.35 pts [+0.04, +0.64] |
| 6 | +1.29 pts [+0.99, +1.59] | +5.7 pts [+4.5, +7.0] | +1.55 pts [+1.21, +1.90] |
| 12 | +2.75 pts [+2.30, +3.19] | +10.2 pts [+8.5, +11.7] | +3.25 pts [+2.75, +3.78] |

The gain grows with herd behaviour. One reading is that the model learns how
much weight to give the popularity features, which content kNN cannot do. At
p = 2 the Recall@10 interval touches zero, while Hit@10 separates the two
models. The cost is catalogue coverage: 5.8% of titles appear in anyone's top
10 against 10.2% for content kNN, falling to 4.4% at p = 12.

At p = 6, 14.6% of viewers find at least one title they went on to watch in the
two tower's top 10, against 8.8% for content kNN.

## Watch-weighted Recall@10 (exploratory)

Each simulated watch has a watch ratio, the share of the title watched. Titles
that fit the viewer better are more likely to be finished (rule in
`simulate.py`). Watch-weighted Recall@10 counts each held-out title by its
watch ratio.

| Model | p | Recall@10 | Watch-weighted Recall@10 | Ratio |
|---|---|---|---|---|
| Two tower | 6 | 0.0302 | 0.0347 | 1.15 |
| Content kNN | 6 | 0.0174 | 0.0192 | 1.11 |
| Most watched | 6 | 0.0115 | 0.0133 | 1.15 |
| Popularity | 6 | 0.0022 | 0.0022 | 1.03 |

Every personalised model finds finished titles a little more often than titles
the viewer quit, and the ranking of models does not change. This is expected:
the watch ratio is a rule we wrote, and it rewards the same fit that drives
the watches. It shows the ranking is consistent with that rule. It says
nothing about whether real viewers would finish these titles.

## Settings search (exploratory)

The model settings were standard defaults fixed before training. A random
search then tried 20 settings at p = 6 (activation ReLU, GELU or SiLU, output
size, hidden size, dropout, learning rate, temperature), trained on each
viewer's earlier training events and scored on their latest training events.
The test set was not used to choose.

| Trial | Activation | Output | Hidden | Dropout | Learning rate | Temperature | Validation Recall@10 |
|---|---|---|---|---|---|---|---|
| 0 (default) | ReLU | 64 | 256 | 0.2 | 0.001 | 0.07 | 0.0303 |
| 16 | ReLU | 128 | 128 | 0.0 | 0.003 | 0.07 | 0.0301 |
| 9 | ReLU | 64 | 512 | 0.1 | 0.001 | 0.10 | 0.0301 |
| 13 | GELU | 64 | 512 | 0.4 | 0.0005 | 0.10 | 0.0297 |
| 5 | GELU | 128 | 256 | 0.0 | 0.003 | 0.10 | 0.0297 |

Full list in `artifacts/tuning_trials.csv`. The best SiLU setting scored 0.0285.

**The defaults scored highest, so the search kept them.** The gaps at the top
(0.02 to 0.06 points) are smaller than the run to run noise of training, so
the fair reading is that no setting beat the defaults, and the choice between
ReLU, GELU and SiLU makes no measurable difference here. In a random search the
activation also varies together with the other settings, so this is not a
clean activation comparison.

Because the defaults won, the "tuned" runs repeat the pre-registered model with
the same seeds. That gives a direct measure of training noise. At p = 6 the two
identical models differ by −0.05 points of Recall@10, and the paired bootstrap
over viewers puts the interval at [−0.09, −0.02], away from zero. Viewer-level
intervals do not include training randomness, so a gap under about 0.1 points
of Recall@10 between two trained models should not be read as real. The H1
rule applied to the repeat gives the same verdict: inconclusive, −0.12 pts
[−0.31, +0.08].

## Sample recommendations

Three hand-seeded viewers. Full lists in `artifacts/sample_recommendations.md`.

1. **Action and science fiction** (Inception, John Wick, Mad Max: Fury Road,
   Extraction, Edge of Tomorrow): Tenet, Battleship, Predators, Maze Runner:
   The Scorch Trials, Infinite. Nine English movies and one French, all action
   and science fiction. Coherent, and close to the head of the catalogue
   (median 6,460 votes).
2. **Korean drama** (Crash Landing on You, Squid Game, Itaewon Class,
   Vincenzo, Parasite, Our Beloved Summer): ten Korean comedy-dramas,
   including low-vote titles such as Sweet Revenge (36 votes). The model has
   learned language and genre as one taste. The thriller side of the seeds
   (Parasite, Squid Game) is lost, because averaging the seeds collapses mixed
   tastes into one point.
3. **Documentary** (Our Planet, Free Solo, The Social Dilemma, Chef's Table,
   My Octopus Teacher, Making a Murderer): Drain the Oceans, Hamilton's
   Pharmacopeia, What on Earth? and other documentaries, 7 of them TV, median
   29 votes. The documentary pool is small, so the model reaches the tail. Good
   for discovery; some titles are thin.

The content kNN baseline recommended two adult-industry documentaries to the
documentary viewer. They pass the keyword filter because they have more than
100 votes. This confirms that a keyword list is not a maturity system.

## Strengths

1. Decision rules were committed before training, and every later change is
   logged with its reason.
2. Five seeds and viewer-level intervals turned a pilot claim ("text hurts")
   into an honest "no detectable effect".
3. The two tower model serves non-English titles within about 4 points of what
   viewers watch, where popularity and content kNN under-serve them by 26 to 41
   points.
4. It beats content kNN on Hit@10 at every p, and on Recall@10 when viewers are
   moderately or strongly herd-driven.
5. The user tower has no parameters, so a new viewer is served after one watch.
6. A 20-setting search on a validation split kept the defaults; the test set
   played no part in the choice.
7. One command reproduces every number from pinned, hash-checked inputs: about
   8 minutes for the main results, 27 with the settings search.

## Weaknesses

1. Text features add nothing measurable (H1).
2. Catalogue coverage is about half that of content kNN and falls as herd
   behaviour rises.
3. Mean pooling collapses mixed tastes into one vector.
4. The simulator defines the ground truth, including the watch ratio, its
   persona pools differ in size, and nothing here estimates real viewer lift.
5. Two tower training is not bit-for-bit repeatable on CPU, so gaps under
   about 0.1 points of Recall@10 are noise.
6. The keyword filter is not a maturity system.

## Next steps

1. Positive control for H1: a stronger text term in the simulator.
2. Pretrained sentence encoder for the text block.
3. Coverage: a diversity re-rank such as MMR, or one vector per interest
   cluster.
4. Repeatable training (deterministic kernels or one thread), or more seeds per
   comparison, so small differences can be trusted.
5. With real logs: replace the simulator, then run an online A/B test on watch
   completion with non-English exposure as a guardrail.
