# Results

Final run: 17 November 2026, laptop CPU. 3,000 simulated viewers per dataset,
popularity bias p = 2, 6 and 12, two tower trained for 10 epochs (5 seeds at
p = 6, seed 42 elsewhere). The input files matched the pinned SHA-256 hashes.

Catalogue: 25,280 recommendable titles (at least 1 vote, a genre and a
description; 41.5% TV, 49.3% non-English). The same experiment with a 5-vote
floor (21,700 titles) is the sensitivity check in `artifacts/sensitivity_min5/`.
Why the floor changed after the first runs is logged in
[`HYPOTHESES.md`](HYPOTHESES.md) (2026-11-17).

All numbers come from simulated viewers. They measure how well a model
recovers the simulator's rules, not quality for real viewers. Generated tables
are in `artifacts/results_summary.md`.

**Repeat runs.** Simulated events, baselines, the oracle and the EDA are exact
from run to run. Two tower training is not bit-for-bit repeatable on CPU: the
same seed and settings can differ by up to about 0.1 points of Recall@10.

## Summary

| Hypothesis | Pre-registered rule | Main run (1 vote) | 5-vote check |
|---|---|---|---|
| H1: text adds signal beyond genre | 95% CI of the Recall@10 difference (with text minus without) above 0 | −0.13 pts [−0.31, +0.03]: **Inconclusive** | −0.20 pts [−0.39, −0.00]: **Rejected (text hurts)** |
| H2a: popularity hides non-English titles | Popularity gap below −10 pts at every p | −46, −42 and −41 pts: **Supported** | −43, −38 and −36 pts: **Supported** |
| H2b: two tower matches what viewers watch | Two tower gap CI inside ±5 pts at every p | Fails at p = 2: −4.7 [−6.0, −3.5]: **Not supported** | Fails at p = 12: +5.0 [+3.7, +6.1]: **Not supported** |

Both runs tell the same story: popularity ranking hides non-English titles,
the two tower comes close to what viewers watch but not within ±5 points at
every p, and the text features do not help.

## The ceiling: how much is there to find?

Each viewer's watches are random draws from the simulator's watch chances, so
no model can find every held-out title. The oracle ranks titles by each
viewer's true watch chance, the exact rule that produced the watches. It is the
ceiling for any model on this data.

| p | Oracle | Oracle without text | Two tower | Content kNN | Random |
|---|---|---|---|---|---|
| 2 | 0.042 | 0.022 | 0.012 (29% of ceiling) | 0.010 (23%) | 0.0004 |
| 6 | 0.069 | 0.044 | 0.025 (36%) | 0.015 (21%) | 0.0004 |
| 12 | 0.111 | 0.078 | 0.048 (43%) | 0.020 (18%) | 0.0004 |

Recall@10. Random is 10 picks out of 25,280 titles. The 5-vote check gives
the same picture: two tower 32%, 37% and 45% of the ceiling, content kNN 25%,
24% and 19%.

1. The two tower reaches about a third of what is reachable, against a fifth
   for content kNN. The absolute numbers look small because the ceiling is
   small: even the oracle finds only 7% of held-out titles at p = 6.
2. The oracle without its favourite-title term loses 47%, 37% and 30% of the
   ceiling (46%, 34% and 23% in the 5-vote check). **The text taste is a strong signal in the simulator.** This
   settles the open question from H1: the signal is there, and the model does
   not use it.

## H1: text beyond genre

Removing the text features did not lower Recall@10. The difference is −0.13
points with a 95% interval of −0.31 to +0.03: inconclusive under the rule.
NDCG@10 moves the same way (−0.15 points, interval −0.29 to −0.00). In the
5-vote check the Recall@10 interval ends at −0.00, so the rule reads "text
hurts". Both runs point the same way. The size of the effect is close to the
run to run noise of training, so the safe reading is "no benefit from text".

| Seed | Recall@10 with text | Recall@10 without text |
|---|---|---|
| 42 | 0.0249 | 0.0264 |
| 43 | 0.0240 | 0.0247 |
| 44 | 0.0248 | 0.0276 |
| 45 | 0.0257 | 0.0259 |
| 46 | 0.0245 | 0.0260 |

The non-learning baselines agree: content kNN scores 0.015 with text, 0.016
without, and 0.004 on text alone.

**Why.** The ceiling shows the text taste carries 37% of the reachable recall
at p = 6, so the problem is the model, not the data. Two likely causes:

1. The simulator's text taste is "titles like one favourite title". The viewer
   side of the model is the average of everything watched, which blurs one
   title's signal into the mix.
2. A 128-dimension SVD of TF-IDF is a coarse text representation.

**Conclusion.** Do not invest more in TF-IDF text features in this form. The
next experiments are a viewer side that keeps single-title similarity (for
example scoring against the closest watched title, or attention over the
history) and a pretrained sentence encoder for the text.

## H2: non-English titles

**H2a is supported in both runs.** The popularity ranking shows 41 to 46
points fewer non-English titles than viewers watch (36 to 43 in the 5-vote
check). It is a language effect, not only a TV effect: the type-balanced
popularity baseline still misses by 22 to 26 points, and within TV titles the
popularity list is 0.6% non-English against 59.9% in viewers' held-out TV
watches (p = 6).

**H2b is not supported in either run, and it fails in different places.**

| p | Main run gap | 5-vote check gap |
|---|---|---|
| 2 | −4.7 [−6.0, −3.5] | +0.0 [−1.2, +1.3] |
| 6 | −2.1 [−3.3, −0.9] | −0.4 [−1.6, +0.8] |
| 12 | +2.6 [+1.4, +3.7] | +5.0 [+3.7, +6.1] |

The two tower stays within 5 points on average everywhere, far closer than
any baseline (popularity −36 to −46, content kNN −25 to −27, most watched −11
to −36). But the gap rises with p in both runs: the model under-serves
non-English titles when viewers follow the crowd least and over-serves them
when viewers follow it most. The rule asks for the interval inside ±5 points
at every p, and that does not hold.

Within each content type (main run, p = 6, exploratory):

| | TV share of top 10 | Non-English within movies | Non-English within TV |
|---|---|---|---|
| Two tower | 43.3% | 27.6% | 57.5% |
| Held-out watches | 35.8% | 32.8% | 59.9% |
| Popularity | 90.0% | 0.3% | 0.6% |

**Conclusion.** Popularity-driven ranking would hide most non-English titles.
The two tower largely removes that gap, but its balance drifts with how herd
driven viewers are, so it is not reliable enough to skip a non-English
exposure guardrail in production.

## Retrieval quality against the baselines

Secondary results, main run. Two tower at p = 6 is the mean of 5 seeds. Hit@10
is the share of viewers with at least one held-out title in their top 10.

| Model | p | Recall@10 | NDCG@10 | Hit@10 | Coverage@10 | Non-English@10 | TV@10 |
|---|---|---|---|---|---|---|---|
| Two tower | 2 | 0.012 | 0.010 | 6.6% | 7.3% | 41.5% | 37.5% |
| Content kNN | 2 | 0.010 | 0.008 | 4.9% | 10.9% | 21.3% | 42.3% |
| Two tower | 6 | 0.025 | 0.020 | 12.1% | 5.2% | 40.5% | 43.3% |
| Content kNN | 6 | 0.015 | 0.012 | 7.6% | 9.4% | 15.3% | 40.0% |
| Most watched | 6 | 0.008 | 0.007 | 3.4% | 0.1% | 29.5% | 50.2% |
| Popularity | 6 | 0.002 | 0.002 | 1.2% | 0.1% | 0.6% | 90.0% |
| Two tower | 12 | 0.048 | 0.039 | 20.8% | 4.0% | 44.5% | 42.9% |
| Content kNN | 12 | 0.020 | 0.017 | 9.9% | 8.2% | 15.1% | 41.0% |

Paired bootstrap, two tower minus content kNN:

| p | Recall@10 | Hit@10 | Watch-weighted Recall@10 |
|---|---|---|---|
| 2 | +0.26 pts [+0.01, +0.49] | +1.7 pts [+0.6, +2.8] | +0.29 pts [+0.01, +0.57] |
| 6 | +1.00 pts [+0.74, +1.28] | +4.5 pts [+3.4, +5.7] | +1.17 pts [+0.85, +1.51] |
| 12 | +2.81 pts [+2.38, +3.24] | +11.0 pts [+9.4, +12.5] | +3.19 pts [+2.71, +3.68] |

The two tower beats content kNN at every p on every measure, and the gain
grows with herd behaviour. The 5-vote check gives the same picture (Hit@10
+2.4, +4.7 and +11.7 points). The cost is catalogue coverage: 5.2% of titles
appear in anyone's top 10 against 9.4% for content kNN.

Watch-weighted Recall@10 counts each held-out title by its simulated watch
ratio. Every model finds finished titles slightly more often (ratio 1.11 to
1.16 at p = 6) and the order of models does not change. The watch ratio is a
rule we wrote, so this shows consistency with that rule, not that real viewers
would finish these titles.

## Item cold start (exploratory)

2,528 titles (10%, split by type) were hidden from training entirely: not a
single watch, and not even as negatives in the softmax. For the 2,636 viewers
who watched one of them, each model ranks only the hidden titles.

| Model | Recall@10 | Hit@10 | NDCG@10 |
|---|---|---|---|
| Two tower (5 seeds) | 0.184 | 40.1% | 0.123 |
| Content kNN | 0.111 | 27.9% | 0.078 |
| Popularity | 0.019 | 5.8% | 0.013 |
| Random | 0.004 | | |

Two tower minus content kNN: Recall@10 +7.3 points [+6.3, +8.2], Hit@10 +12.2
points [+10.5, +13.9]. **The item tower turns a new title's metadata into a
useful vector**: it was trained only on other titles, and it still ranks new
titles better than direct feature similarity. This is the evidence for the
cold-start claim. The numbers are higher than in the main table because each
viewer chooses among 2,528 titles instead of 25,280.

## Settings search (exploratory)

The model settings were standard defaults fixed before training. A random
search tried 20 settings at p = 6 on a validation split cut from each viewer's
training events (activation ReLU, GELU or SiLU, output size, hidden size,
dropout, learning rate, temperature). It chose GELU, 512 hidden units, learning
rate 0.0005 and temperature 0.05: validation Recall@10 0.0283 against 0.0246
for the defaults.

On the test set the chosen settings were not better: −0.07 points
[−0.21, +0.06] at p = 6 and −0.22 points [−0.41, −0.02] at p = 2. The
validation gain came from one seed and one split, and it did not hold. The
main results keep the defaults. The H1 rule applied to the tuned model gives
the same verdict: inconclusive, +0.01 points [−0.19, +0.21]. The first run of
this search (15 November catalogue) kept the defaults outright. Together this
says the choice of activation and size makes no reliable difference here.

## Sample recommendations

Three hand-seeded viewers. Full lists in `artifacts/sample_recommendations.md`.

1. **Action and science fiction** (Inception, John Wick, Mad Max: Fury Road,
   Extraction, Edge of Tomorrow): Green Lantern, Captain America: The Winter
   Soldier, Maze Runner: The Death Cure, X-Men: Days of Future Past, The Hunger
   Games: Catching Fire. All English movies, all action and science fiction,
   close to the head of the catalogue (median 14,018 votes).
2. **Korean drama** (Crash Landing on You, Squid Game, Itaewon Class,
   Vincenzo, Parasite, Our Beloved Summer): ten Korean comedy-dramas,
   including very low-vote titles such as Smile Again (2 votes). The model has
   learned language and genre as one taste. The thriller side of the seeds
   (Parasite, Squid Game) is lost, because averaging the seeds collapses mixed
   tastes into one point. This is the same weakness H1 points to.
3. **Documentary** (Our Planet, Free Solo, The Social Dilemma, Chef's Table,
   My Octopus Teacher, Making a Murderer): Eli Roth's History of Horror,
   Abandoned Engineering, Peaky Blinders: The True Story and other
   documentaries, 8 of them TV, median 15 votes. The documentary pool is small,
   so the model reaches the tail. Good for discovery; some titles are thin.

## Strengths

1. Decision rules were committed before training, and every later change is
   logged with its reason, including the ones made after seeing results.
2. The two tower beats content kNN at every p on Recall@10 and Hit@10, in both
   catalogues, and reaches 36% of the reachable ceiling against 21%.
3. On titles it never saw in training, it beats content kNN by 7.3 points of
   Recall@10. The cold-start claim has its own test.
4. It serves non-English titles within about 5 points of what viewers watch,
   where popularity and content kNN under-serve them by 25 to 46 points.
5. The user tower has no parameters, so a new viewer is served after one watch.
6. One command reproduces every number from pinned, hash-checked inputs.

## Weaknesses

1. Text features add nothing (H1), even though the ceiling shows the text taste
   carries a third of the reachable recall.
2. The non-English balance drifts with herd behaviour, so H2b fails.
3. Catalogue coverage is about half that of content kNN.
4. Mean pooling collapses mixed tastes into one vector.
5. The simulator defines the ground truth, including the watch ratio, and
   nothing here estimates real viewer lift.
6. Two tower training is not bit-for-bit repeatable on CPU; gaps under about
   0.1 points of Recall@10 are noise.
7. The keyword filter is not a maturity system.

## Next steps

1. Viewer side that keeps single-title similarity (closest watched title,
   attention or one vector per interest); measure it against the ceiling.
2. Pretrained sentence encoder for the text block, then rerun H1.
3. Coverage: a diversity re-rank such as MMR.
4. A non-English exposure guardrail in serving, since the model's balance is
   not stable across viewer behaviour.
5. With real logs: replace the simulator, then run an online A/B test on watch
   completion with non-English exposure as a guardrail.
