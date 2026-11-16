# Hypotheses

This file is written and committed before the final experiment is run.

An earlier pilot of the same two tower model used two seeds and compared H2
against the catalogue's non-English share. The pilot informed this design.
The decision rules below were fixed after the pilot and before this run.

## Setup

The dataset has no viewers. The experiment simulates 3,000 viewers per dataset
from explicit rules: a persona genre taste, one "anchor" title for text taste,
a popularity term raised to a power `p` (how herd-driven viewers are), and a
language boost for some personas. Each viewer's last 20% of watches is the
test set.

The metrics measure how well a model recovers the simulator's rules. They are
not evidence of quality for real viewers.

## H1: description text adds signal beyond genre and metadata

**Statement.** The two tower model with description text finds more of a
viewer's next watches than the same model without text.

**Why it matters.** Genre tags are coarse. If text adds nothing, the item tower
can be a cheap lookup of tags. If text helps, text encoding is where to invest
next (for example a pretrained multilingual sentence encoder).

**Test.**

1. Train the full model and a model with the text features set to zero.
2. Use the p = 6 dataset and 5 seeds per model (42 to 46).
3. Average each viewer's Recall@10 over the 5 seeds.
4. Compute the per-viewer difference (full minus no text) and a 95% confidence
   interval with a paired bootstrap over viewers (2,000 resamples).

**Decision.**

1. Supported if the whole interval is above 0.
2. Rejected, text hurts, if the whole interval is below 0.
3. Inconclusive if the interval includes 0.

NDCG@10 is reported as a secondary check.

**Why this test is fair.** Both models see the genre, language and popularity
signals that the simulator uses, so those cancel out. Only the text signal
differs. The simulator scores text with sparse TF-IDF to one anchor title. The
model sees a 128-dimension SVD of a separate TF-IDF, so it cannot copy the rule
directly.

## H2: popularity ranking hides non-English titles; the two tower does not

**Statement.** A popularity ranking shows far fewer non-English titles than
viewers watch. The two tower model shows about as many as viewers watch.

**Why it matters.** More than half of the catalogue is non-English, but
English share rises with popularity. A home screen driven by popularity would
hide much of the catalogue from viewers who want it.

**Reference point.** Each viewer's own held-out watches. The pilot used the
catalogue share instead, which depends on how many personas we gave a
non-English taste. The held-out share measures what viewers want.

**Test.**

1. Run the popularity baseline and the two tower model on the p = 2, 6 and 12
   datasets.
2. For each viewer, compute the non-English share of their top 10 minus the
   non-English share of their held-out watches.
3. Average over viewers and compute a 95% confidence interval with a paired
   bootstrap over viewers.
4. At p = 6 the two tower result is averaged over the 5 seeds from H1. At
   p = 2 and p = 12 it uses seed 42.

**Decision.**

1. H2a is supported if the popularity baseline's gap is at least 10 points
   below zero (upper end of the interval below −10 points) at every p.
2. H2b is supported if the two tower model's interval lies inside −5 to +5
   points at every p.
3. If the two tower gap falls as p rises, the model has learned the popularity
   bias. That is reported as a weakness.

Coverage@10 and TV share@10 are reported as secondary checks.

## What these results cannot show

1. Real viewer lift. That needs logged impressions and watches.
2. Whether the simulator's rules match real behaviour.

## Deviations

Changes made after the decision rules above were committed. None of them
changes a decision rule.

**2026-11-15, before any two tower training.** A second round of EDA
(`artifacts/eda_*.png`) led to three changes.

1. **Adult keyword filter fixed.** Keywords matched inside words, so "lover"
   removed *10 Cloverfield Lane* and "lust" removed *Wanderlust*. Keywords now
   match whole words, and only titles with fewer than 100 votes are checked,
   because the adult cluster sits in low-vote titles. Recommendable titles go
   from 21,917 to 22,059, so the simulated events change too.
2. **Exploratory baseline added: popularity by type.** TV titles score about
   one point higher than movies, which pushes the popularity list towards TV.
   This baseline ranks rating as a percentile within type. It checks whether
   H2a is only a type-mix effect. It is reported but is not part of any
   verdict.
3. **Exploratory reading of H2 within each content type.** TV is 58%
   non-English and movies 38%, so a list's non-English share depends on its TV
   share. The non-English share within movies and within TV is reported next to
   the held-out watches. It is not part of any verdict.

For transparency: the baseline results behind H2a were computed during
development, both before and after this change. The two tower model had not
been trained when this entry was written.

**2026-11-16, after the main run.** Two exploratory metrics added. Neither
enters a decision rule, and the H1 and H2 verdicts stand as reported.

1. **Hit@10.** 1 if at least one held-out title is in the viewer's top 10. It
   is easier to explain than Recall@10, but coarser: finding one held-out title
   or five counts the same.
2. **Watch-weighted Recall@10.** The simulator now also gives each event a
   watch ratio, the share of the title the viewer watched. It is drawn from
   Beta(1 + 5 × fit, 2), where fit is the title's rank by watch chance among
   that viewer's own picks, so a better fit is more likely to be finished.
   Recall then counts each held-out title by its watch ratio. The ratio comes
   from its own random stream, so the watch events, the split and every
   existing number are unchanged (checked against the previous simulator at
   p = 2, 6 and 12). Training never sees the ratio.

The watch ratio is a rule we wrote, like the rest of the simulator. It shows
whether the ranking also favours titles a viewer would finish under that rule,
not whether real viewers would finish them.

**2026-11-16, after the main run.** Settings search added (`tune` stage).
The pre-registered settings (ReLU, 64-d output, 256 hidden units, dropout
0.2, learning rate 1e-3, temperature 0.07) were chosen as standard defaults,
not tuned. To check that choice, a random search tries 20 settings at
p = 6: activation (ReLU, GELU, SiLU), output size, hidden size, dropout,
learning rate and temperature, with the pre-registered settings as trial 0.

1. **The test set plays no part in the choice.** Each viewer's latest training
   events (same 20% rule as the test split) form a validation set. The search
   trains on the rest and picks the highest validation Recall@10.
2. **The chosen settings run once on the test set**, with the same seeds as the
   main run, including the H1 text ablation.
3. **The verdicts above are not changed.** They come from the pre-registered
   settings. The tuned model, and the H1 rule applied to it, are reported next
   to them as exploratory results.

**2026-11-17, after the main run.** The catalogue rules changed, and the old
setting is kept as a sensitivity run.

1. **Vote floor lowered from 5 to 1.** Further EDA showed that the 5-vote
   floor removed half of the TV titles (4,833 with full metadata), 80% of them
   non-English, and 58% of the 2025 releases that have at least one vote. That
   biases the H2 test, which is about non-English exposure, and it removes the
   new titles that cold start is about. The floor's other purpose, noisy
   ratings, is already handled by the Bayesian rating: 97 of the top 100
   titles by raw average had fewer than 5 votes, and none do after shrinkage.
   A title with no votes at all is still excluded, because nothing shows it
   was ever watched.
2. **A title now needs a genre and a description** (before: either one), so
   every title has real text for H1. Recommendable titles: 25,280 (41.5% TV,
   49.3% non-English), against 22,059 (35% TV, 45% non-English) before.
3. **The full experiment re-runs on the new catalogue** with the same decision
   rules. The same run with a 5-vote floor and the new metadata rule is
   reported next to it as a sensitivity check (`artifacts/sensitivity_min5/`).
4. **Exploratory item cold-start test added** (`coldstart` stage): 10% of
   titles are hidden from training entirely, and each model ranks only those
   titles for the viewers who watched them.
5. **Exploratory ceiling added** (`oracle` stage): the simulator's own ranking,
   scored on the same split. It shows how much of the reachable Recall@10 each
   model gets, and, without its favourite-title term, how much of the ceiling
   the text taste carries. It changes no model and no verdict.

For transparency: this change was made after seeing the 5-vote results, in
which H2b narrowly failed at p = 2. Both runs are reported, and each verdict
is read from the rules exactly as written above.
