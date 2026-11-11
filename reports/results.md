# Results

All metrics are measured on simulated viewers. They show how well each model recovers the simulator's rules, not real viewer taste.

## H1: text beyond genre (p = 6, two seeds)

| model | recall@10 mean (min to max) | ndcg@10 mean (min to max) | coverage@10 mean (min to max) |
|---|---|---|---|
| two_tower | 0.030 (0.029 to 0.031) | 0.026 (0.026 to 0.026) | 0.056 (0.055 to 0.056) |
| two_tower_no_text | 0.034 (0.033 to 0.035) | 0.027 (0.026 to 0.029) | 0.061 (0.061 to 0.061) |
| content_knn | 0.017 (0.017 to 0.017) | 0.014 (0.014 to 0.014) | 0.101 (0.101 to 0.101) |
| content_knn_no_text | 0.019 (0.019 to 0.019) | 0.018 (0.018 to 0.018) | 0.098 (0.098 to 0.098) |

Verdict: reversed: removing text improves recall@10 by 0.003, larger than the seed spread of 0.002.

## H2: popularity bias against non-English titles

Reference: 45% of recommendable titles are non-English.

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

Figure: `reports/figures/h2_non_english_share.png`.

## All runs

| model                 |   pop_exponent |   recall@10 |   ndcg@10 |   coverage@10 |   non_en_share@10 |   tv_share@10 |   seed |   train_seconds |
|:----------------------|---------------:|------------:|----------:|--------------:|------------------:|--------------:|-------:|----------------:|
| content_knn           |              2 |      0.0114 |    0.0087 |        0.1214 |            0.1345 |        0.315  |     42 |             nan |
| content_knn_no_text   |              2 |      0.0108 |    0.0084 |        0.1184 |            0.1338 |        0.3182 |     42 |             nan |
| content_knn_text_only |              2 |      0.0043 |    0.0031 |        0.2762 |            0.5597 |        0.4448 |     42 |             nan |
| most_watched          |              2 |      0.0034 |    0.003  |        0.0007 |            0.3025 |        0.5003 |     42 |             nan |
| popularity            |              2 |      0.0018 |    0.0013 |        0.0005 |            0.003  |        0.9004 |     42 |             nan |
| two_tower             |              2 |      0.0126 |    0.0102 |        0.0819 |            0.4113 |        0.3782 |     42 |              80 |
| content_knn           |              6 |      0.0165 |    0.0141 |        0.1008 |            0.1239 |        0.3059 |     42 |             nan |
| content_knn_no_text   |              6 |      0.0193 |    0.0175 |        0.0981 |            0.1232 |        0.3136 |     42 |             nan |
| content_knn_text_only |              6 |      0.0027 |    0.002  |        0.2763 |            0.5294 |        0.4495 |     42 |             nan |
| most_watched          |              6 |      0.0089 |    0.008  |        0.001  |            0.4998 |        0.4045 |     42 |             nan |
| popularity            |              6 |      0.0017 |    0.0015 |        0.0005 |            0.0057 |        0.9    |     42 |             nan |
| two_tower             |              6 |      0.0312 |    0.0257 |        0.0551 |            0.4392 |        0.3977 |     42 |              55 |
| two_tower             |              6 |      0.0294 |    0.0257 |        0.0559 |            0.4353 |        0.3671 |     43 |              55 |
| two_tower_no_text     |              6 |      0.0325 |    0.0265 |        0.0607 |            0.4586 |        0.4027 |     42 |              71 |
| two_tower_no_text     |              6 |      0.035  |    0.0285 |        0.0606 |            0.4206 |        0.3614 |     43 |              50 |
| content_knn           |             12 |      0.0234 |    0.0198 |        0.0903 |            0.1149 |        0.3148 |     42 |             nan |
| content_knn_no_text   |             12 |      0.0253 |    0.022  |        0.0873 |            0.1148 |        0.3266 |     42 |             nan |
| content_knn_text_only |             12 |      0.003  |    0.0026 |        0.2672 |            0.513  |        0.4499 |     42 |             nan |
| most_watched          |             12 |      0.0197 |    0.0169 |        0.0012 |            0.316  |        0.6795 |     42 |             nan |
| popularity            |             12 |      0.0044 |    0.0038 |        0.0006 |            0.0098 |        0.8999 |     42 |             nan |
| two_tower             |             12 |      0.0506 |    0.0445 |        0.0442 |            0.4111 |        0.4052 |     42 |              54 |
