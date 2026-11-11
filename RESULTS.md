# Experiment results

## Executive result

The compact neural metric-learning encoder is the strongest tested retrieval model on
the catalog-only cold-start proxy. Adding MMR reranking provides a small further gain.

The evaluation contains 500 synthetic episodes. Each episode uses three related
pre-2025 titles as a profile and measures the rank of a held-out 2025 title among 1,991
2025 candidates. This is a metadata-consistency experiment, not an estimate of customer
watch behavior.

| Model | Recall@10 | NDCG@10 | MRR@10 | Median rank | Coverage@10 | Diversity@10 |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 0.006 | 0.004 | 0.003 | 892.5 | 0.005 | 0.890 |
| TF-IDF | 0.366 | 0.178 | 0.122 | 22.5 | 0.637 | 0.663 |
| SVD | 0.340 | 0.185 | 0.139 | 24.0 | 0.629 | 0.598 |
| Neural | 0.468 | 0.261 | 0.198 | 13.0 | 0.621 | 0.657 |
| Neural + MMR | **0.478** | **0.270** | **0.206** | **12.0** | **0.668** | **0.672** |

Relative to TF-IDF, neural + MMR improves Recall@10 by 30.6%, NDCG@10 by 51.8%,
and MRR@10 by 69.6%. These gains support the first research hypothesis within the
explicit limitations of the proxy labels.

MMR improves coverage by 7.6% relative to the neural model and increases diversity by
2.2%. This direction is positive but does not meet the originally proposed 10% coverage
and 15% diversity thresholds. The diversity hypothesis is therefore not yet validated.
The next experiment should tune MMR on 2024 validation episodes or introduce explicit
genre/language quotas.

## EDA decisions reflected in the implementation

- Nine duplicate TV IDs are resolved by retaining the most complete row.
- A composite movie/TV item ID prevents cross-table ID collisions.
- The inconsistent movie and TV genre taxonomies are canonicalized.
- Movie duration and TV duration are excluded because they contain no useful variation.
- Zero-vote ratings are treated as unobserved rather than as customer dislike.
- Popularity, votes and ratings are not inputs to the temporal retrieval embedding.
- Feature vocabularies and SVD are fitted only on 2010-2023 rows.
- 2024 is reserved as a validation period and 2025 is the test candidate period.

## Interpretation of sample outputs

The family-animation persona retrieves semantically consistent animation and family
titles. The science-fiction and Korean personas generally retrieve the correct broad
taste, but some results are too broad because the proxy training groups emphasize
primary genre and language. This is a known weakness of content-only recommendation.

The next model iteration should add a pretrained multilingual transformer and stronger
hard negatives. In a telecom production system, impression, play, completion, skip,
search and entitlement events should replace metadata proxy labels and train a genuine
user tower.

## Reproducibility

- Random seed: 42
- Training rows: 28,000
- Validation rows: 2,000
- Test candidates after deduplication: 1,991
- Training triplets: 27,747
- TF-IDF features: 8,000
- SVD dimensions: 256
- Neural embedding dimensions: 128
- Neural epochs: 15

See `README.md` for the exact execution command and `artifacts/` for all generated
metrics, figures, model weights and recommendation samples.
