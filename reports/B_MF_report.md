# Matrix Factorisation for Movie Recommendation

## Method

Matrix factorisation represents each user and movie by a latent vector. Their inner product models user–movie preference, as introduced in CS5481 Lecture 7, slides 34–35. We use an explicit-rating model with user and movie biases:

\[
\hat r_{ui}=\mu+b_u+c_i+p_u^{\mathsf T}q_i.
\]

Here, μ is the mean training rating, b_u and c_i account for user and movie rating tendencies, and p_u and q_i are d-dimensional latent vectors. The parameters minimise

\[
L=\sum_{(u,i)\in\Omega_{\mathrm{train}}}(r_{ui}-\hat r_{ui})^2
+\lambda\left[\sum_u(b_u^2+\lVert p_u\rVert_2^2)
+\sum_i(c_i^2+\lVert q_i\rVert_2^2)\right].
\]

Only observed training ratings contribute to the error term; missing ratings are not zero-valued targets. The mean μ stays fixed. Both biases and latent vectors receive the same L2 penalty.

We optimise by alternating least squares (ALS). With movie parameters fixed, each user update solves a ridge regression with predictors [1, q_i] and response r_ui − μ − c_i. The first fitted coefficient is the user bias. Movie updates solve the analogous problem with fixed user parameters. One epoch updates every user and then every movie. This optimiser, the bias terms and the regularisation grid are implementation choices within the lecture's matrix-factorisation method.

Latent factors are initialised from a zero-mean normal distribution with standard deviation 0.05 and seed 42. Biases start at zero. Predictions are ranked without clipping, preserving differences between raw estimated ratings. Equal scores are ordered by increasing original movie ID. An unknown user receives μ + c_i; this fallback is non-personalised.

## Shared data and evaluation

We use A's preprocessing at team commit `417d758ebe439c4a9a842c0d16503b0c8b1ff480`. It produces 988,134 training, 6,037 validation and 6,038 test ratings from MovieLens 1M. The three generated CSV files match the SHA-256 hashes in the team's recorded LRURec experiment exactly. A's timestamp-tie handling is retained unchanged.

For each user, the last positive rating is assigned to test and the preceding positive to validation; all remaining ratings are assigned to train. Rating ≥ 4 defines relevance. MF fits all training ratings, including low ratings, from all 6,040 training users. It does not use validation or test ratings to fit parameters or estimate the training mean.

The comparison uses LRURec's fixed catalogue of 3,706 rated movies and its qualification table. Both validation and test contain 6,035 eligible users with the required positive history and targets. The shared test set has three additional target users who do not meet this qualification rule; they are excluded consistently. The catalogue and population are verified against the reference files before training. The resulting comparison is not an all-user cold-start evaluation.

Validation candidates exclude all train-rated movies. Test candidates exclude train plus validation movies. Low-rated movies count as observed history. Each evaluated user has one relevant target, so Recall@K equals the hit indicator. NDCG@K is 1/log2(rank+1) when the target is in the top K and zero otherwise; MRR@K is 1/rank within the top K. Metrics are averaged equally over users. No negative sampling is used.

## Model selection

Three configurations were specified before training. Each runs for at most 20 epochs and stops after five consecutive epochs without improved validation Recall@10. Validation selects both configuration and checkpoint; exact ties retain the earlier trial or epoch. Test evaluation occurs after this selection, with no train+validation refit.

| Latent dimensions | Regularisation λ | Best epoch | Completed epochs | Best validation Recall@10 |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 10 | 4 | 9 | 0.045236 |
| 32 | 10 | 3 | 8 | 0.046396 |
| 32 | 30 | 4 | 9 | 0.043745 |

The selected model has **32 latent dimensions, λ = 10, and three ALS epochs**. These are the best values within the declared search, not a claim of global optimality. The test results were not used for further parameter selection.

## Test results

| Measure | MF |
| --- | ---: |
| Recall@5 | 0.026346 |
| NDCG@5 | 0.017091 |
| Recall@10 | 0.042088 |
| NDCG@10 | 0.022095 |
| MRR@10 | 0.016074 |
| Recall@20 | 0.063795 |
| NDCG@20 | 0.027537 |
| Evaluated users | 6,035 |

The following comparison uses the current MF run and the LRURec results recorded at the pinned team commit. LRURec training was not rerun for this comparison.

| Model | Test users | Recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: |
| MF | 6,035 | 0.042088 | 0.022095 |
| LRURec | 6,035 | 0.243413 | 0.128483 |

LRURec has higher recorded ranking metrics under this common evaluation population, catalogue and filtering policy. MF's squared-rating-error objective does not directly optimise top-K retrieval, which may contribute to its weaker ranking performance. This explanation has not been isolated by an ablation study.

The training signals also differ. MF fits all training ratings and freezes its user vectors for test. LRURec trains on positive sequences and includes the validation item as test-time sequence context. The models use different objectives and search budgets, and each result represents one seed. The table therefore compares these implementations under the shared repository protocol; it does not isolate architecture as the sole cause of the difference. BPR results remain to be added by the team.

## Recommendation examples and implementation checks

The [test](../results/tables/mf_test_top5.csv) and [final-display](../results/tables/mf_final_top5.csv) tables each contain five recommendations for users 1–5, with original movie IDs, titles, genres and estimated scores. Test examples filter train plus validation history and may include the test target. Final-display examples filter all observed train, validation and test movies. Both use the same frozen MF checkpoint; no held-out ratings are fitted for the display.

For example, user 1's final list is *Rear Window*, *The Shawshank Redemption*, *The Bridge on the River Kwai*, *Life Is Beautiful* and *The Godfather*. The interactive program accepts a user ID and applies the complete-history filter before displaying five movies.

Twelve automated tests cover an independently calculated ridge update, decreasing ALS objective, observed-only loss, exact target ranks, ranking metrics, history exclusion, cold start, checkpoint loading, input integrity and deterministic ties. Reproduction commands are in [MF run instructions](../docs/MF运行说明.md); per-user results, validation history and numerical-library versions are in [results/mf](../results/mf/).

## Limitations

The shared split retains some low training ratings whose timestamps are later than the held-out positive. MF fits those ratings, so it can use future low-rating information relative to the target event. LRURec does not train on low-rating records, although its filtering follows the same split. These results are not a strict chronological backtest. Removing this time overlap would require an agreed new split and rerunning all three algorithms.

The model uses rating interactions rather than genres, demographics or temporal dynamics. Cold-start scores rely on biases, and the common evaluation cohort excludes users without the required positive history. Single-seed results and a small search grid limit generalisation. Repeated runs and a shared tuning budget would be needed to assess stability and make stronger model comparisons.

## References

Yifan Zhang. *CS5481 Data Engineering, Lecture 7: Data Driven Applications (1)*, slides 34–35.

CS5481 movie recommendation team. [Preprocessing, LRURec protocol and recorded results](https://github.com/ayame2847/cs5481-movie-recommendation-system/tree/417d758ebe439c4a9a842c0d16503b0c8b1ff480).

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems* 5(4), Article 19. [doi:10.1145/2827872](https://doi.org/10.1145/2827872).
