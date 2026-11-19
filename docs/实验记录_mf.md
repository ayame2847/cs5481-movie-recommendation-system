# MF experiment record

## Experiment

- ID: MF shared split, repository integration, 19 November 2026.
- Contributor: Lu Bai (Part B).
- Question: how well does the lecture's matrix-factorisation method rank unseen movies under the team's shared evaluation protocol?
- Model: explicit-rating matrix factorisation with user/movie biases and ALS; no pretrained weights.
- Method source: CS5481 Lecture 7, slides 34–35. Biases, ALS and the regularisation grid are implementation choices.
- Shared preprocessing and LRURec reference: team commit `417d758ebe439c4a9a842c0d16503b0c8b1ff480`.
- Runtime code and experiment-config SHA-256 values: [metrics.json](../results/mf/metrics.json). These identify the executed files independently of commit metadata.

## Environment and data

- OS/architecture: Darwin 25.6.0, arm64; CPU execution, no GPU; one BLAS thread.
- Python 3.11.15; numpy 2.2.6, pandas 2.2.3, scipy 1.15.3, PyYAML 6.0.2.
- Data: official MovieLens 1M; A's preprocessing was rerun unchanged. Split sizes: 988,134 train, 6,037 validation, 6,038 test ratings. All three CSV hashes match the recorded LRURec run.
- Relevance: rating ≥ 4. The last two positives per user are held out. All remaining ratings, including low ratings, stay in train.
- MF fits only the training ratings; training mean and factors exclude held-out targets. Seed 42 initialises factors. No negative sampling or sequence truncation.
- Evaluation: 6,035 shared users and 3,706 rated movies, verified against the LRURec population and mapping. Each user has one target. Recall@K, NDCG@K and MRR@K are user-macro averages.
- Validation filters all train-rated movies; test filters train plus validation; final recommendations filter all observed ratings. Exact ties use ascending original MovieID. Unknown users receive the global mean plus movie bias.

## Configuration and commands

`configs/config.yaml` supplies shared data settings. `configs/mf.yaml` specifies the ordered grid (16,10), (32,10), (32,30), a 20-epoch limit and patience 5. Validation Recall@10 selects trial and checkpoint; exact ties retain the earlier choice. The selected model is frozen for one test evaluation, with no train+validation refit.

From the repository root, after installing `requirements-mf.txt` and placing raw data in `ml-1m/`:

```bash
python src/preprocess.py
python src/run_mf.py --project . --output runs/mf-formal
python src/mf_demo.py --project . --model runs/mf-formal/best_model.npz --user 1
python src/mf_demo.py --project . --model runs/mf-formal/best_model.npz
python -m unittest discover -s tests -p 'test_mf.py' -v
```

Use a new run directory each time. The runner writes all output files there; curated metrics and recommendation tables are copied to `results/tables/`, and detailed records to `results/mf/`. Weights remain local.

## Results

- Selected: 32 dimensions, regularisation 10, epoch 3. Best validation Recall@10: 0.046396.
- Test Recall@10: 0.042088; NDCG@10: 0.022095; MRR@10: 0.016074.
- Training plus validation: 16.139 seconds; test evaluation: 0.238 seconds. Times describe this local run only.
- The repository-layout run reproduced the previous MF model selection, every test-user rank and all ranking metrics. Twelve numerical and data-integrity tests pass.
- [Test metrics](../results/tables/mf_metrics.csv), [validation history](../results/mf/validation_history.csv), [model selection](../results/mf/model_selection.csv), [per-user test results](../results/mf/test_per_user.csv).
- [Test recommendations](../results/tables/mf_test_top5.csv) and [final recommendations](../results/tables/mf_final_top5.csv): five movies each for users 1–5, with titles and genres.
- [MF/LRURec comparison](../results/tables/mf_lrurec_comparison.csv): LRURec's recorded Recall@10 is 0.243413 and NDCG@10 is 0.128483. LRURec was not retrained for this experiment.

## Interpretation and limits

MF has lower ranking metrics in this comparison. Its squared-rating-error objective does not directly optimise top-K retrieval; the contribution of this choice has not been isolated by ablation. MF fits all training ratings with frozen user vectors, whereas LRURec uses positive sequences and validation-item context at test time. Shared users, catalogue and filters therefore do not isolate architecture as the sole explanation.

The shared split can retain low training ratings later than held-out positives, so this is not a strict chronological backtest. A change to that split would require all methods to rerun. The experiment uses one seed and a small search grid; model search budgets differ. BPR's comparison requires aligned users and candidates before incorporation. The complete method and discussion are in [the MF report](../reports/B_MF_report.md).
