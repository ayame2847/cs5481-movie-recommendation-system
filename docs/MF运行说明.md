# MF — Run instructions

[Project overview](../README.md) · [Report](../reports/B_MF_report.md)

Explicit-rating matrix factorisation with user and movie biases, fitted by alternating least squares (ALS). The model implements the lecture component of the team's MF/BPR/LRURec comparison.

## Reproduce the experiment

Use Python 3.11 and run these commands from the repository root. Prepare `ml-1m/` as described in the project README.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mf.txt
python src/preprocess.py
python src/run_mf.py --project . --output runs/mf-formal
python src/mf_demo.py --project . --model runs/mf-formal/best_model.npz
python -m unittest discover -s tests -p 'test_mf.py' -v
```

On Windows, activate the environment with `.venv\Scripts\activate`. The dataset is obtained from [GroupLens](https://grouplens.org/datasets/movielens/1m/); read its included README and usage terms. A's preprocessing is unchanged from team commit `417d758ebe439c4a9a842c0d16503b0c8b1ff480`. The three generated rating files must match the hashes in the team's LRURec record; the runner stops on a mismatch.

If A's files already exist in a team checkout, pass that checkout to `--project`. Use a new output directory for each experiment. No GPU is required. The default run reads `configs/mf.yaml`: three configurations, at most 20 epochs each and validation patience 5. Use `--config` to select another experiment file; `--epochs` and `--patience` override its limits. The shared seed, relevance threshold and data paths remain in `configs/config.yaml`. BLAS thread limits are set to one in the training process.

The demo accepts a user ID and displays five unrated movies, including titles and genres. Enter `q` to exit. For one-shot output, append `--user 1`. New users receive non-personalised scores from the global mean and movie biases. Scores are raw rating estimates and may fall outside [1,5]; clipping is not applied to ranking.

## Data and comparison protocol

Required A outputs are `data/processed/train.csv`, `val.csv`, `test.csv` and `movies_features.csv`. Ratings contain `UserID,MovieID,Rating,Timestamp,label`; movie metadata contains `MovieID,Title,Genres`. The path is read from the team's `configs/config.yaml`.

The shared split holds out each user's last positive rating for test and preceding positive for validation, using A's deterministic tie handling. All other ratings remain in train. Rating ≥ 4 is relevant. MF fits all 988,134 training ratings; held-out ratings do not enter its loss or fitted parameters.

Evaluation reproduces the LRURec user-qualification table and the fixed catalogue of 3,706 movies appearing in the shared splits. Both validation and test use 6,035 eligible users. This common cohort excludes some users whom MF could score, so the reported metrics are not an all-user cold-start evaluation. Training still includes all 6,040 users in A's train file.

Validation filters all train-rated movies. Test filters train plus validation movies. Final display filters all observed ratings. Each eligible user has one held-out positive; Recall@K is therefore equal to HitRate@K. NDCG uses binary relevance, MRR uses reciprocal target rank, and all metrics are user-macro averages. Equal scores are ordered by ascending original movie ID.

The candidate grid is (factors, regularisation) = (16,10), (32,10), (32,30). Validation Recall@10 selects the configuration and epoch. Exact ties retain the earlier trial/epoch. The selected checkpoint is frozen for one final test evaluation, without a train+validation refit. Seed 42 controls factor initialisation.

The shared split can retain low training ratings later than the held-out positive. MF fits these ratings; the experiment is not a strict chronological backtest. LRURec trains on positive sequences and uses validation context at test time, whereas MF retains frozen train-only user vectors. The report discusses these differences.

## Files and outputs

| File | Purpose |
| --- | --- |
| `src/mf.py` | Data validation, biased MF, ALS updates, ranking and recommendation functions |
| `src/run_mf.py` | Shared-protocol checks, validation selection and final evaluation |
| `src/mf_demo.py` | Interactive recommendations from a saved checkpoint |
| `requirements-mf.txt` | Exact numerical-library versions |
| `reports/B_MF_report.md` | Method, experiment, results and limitations |
| `results/tables/mf_metrics.csv` | Final test metric table |
| `configs/mf.yaml` | Declared search grid and stopping limits |
| `results/mf/metrics.json` | Configuration, metrics, input/code hashes and versions |
| `results/mf/model_selection.csv` | Best validation result for each configuration |
| `results/mf/validation_history.csv` | Full epoch history |
| `results/mf/test_per_user.csv` | Original user/target IDs, ranks and per-user metrics |
| `results/tables/mf_test_top5.csv` | Five recommendations each for users 1–5 under the test history filter |
| `results/tables/mf_final_top5.csv` | Five recommendations each for users 1–5 after excluding their complete history |
| `results/tables/mf_lrurec_comparison.csv` | MF results alongside the team's recorded LRURec results |
| `results/mf/user_population.csv` | Common cohort eligibility |
| `results/mf/item_mapping.json` | Original movie IDs mapped to the shared one-based item indices |

Training writes all outputs into the supplied run directory, including `best_model.npz`. The committed metric and recommendation tables are in `results/tables/`; detailed records are in `results/mf/`. Raw data, environments and weights remain local and are excluded from Git.

For integration, import `BiasedMF`, `load_shared` and `recommend` from `src/mf.py`. A checkpoint can be loaded with `BiasedMF.load(path)`. Call `recommend(model, data, user_id, stage='final', n=5)` to apply the complete-history filter. The supplied CLI checks that the checkpoint and data use the same movie catalogue.

## References

- Yifan Zhang. *CS5481 Lecture 7: Data Driven Applications (1)*, slides 34–35 (matrix factorisation).
- [Team preprocessing and evaluation protocol](https://github.com/ayame2847/cs5481-movie-recommendation-system/tree/417d758ebe439c4a9a842c0d16503b0c8b1ff480).
- F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM TiiS 5(4), Article 19. [doi:10.1145/2827872](https://doi.org/10.1145/2827872).
