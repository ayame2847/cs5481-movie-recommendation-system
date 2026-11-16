# Part A: Dataset Information, Exploration and Preprocessing

> CS5481 Group Project — Movie Recommendation (MovieLens 1M)
> This is the draft of the "Dataset information (20%) + Data exploration and preprocessing (20%)" sections of the technical report. All figures and numbers were produced by actually running the scripts under `src/`; the corresponding output tables and figures are under `results/`.

## 1. Dataset Information

### 1.1 Data Source

MovieLens 1M (GroupLens), official archive `https://files.grouplens.org/datasets/ml-1m.zip`. It contains three tables — `ratings.dat`, `users.dat`, `movies.dat` — all `::`-separated with no header row. Citation: Harper & Konstan (2015), *The MovieLens Datasets: History and Context*, ACM TiiS 5(4), Article 19, DOI: 10.1145/2827872. The raw data is neither redistributed nor committed to the repository.

### 1.2 Shape

| Dataset | Rows | Columns |
|---|---:|---:|
| ratings | 1,000,209 | 4 |
| users | 6,040 | 5 |
| movies | 3,883 | 3 |
| comparison (merged view) | 1,000,209 | 10 |

Output table: `results/tables/01_shapes.csv`.

### 1.3 Missing Values

- The **total number of missing values is 0** in all three raw tables and in the merged view; **no column contains any nulls**.
- Detection results: `results/tables/02_null_summary.csv` (per-column null counts plus a `__TOTAL__` row per table).

Consequently, no missing-value imputation is required. The preprocessing effort therefore focuses on type handling, encoding and splitting (Section 3).

### 1.4 Comparison Dataset

We build the **comparison dataset** as a full analysis view via `ratings ⋈ users ⋈ movies` (key joins with `many_to_one` validation): every rating record carries the user's demographic attributes (Gender/Age/Occupation/Zip-code) and the movie's attributes (Title/Genres), yielding 1,000,209 rows × 10 columns. The merged view contains no nulls, confirming that every rating can be fully joined to a valid user and movie.

- File: `data/processed/comparison_dataset.csv` (~73 MB, local only, not committed; reproducible via `src/dataset_info.py`)
- Summary: `results/tables/06_comparison_dataset_summary.csv` (rows/columns/nulls/memory of each table)

This view supports joint exploratory analysis and serves as the shared data basis on which all three algorithms are compared fairly.

### 1.5 General Information, Unique Values and Data Types

- Full `info()` output and the first 3 rows of each table: `results/tables/05_info.txt`
- Unique values per column: `results/tables/04_nunique.csv`. Highlights: 6,040 unique UserIDs and 3,706 unique MovieIDs in ratings (i.e., 3,706 of the 3,883 movies have at least one rating); Rating takes 5 values (1–5).
- Data types per column: `results/tables/03_dtypes.csv`. Raw types: UserID/MovieID/Rating/Timestamp are int64; Gender/Title/Genres are object; Age/Occupation are int64-coded categories; Zip-code is read as string (preserves leading zeros).

## 2. Data Exploration

Summary statistics: `results/tables/07_eda_summary.csv`; figures: `results/figures/`.

### 2.1 Rating Distribution (fig01)

Mean rating **3.5816**, median **4**. A rating of 4 is the most frequent (348,971 ratings, 34.9%), followed by 3 (261,197) and 5 (226,310); 1 is the rarest (56,174). The distribution is clearly skewed towards high ratings, and **ratings ≥ 4 account for 57.52%** of all ratings — the direct motivation for our label threshold (Section 3.2).

### 2.2 User Activity (fig02)

Ratings per user: min 20, max 2,314, mean 165.6, median 96. The mean far exceeds the median, indicating a right-skewed long tail: a small group of heavy users contributes a large share of ratings. The guarantee of at least 20 ratings per user makes leave-one-out splitting feasible for every user.

### 2.3 Movie Popularity and the Long Tail (fig03)

Ratings per movie: mean 269.9, median 123.5, max 3,428. The rank-count curve under log-log axes decays approximately as a power law — a pronounced long tail. This points to potential **popularity bias** in evaluation (popular movies dominating recommendations), which we will revisit when discussing recommendation results.

### 2.4 Temporal Pattern (fig04)

Ratings span **2000-04-25 to 2003-02-28**. Rating volume concentrates in the second half of 2000 (peak in **November 2000 with 290,793 ratings**), then drops sharply after early 2001 and flattens. Since activity is highly non-uniform over time, **splitting by per-user timestamps** matches the real recommendation scenario better than a global random split.

### 2.5 Genres (fig05)

A movie may belong to multiple genres (`|`-separated, 18 genres in total). Drama (1,603 movies) and Comedy (1,200) dominate, followed by Action (503), Thriller (492) and Romance (471).

### 2.6 User Demographics (fig06)

- Gender: 4,331 male (71.7%) vs 1,709 female (28.3%) — a clear imbalance.
- Age: 25–34 is the largest group (2,096 users, 34.7%), followed by 35–44 (1,193) and 18–24 (1,103); Under 18 is the smallest (222).
- Occupation: college/grad student is the largest (759), followed by other/not specified (711) and executive/managerial (679).

### 2.7 Sparsity

User–movie matrix sparsity = 1 − 1,000,209 / (6,040 × 3,883) = **95.74%**. This high sparsity is a central challenge for collaborative filtering and a key motivation for matrix factorization and sequential modelling.

## 3. Data Preprocessing

Script: `src/preprocess.py`; cleaning comparison: `results/tables/08_cleaning_comparison.csv`.

### 3.1 Cleaning Checks

Five checks are applied sequentially to the ratings table; **none removed any rows on this dataset** (the logic is kept so the pipeline transfers to other datasets):

| Check | Rule | Rows removed |
|---|---|---:|
| Duplicate user–movie pairs | Keep the record with the latest timestamp | 0 |
| Invalid ratings | Keep only 1–5 | 0 |
| Invalid timestamps | Keep only positive values | 0 |
| Ratings with no matching user | Drop | 0 |
| Ratings with no matching movie | Drop (guarantees titles/genres are displayable) | 0 |

### 3.2 Relevance Labels

We convert explicit ratings into implicit relevance as follows: **Rating ≥ 4 → label = 1 (positive); Rating ≤ 3 → label = 0**. Rationale:

1. Semantically, 4–5 stars mean "like / would recommend", whereas 1–3 do not express positive preference;
2. At this threshold positives cover 57.52% of all ratings — a balanced, stable evaluation basis;
3. All three algorithms share the same label definition, ensuring a fair comparison.

The train/val/test files keep both the original `Rating` and the derived `label`: explicit models (e.g., MF) may regress on 1–5 ratings, while implicit/ranking models (BPR, LRURec) consume positives only.

### 3.3 Feature Encoding and Standardization

- **Users** (`data/processed/users_features.csv`): Gender is binary-encoded (`Gender_F`); Age keeps its original ordinal codes (1<18<25<35<45<50<56, ordered by age) with a text label; Occupation keeps its category code with a text label; Zip-code is treated as a string identifier, not a numeric feature.
- **Movies** (`data/processed/movies_features.csv`, 23 columns): release `Year` is extracted from Title (1919–2000, no missing values); `Genres` is expanded into an **18-dimensional multi-hot** vector (`genre_Action` … `genre_Western`); as a standardization example, `Year_zscore` provides z-score normalized year. Release year is intrinsic content metadata, independent of rating behaviour, so no label leakage is involved.
- **Ratings**: `Timestamp` can be converted to datetime (already used in the temporal EDA); split files preserve raw epoch seconds for sequential models.

### 3.4 Reproducible Train / Validation / Test Split

We use **leave-one-out (per user, time-ordered)**, executed deterministically with no randomness (global seed 42 is recorded in `configs/config.yaml` for later sampling and model use):

1. Sort each user's **positive ratings** by timestamp ascending;
2. Last positive → **test**; second-to-last positive → **val**;
3. **All** remaining rating records (low ratings and remaining positives) → **train**.

Edge cases: 2 users have no positive rating at all (all records go to train; excluded from evaluation); 1 user has exactly one positive (goes to test; empty val).

Split statistics (`results/tables/09_split_stats.csv`):

| Split | Rows | Users | Positives |
|---|---:|---:|---:|
| train | 988,134 | 6,040 | 563,206 |
| val | 6,037 | 6,037 | 6,037 |
| test | 6,038 | 6,038 | 6,038 |

The total is 1,000,209 rows, matching the raw ratings. Built-in assertions verify: no (UserID, MovieID) pair crosses splits; each user has at most one val and one test record; row counts are preserved — all passed.

Tied timestamps: ML-1M users often rate several movies within the same second. The sort key is `(UserID, Timestamp, MovieID)`, so ties are broken by MovieID; the last train positive may therefore share its timestamp with val/test, which is expected.

Low ratings after the held-out positives: train retains low-rated records for explicit models; 15,295 of them are later than the val positive and 8,263 are later than the test positive, **all with label = 0**. In other words, no positive in train is later than val/test (ties excepted), so no future positive information leaks into training. Implicit/sequential models use train positives only and strictly respect the temporal direction; explicit MF uses low ratings as auxiliary signals, and all three models share the same train set, keeping the comparison fair.

### 3.5 Leakage Prevention

1. **Temporal direction**: val/test are each user's latest positives; every positive in train is no later than val/test (same-second ties excepted). Records later than val/test in train are exclusively low-rated auxiliary signals (quantified in 3.4); held-out positives never enter training, emulating "predict the future from the past".
2. **Statistics computed from train only**: item popularity (`data/processed/train_item_popularity.csv`: train rating count, train positive count, train mean rating) and any future behavioural feature must never incorporate val/test information.
3. **Candidate filtering during evaluation**: when evaluating on val, filter out movies already observed in train; when evaluating on test, filter train + val; for final demo recommendations, filter all observed movies (train + val + test), guaranteeing "recommend unrated movies".
4. **Tuning and model selection use val only**; test is touched exactly once for final evaluation.

## 4. Reproducibility and File Manifest

```bash
pip install -r requirements.txt   # pandas / numpy / matplotlib / pyyaml
python src/dataset_info.py        # Task 1: dataset information + comparison dataset
python src/eda.py                 # Task 2a: EDA figures and statistics
python src/preprocess.py          # Task 2b: cleaning, encoding, splitting, checks
```

Environment: Windows + Python 3 (managed 3.x runtime); global seed 42. Place the raw archive under `ml-1m/` (or adjust `raw_dir` in `configs/config.yaml`).

| File | Content |
|---|---|
| `results/tables/01–06` | shape / nulls / dtypes / nunique / info / comparison summary |
| `results/tables/07_eda_summary.csv` | EDA summary statistics |
| `results/tables/08_cleaning_comparison.csv` | Before/after cleaning comparison |
| `results/tables/09_split_stats.csv` | Split statistics |
| `results/figures/fig01–fig06` | Ratings / users / movies / time / genres / demographics |
| `data/processed/train.csv, val.csv, test.csv` | Unified splits (UserID,MovieID,Rating,Timestamp,label) |
| `data/processed/users_features.csv, movies_features.csv` | Encoded features |
| `data/processed/train_item_popularity.csv` | Train-only popularity statistics |
| `data/processed/comparison_dataset.csv` | Merged comparison dataset (73 MB, not committed) |

## 5. Interfaces for B / C / D / E

- **B (MF)**: train on explicit ratings in `train.csv`; tune on `val.csv`; score and rank candidate movies with the filtering rules in 3.5.
- **C (BPR)**: use `label = 1` records in `train.csv` as implicit positive feedback; negative sampling with seed 42.
- **D (LRURec)**: build positive sequences ordered by `UserID, Timestamp` from train only (under the standard leave-one-out protocol when evaluating val/test).
- **E (Evaluation & demo)**: unified metrics Recall@10 / NDCG@10 with full-candidate ranking (3,706 rated movies or all 3,883 — fix one choice in the evaluation script and record it); join `movies_features.csv` to display titles and genres in recommendations.

> Out of scope for Part A: training and tuning the three algorithms, implementing the unified evaluation, comparing recommendation results, the demo interface, and final report integration.
