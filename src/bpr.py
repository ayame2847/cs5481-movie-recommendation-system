"""BPR matrix factorization and full-catalog ranking on the Part A split."""

import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd

from config import CFG, POSITIVE_THRESHOLD, PROCESSED_DIR, ROOT, SEED, TABLES_DIR


MODEL_PATH = ROOT / "artifacts" / "bpr.npz"
SHARED_MODEL_PATH = ROOT / "artifacts" / "bpr_shared.npz"
LRUREC_RESULTS = ROOT / "results" / "lrurec"
SAMPLE_USERS = [1, 42]
K = 10
SHARED_KS = (1, 5, 10, 20, 50)
BPR_CONFIG = CFG["bpr"]
BATCH_SIZE = BPR_CONFIG["batch_size"]
LEARNING_RATE = BPR_CONFIG["learning_rate"]
REGULARIZATION = BPR_CONFIG["regularization"]
DIMENSIONS = BPR_CONFIG["factors"]
VALIDATE_AT = BPR_CONFIG["validate_at"]


class MovieLensSplit:
    def __init__(self):
        self.movies = pd.read_csv(
            PROCESSED_DIR / "movies_features.csv", usecols=["MovieID", "Title", "Genres"]
        ).set_index("MovieID").sort_index()
        self.users = pd.Index(
            pd.read_csv(PROCESSED_DIR / "users_features.csv", usecols=["UserID"])
            ["UserID"].sort_values()
        )

    def read(self, split):
        frame = pd.read_csv(
            PROCESSED_DIR / f"{split}.csv", usecols=["UserID", "MovieID", "label"]
        )
        users = self.users.get_indexer(frame["UserID"])
        movies = self.movies.index.get_indexer(frame["MovieID"])
        if (users < 0).any() or (movies < 0).any():
            raise ValueError(f"{split}.csv contains IDs missing from the feature tables")
        return users, movies, frame["label"].to_numpy() == 1

    def mask(self, users, movies):
        seen = np.zeros((len(self.users), len(self.movies)), dtype=bool)
        seen[users, movies] = True
        return seen


class BPR:
    def __init__(self, positives, dimensions, seed):
        self.rng = np.random.default_rng(seed)
        self.positives = positives
        self.positive_users, self.positive_movies = positives.nonzero()
        n_users, n_movies = positives.shape
        if not len(self.positive_users) or positives.all(axis=1).any():
            raise ValueError("BPR requires training positives and a comparison item per active user")
        self.user_factors = self.rng.normal(0, 0.01, (n_users, dimensions)).astype(np.float32)
        self.item_factors = self.rng.normal(0, 0.01, (n_movies, dimensions)).astype(np.float32)
        self.user_factors[~positives.any(axis=1)] = 0

    def train_epoch(self):
        n_pairs = len(self.positive_users)
        n_movies = self.item_factors.shape[0]
        pairs = self.rng.integers(n_pairs, size=n_pairs)
        users = self.positive_users[pairs]
        liked = self.positive_movies[pairs]
        other = self.rng.integers(n_movies, size=n_pairs)
        # Rejection sampling is uniform over this user's non-positive movies.
        invalid = self.positives[users, other]
        while invalid.any():
            other[invalid] = self.rng.integers(n_movies, size=invalid.sum())
            invalid = self.positives[users, other]

        loss = 0.0
        for start in range(0, n_pairs, BATCH_SIZE):
            batch = slice(start, start + BATCH_SIZE)
            u, i, j = users[batch], liked[batch], other[batch]
            p = self.user_factors[u]
            qi, qj = self.item_factors[i], self.item_factors[j]
            difference = qi - qj
            margin = np.einsum("ij,ij->i", p, difference)
            pair_loss = np.logaddexp(0.0, -margin)
            loss += pair_loss.sum(dtype=np.float64)
            weight = -np.expm1(-pair_loss)[:, None]  # sigmoid(-margin)
            # Compute all gradients from pre-update factors. Accumulate repeats.
            np.add.at(self.user_factors, u, LEARNING_RATE * (weight * difference - REGULARIZATION * p))
            np.add.at(self.item_factors, i, LEARNING_RATE * (weight * p - REGULARIZATION * qi))
            np.add.at(self.item_factors, j, LEARNING_RATE * (-weight * p - REGULARIZATION * qj))
        return loss / n_pairs


def target_ranks(user_factors, item_factors, seen, users, targets):
    """One positive per user. Ties go to the smaller MovieID."""
    if len(users) == 0 or len(np.unique(users)) != len(users) or seen[users, targets].any():
        raise ValueError("Expected one unmasked held-out movie per evaluation user")
    ranks = np.empty(len(users), dtype=np.int64)
    movie_indices = np.arange(len(item_factors))
    for start in range(0, len(users), 256):
        batch = slice(start, start + 256)
        batch_users, batch_targets = users[batch], targets[batch]
        scores = user_factors[batch_users] @ item_factors.T
        scores[seen[batch_users]] = -np.inf
        target_scores = scores[np.arange(len(batch_users)), batch_targets][:, None]
        ahead = (scores > target_scores) | (
            (scores == target_scores) & (movie_indices < batch_targets[:, None])
        )
        ranks[batch] = 1 + ahead.sum(axis=1)
    return ranks


def ranking_metrics(ranks, k):
    top_ranks = ranks[ranks <= k]
    return {
        f"hits_at_{k}": len(top_ranks),
        f"recall_at_{k}": len(top_ranks) / len(ranks),
        f"ndcg_at_{k}": float((1 / np.log2(top_ranks + 1)).sum() / len(ranks)),
        f"mrr_at_{k}": float((1 / top_ranks).sum() / len(ranks)),
    }


def metrics_at_10(ranks):
    metrics = ranking_metrics(ranks, K)
    del metrics["mrr_at_10"]
    return {"users": len(ranks), **metrics}


def evaluate(user_factors, item_factors, seen, users, targets):
    ranks = target_ranks(user_factors, item_factors, seen, users, targets)
    return metrics_at_10(ranks)


def load_shared_evaluation(data):
    """Read LRURec's committed cohort and catalogue for the same split."""
    reference = json.loads((LRUREC_RESULTS / "run_metadata.json").read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        contents = (PROCESSED_DIR / f"{split}.csv").read_bytes()
        # A's generated CSVs may use CRLF on Windows and LF on Colab.
        if sha256(contents.replace(b"\r\n", b"\n")).hexdigest() != reference["input_sha256"][split]:
            raise ValueError(f"{split}.csv differs from the LRURec evaluation split")
    if SEED != reference["seed"] or POSITIVE_THRESHOLD != reference["threshold"]:
        raise ValueError("Seed or relevance threshold differs from the LRURec run")

    mapping = json.loads((LRUREC_RESULTS / "item_mapping.json").read_text(encoding="utf-8"))
    movie_ids = [int(movie_id) for movie_id in mapping]
    candidate_movies = data.movies.index.isin(movie_ids)
    population = pd.read_csv(
        LRUREC_RESULTS / "user_population.csv",
        usecols=["UserID", "val_eligible", "test_eligible"],
        dtype={"val_eligible": "boolean", "test_eligible": "boolean"},
    ).set_index("UserID", verify_integrity=True).sort_index()
    if not population.index.equals(data.users) or population.isna().any().any():
        raise ValueError("LRURec population does not match the preprocessed user table")
    val_users = population["val_eligible"].to_numpy(dtype=bool)
    test_users = population["test_eligible"].to_numpy(dtype=bool)
    counts = reference["population"]
    if (candidate_movies.sum() != len(movie_ids) or len(movie_ids) != counts["n_items"]
            or val_users.sum() != counts["val_users"] or test_users.sum() != counts["test_users"]):
        raise ValueError("LRURec cohort or catalogue differs from the recorded experiment")
    return candidate_movies, val_users, test_users


def recommendations(data, user_factors, item_factors, seen, userid, k):
    user = data.users.get_indexer([userid])[0]
    if user < 0:
        raise ValueError(f"Unknown UserID: {userid}")
    candidates = np.flatnonzero(~seen[user])
    scores = item_factors[candidates] @ user_factors[user]
    # Candidate indices are in ascending MovieID order; stable sorting breaks ties.
    chosen = np.argsort(-scores, kind="stable")[:k]
    table = data.movies.iloc[candidates[chosen]].reset_index()
    table.insert(0, "user_id", userid)
    table.insert(1, "rank", np.arange(1, len(table) + 1))
    table["bpr_score"] = scores[chosen]
    return table


def train(data, shared=False):
    train_users, train_movies, labels = data.read("train")
    val_users, val_movies, val_labels = data.read("val")
    seen = data.mask(train_users, train_movies)
    positives = data.mask(train_users[labels], train_movies[labels])
    val_eligible = val_labels
    selection_metric = "ndcg_at_10"
    model_path = MODEL_PATH
    if shared:
        candidates, shared_val, shared_test = load_shared_evaluation(data)
        seen[:, ~candidates] = True
        val_eligible = val_labels & shared_val[val_users]
        if val_eligible.sum() != shared_val.sum():
            raise ValueError("LRURec validation cohort is missing a BPR target")
        selection_metric = "recall_at_10"
        model_path = SHARED_MODEL_PATH
        print(f"Shared evaluation: {val_eligible.sum()} validation users, "
              f"{shared_test.sum()} test users, {candidates.sum()} catalogue movies. "
              "Selecting by validation Recall@10.", flush=True)
    metrics = []
    best_score = -np.inf
    for dimensions in DIMENSIONS:
        model = BPR(positives, dimensions, SEED)
        training_seconds = 0.0
        for epoch in range(1, max(VALIDATE_AT) + 1):
            started = perf_counter()
            pair_loss = model.train_epoch()
            training_seconds += perf_counter() - started
            if epoch not in VALIDATE_AT:
                continue
            result = evaluate(
                model.user_factors, model.item_factors, seen,
                val_users[val_eligible], val_movies[val_eligible],
            )
            metrics.append({
                "split": "val", "factors": dimensions, "epoch": epoch,
                "training_seconds": round(training_seconds, 2),
                "last_epoch_pair_loss": pair_loss, **result,
            })
            print(f"val  d={dimensions:2} epoch={epoch:2}  "
                  f"Recall@10={result['recall_at_10']:.4f}  NDCG@10={result['ndcg_at_10']:.4f}  "
                  f"train={training_seconds:.1f}s", flush=True)
            if result[selection_metric] > best_score:
                best_score = result[selection_metric]
                best_row = metrics[-1]
                best_users = model.user_factors.copy()
                best_movies = model.item_factors.copy()

    # Rank test targets only after the configuration and checkpoint are fixed.
    seen[val_users, val_movies] = True
    test_users, test_movies, test_labels = data.read("test")
    test_eligible = test_labels
    if shared:
        test_eligible = test_labels & shared_test[test_users]
        if test_eligible.sum() != shared_test.sum():
            raise ValueError("LRURec test cohort is missing a BPR target")
    ranks = target_ranks(
        best_users, best_movies, seen, test_users[test_eligible], test_movies[test_eligible]
    )
    test = metrics_at_10(ranks)
    print(f"test d={best_row['factors']:2} epoch={best_row['epoch']:2}  "
          f"Recall@10={test['recall_at_10']:.4f}  NDCG@10={test['ndcg_at_10']:.4f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path, users=data.users.to_numpy(), movies=data.movies.index.to_numpy(),
        user_factors=best_users, item_factors=best_movies,
    )
    if shared:
        result = {
            "model": "BPR", "seed": SEED, "factors": best_row["factors"],
            "epoch": best_row["epoch"], "users": len(ranks),
            "candidate_movies": int(candidates.sum()),
        }
        for k in SHARED_KS:
            result.update(ranking_metrics(ranks, k))
        validation = pd.DataFrame(metrics).assign(candidate_movies=int(candidates.sum()))
        save_table(validation, TABLES_DIR / "bpr_shared_validation.csv", "%.8f")
        save_table(pd.DataFrame([result]), TABLES_DIR / "bpr_shared_metrics.csv", "%.8f")
        return

    metrics.append({**best_row, "split": "test", **test})
    save_table(pd.DataFrame(metrics), TABLES_DIR / "bpr_metrics.csv", "%.8f")
    seen[test_users, test_movies] = True
    samples = [
        recommendations(data, best_users, best_movies, seen, userid, 5)
        for userid in SAMPLE_USERS
    ]
    save_table(
        pd.concat(samples, ignore_index=True),
        TABLES_DIR / "bpr_recommendations.csv", "%.6f",
    )


def save_table(frame, path, float_format):
    """Replace a complete CSV atomically, keeping a new copy if the target is locked."""
    with TemporaryDirectory(dir=path.parent, prefix=".bpr-") as directory:
        pending = Path(directory) / path.name
        frame.to_csv(pending, index=False, float_format=float_format)
        try:
            pending.replace(path)
        except PermissionError:
            print(f"Cannot replace {path.name}; close it in WPS/Excel to allow overwriting.")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
            pending.replace(path)
    print(f"Saved result -> {path}", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="train, validate, and test")
    training.add_argument(
        "--shared", action="store_true",
        help="use LRURec's evaluation cohort and catalogue, selecting by validation Recall@10",
    )
    recommend = commands.add_parser("recommend", help="rank movies the user has not rated")
    recommend.add_argument("--user", type=int, required=True, help="MovieLens UserID")
    recommend.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.command == "recommend" and args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.command == "recommend" and not MODEL_PATH.is_file():
        parser.error("run python src/bpr.py train first")

    try:
        data = MovieLensSplit()
        if args.command == "train":
            train(data, shared=args.shared)
            return

        with np.load(MODEL_PATH) as model:
            if not (np.array_equal(model["users"], data.users)
                    and np.array_equal(model["movies"], data.movies.index)):
                parser.error("checkpoint IDs differ from current data; retrain with this split")
            seen = data.mask(*data.read("train")[:2])
            for split in ("val", "test"):
                users, movies, _ = data.read(split)
                seen[users, movies] = True
            table = recommendations(
                data, model["user_factors"], model["item_factors"], seen, args.user, args.top_k
            )
        if table.empty:
            print(f"User {args.user} has no unrated movies left in the catalog.")
        else:
            print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    except FileNotFoundError as exc:
        parser.exit(1, f"Missing {exc.filename}. Run python src/preprocess.py first.\n")
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
