"""Export everything the static web demo needs into webapp/data/demo_data.js.

The interface is a plain HTML page with no server and no third-party libraries,
so all inference is precomputed here:

* MF     - the exported best checkpoint (shared protocol: 3,706 movies)
* BPR    - the exported checkpoint, scored on the same 3,706-movie catalogue
           (the shared-protocol section of the BPR report)
* LRURec - only the five recorded demonstration users (training needs a GPU)

Run from the repository root after the models exist:

    python webapp/tools/export_demo_data.py

Ranking rule matches the evaluation code: score descending, ties to the smaller
MovieID, and each user's observed movies (train + val + test) are hidden.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from load_data import GENRES  # noqa: E402
from mf import BiasedMF, load_shared  # noqa: E402

TOP_K = 10
MF_RUN = ROOT / "runs" / "mf-formal"
BPR_MODEL = ROOT / "artifacts" / "bpr.npz"
OUT = ROOT / "webapp" / "data" / "demo_data.js"
PROCESSED = ROOT / "data" / "processed"


def pack(item_ids, scores, k):
    """(n_users, n_items) scores -> {user: [[movie_id, score], ...]} with ties by MovieID."""
    ids = np.broadcast_to(item_ids, scores.shape)
    order = np.lexsort((ids, -scores), axis=1)[:, :k]
    return order


def recs_mf(model, data, k):
    index = {int(m): i for i, m in enumerate(data.item_ids)}
    result = {}
    users = [int(u) for u in data.user_ids]
    for start in range(0, len(users), 512):
        batch = users[start:start + 512]
        scores = model.scores(batch)
        mask = np.zeros(scores.shape, dtype=bool)
        for row, user in enumerate(batch):
            cols = [index[m] for m in data.seen(user, "final") if m in index]
            mask[row, cols] = True
        scores[mask] = -np.inf
        order = pack(data.item_ids, scores, k)
        for row, user in enumerate(batch):
            result[str(user)] = [[int(data.item_ids[i]), round(float(scores[row, i]), 3)]
                                 for i in order[row]]
    return result


def recs_bpr(state, data, k):
    """BPR scores restricted to the shared catalogue, so all three models compare.

    The saved checkpoint covers 3,883 films; the shared protocol keeps only the
    3,706 films that carry ratings, which is the cohort MF and LRURec use.
    """
    catalogue = data.item_ids
    movies = np.asarray(state["movies"], dtype="int64")
    columns = pd.Index(movies).get_indexer(catalogue)
    if (columns < 0).any():
        raise SystemExit("BPR checkpoint does not cover the shared catalogue")
    item_factors = np.asarray(state["item_factors"], dtype="float64")[columns]
    user_ids = np.asarray(state["users"], dtype="int64")
    user_index = {int(u): i for i, u in enumerate(user_ids)}
    user_factors = np.asarray(state["user_factors"], dtype="float64")

    index = {int(m): i for i, m in enumerate(catalogue)}
    users = [int(u) for u in data.user_ids]
    result = {}
    for start in range(0, len(users), 512):
        batch = users[start:start + 512]
        rows = np.array([user_index[u] for u in batch])
        scores = user_factors[rows] @ item_factors.T
        mask = np.zeros(scores.shape, dtype=bool)
        for local, user in enumerate(batch):
            cols = [index[m] for m in data.seen(user, "final") if m in index]
            mask[local, cols] = True
        scores[mask] = -np.inf
        order = pack(catalogue, scores, k)
        for local, user in enumerate(batch):
            result[str(user)] = [[int(catalogue[i]), round(float(scores[local, i]), 3)]
                                 for i in order[local]]
    return result


def main() -> int:
    data = load_shared(ROOT)
    model = BiasedMF.load(MF_RUN / "best_model.npz")
    with np.load(BPR_MODEL, allow_pickle=False) as loaded:
        bpr_state = {key: loaded[key].copy() for key in loaded.files}

    users_meta = pd.read_csv(PROCESSED / "users_features.csv")
    movies_meta = pd.read_csv(PROCESSED / "movies_features.csv")
    train = pd.read_csv(PROCESSED / "train.csv")

    movie_stats = (train.groupby("MovieID")
                   .agg(n=("Rating", "size"), npos=("label", "sum"), mean=("Rating", "mean")))
    mf_bias = {int(m): round(float(b), 4) for m, b in zip(model.item_ids, model.bi)}
    movie_genres = movies_meta.set_index("MovieID")["Genres"]
    genre_ids = {g: i for i, g in enumerate(GENRES)}

    movies = []
    for row in movies_meta.itertuples():
        st = movie_stats.loc[row.MovieID] if row.MovieID in movie_stats.index else None
        movies.append([
            int(row.MovieID), row.Title, row.Genres,
            None if pd.isna(row.Year) else int(row.Year),
            0 if st is None else int(st["n"]),
            0 if st is None else int(st["npos"]),
            None if st is None else round(float(st["mean"]), 3),
            mf_bias.get(int(row.MovieID)),
        ])
    movies.sort(key=lambda record: record[0])

    joined = train.merge(movies_meta[["MovieID", "Genres"]], on="MovieID", how="left")
    per_user_genres: dict[int, list[int]] = {}
    for uid, group in joined.groupby("UserID"):
        counts = group.Genres.str.split("|").explode().value_counts()
        per_user_genres[int(uid)] = [genre_ids[name] for name in counts.index[:3]]

    liked = train[train.label == 1].sort_values(["UserID", "Timestamp"], ascending=[True, False])
    likes = {int(u): [int(m) for m in g.MovieID.head(3)] for u, g in liked.groupby("UserID")}
    user_stats = (train.groupby("UserID")
                  .agg(n=("Rating", "size"), npos=("label", "sum"), mean=("Rating", "mean")))

    users = []
    for row in users_meta.itertuples():
        uid = int(row.UserID)
        st = user_stats.loc[uid] if uid in user_stats.index else None
        users.append([
            uid, row.Gender, int(row.Age), int(row.Occupation),
            0 if st is None else int(st["n"]),
            0 if st is None else int(st["npos"]),
            None if st is None else round(float(st["mean"]), 3),
            per_user_genres.get(uid, []),
            likes.get(uid, [])[:3],
        ])

    # Every MovieLens user can be asked about; MF scores all of them, and users
    # without training data fall back to the non-personalised bias prediction.
    all_users = [int(u) for u in users_meta.UserID]
    print(f"MF  recommendations for {len(all_users)} users ...", flush=True)
    recs_mf_map = recs_mf(model, data, TOP_K)
    print(f"BPR recommendations for {len(all_users)} users ...", flush=True)
    recs_bpr_map = recs_bpr(bpr_state, data, TOP_K)

    lrurec_final = pd.read_csv(ROOT / "results" / "tables" / "12_lrurec_final_top5.csv")
    recs_lrurec = {}
    for uid, group in lrurec_final.groupby("UserID"):
        group = group.sort_values("Rank")
        recs_lrurec[str(int(uid))] = [[int(m), round(float(s), 3)]
                                      for m, s in zip(group.MovieID, group.Pred_Score)]

    def rank_map(path):
        frame = pd.read_csv(path)
        return {str(int(u)): int(r) for u, r in zip(frame.UserID, frame["rank"])}

    def metric_map(path):
        frame = pd.read_csv(path)
        if "split" in frame.columns:
            frame = frame[frame["split"] == "test"]
        row = frame.iloc[-1]
        aliases = {
            "recall@5": ("Recall@5", "recall_at_5"),
            "ndcg@5": ("NDCG@5", "ndcg_at_5"),
            "recall@10": ("Recall@10", "recall_at_10"),
            "ndcg@10": ("NDCG@10", "ndcg_at_10"),
            "mrr@10": ("MRR@10", "mrr_at_10"),
        }
        out = {}
        for key, names in aliases.items():
            for name in names:
                if name in row.index:
                    out[key] = round(float(row[name]), 4)
                    break
        if "users" in row.index:
            out["users"] = int(row["users"])
        return out

    payload = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": int(data.seed),
            "threshold": int(data.threshold),
            "top_k": TOP_K,
            "n_users": len(users),
            "n_movies": len(movies),
            "mf_catalogue": int(len(model.item_ids)),
            "bpr_catalogue": int(len(data.item_ids)),
            "bpr_protocol": "shared",
            "n_test_users": int(len(rank_map(MF_RUN / "test_per_user.csv"))),
            "environment": {"python": platform.python_version(),
                            "numpy": np.__version__, "pandas": pd.__version__},
            "mf_checkpoint": "runs/mf-formal/best_model.npz",
            "bpr_checkpoint": "artifacts/bpr.npz",
        },
        "genres": list(GENRES),
        "movies": movies,
        "users": users,
        "recs": {"mf": recs_mf_map, "bpr": recs_bpr_map, "lrurec": recs_lrurec},
        "eval_ranks": {"mf": rank_map(MF_RUN / "test_per_user.csv"),
                       "bpr": rank_map(ROOT / "results" / "tables" / "bpr_shared_test_per_user.csv"),
                       "lrurec": rank_map(ROOT / "results" / "lrurec" / "test_per_user.csv")},
        "metrics": {"mf": metric_map(MF_RUN / "mf_metrics.csv"),
                    "bpr": metric_map(ROOT / "results" / "tables" / "bpr_shared_metrics.csv"),
                    "lrurec": metric_map(ROOT / "results" / "tables" / "10_lrurec_metrics.csv")},
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    OUT.write_text(
        "// Generated by webapp/tools/export_demo_data.py - do not edit by hand.\n"
        f"window.DEMO_DATA={body};\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1048576:.2f} MiB)")
    for name, table in payload["metrics"].items():
        print(f"  {name:6s} Recall@10={table.get('recall@10')} NDCG@10={table.get('ndcg@10')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
