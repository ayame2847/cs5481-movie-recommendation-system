"""任务二（下）：数据预处理、特征编码与可复现数据划分。

处理流程：
1. 清洗校验：重复 user-movie 对、非法评分、非法时间戳、关联完整性
   （本数据集均为 0，若有则按注释规则处理并记录）。
2. 类型处理与编码：
   - ratings: Timestamp -> datetime，Rating>=4 -> label=1（相关性标签）
   - users:   Gender 二值编码，Age 有序类别（保留原码+文本标签），Occupation 类别标签
   - movies:  Title 提取上映年份 Year，Genres 拆分为 18 维 multi-hot；
              Year 额外提供 z-score 标准化列（电影固有内容属性，不涉及评分泄漏）
3. 划分（leave-one-out，按时间确定性，可复现）：
   - 每用户正例按时间排序：最后一次 -> test，倒数第二次 -> val
   - 其余全部评分记录（含低分）-> train
   - 无正例用户全部进 train；仅 1 条正例用户 test 有、val 为空
4. 防泄漏：训练集统计特征（电影流行度等）仅由 train 计算；
   评估/推荐时的已观察物品过滤集见 README 与报告说明。

输出（data/processed/）：
  train.csv / val.csv / test.csv   列：UserID,MovieID,Rating,Timestamp,label
  users_features.csv / movies_features.csv
  train_item_popularity.csv        仅训练集统计的电影流行度
输出（results/tables/）：
  08_cleaning_comparison.csv       清洗前后对比
  09_split_stats.csv               划分统计与校验结果

运行：python src/preprocess.py
"""
import numpy as np
import pandas as pd

from config import POSITIVE_THRESHOLD, PROCESSED_DIR, TABLES_DIR
from load_data import (
    AGE_LABELS, GENRES, OCCUPATION_LABELS, load_all,
)


def clean(ratings: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame):
    """清洗并返回 (ratings, 清洗对比记录)。"""
    log = []

    def record(step, before, after, note):
        log.append({"step": step, "rows_before": before,
                    "rows_after": after, "rows_removed": before - after,
                    "note": note})

    n = len(ratings)
    ratings = ratings.drop_duplicates(subset=["UserID", "MovieID"], keep="last")
    record("drop_duplicate_user_movie", n, len(ratings),
           "同一用户对同一电影保留时间戳最新的一条")

    n = len(ratings)
    ratings = ratings[ratings["Rating"].isin([1, 2, 3, 4, 5])]
    record("drop_invalid_rating", n, len(ratings), "评分必须为 1-5 整数")

    n = len(ratings)
    ratings = ratings[ratings["Timestamp"] > 0]
    record("drop_invalid_timestamp", n, len(ratings), "时间戳必须为正")

    n = len(ratings)
    ratings = ratings[ratings["UserID"].isin(users["UserID"])]
    record("drop_unmatched_user", n, len(ratings), "剔除无法关联用户表的评分")

    n = len(ratings)
    ratings = ratings[ratings["MovieID"].isin(movies["MovieID"])]
    record("drop_unmatched_movie", n, len(ratings),
           "剔除无法关联电影表的评分（保证推荐可展示片名/类型）")

    return ratings.reset_index(drop=True), pd.DataFrame(log)


def build_users_features(users: pd.DataFrame) -> pd.DataFrame:
    df = users.copy()
    df["Gender"] = df["Gender"].astype("category")
    df["Gender_F"] = (df["Gender"] == "F").astype(int)  # 二值编码
    # Age 原码本身即有序类别（1<18<25<35<45<50<56），保留顺序信息
    df["Age"] = df["Age"].astype(int)
    df["AgeLabel"] = df["Age"].map(AGE_LABELS)
    df["Occupation"] = df["Occupation"].astype(int)
    df["OccupationLabel"] = df["Occupation"].map(OCCUPATION_LABELS)
    return df


def build_movies_features(movies: pd.DataFrame) -> pd.DataFrame:
    df = movies.copy()
    # 标题末尾 "(1995)" 提取上映年份
    df["Year"] = df["Title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
    # 类型 multi-hot 编码
    genre_split = df["Genres"].str.split("|")
    for g in GENRES:
        df[f"genre_{g}"] = genre_split.apply(lambda xs: int(g in xs))
    # 数值特征标准化示例：Year z-score（内容为固有属性，无评分泄漏）
    df["Year_zscore"] = (df["Year"] - df["Year"].mean()) / df["Year"].std()
    return df


def leave_one_out_split(ratings: pd.DataFrame):
    """按用户 + 时间排序的 leave-one-out 划分，返回带 split 列的全量表。"""
    ratings = ratings.sort_values(
        ["UserID", "Timestamp", "MovieID"]).reset_index(drop=True)
    ratings["split"] = "train"

    pos = ratings[ratings["label"] == 1]
    # 每个用户正例按时间倒序编号：1 -> test，2 -> val
    rank_desc = pos.groupby("UserID")["Timestamp"].rank(
        method="first", ascending=False)
    test_idx = pos.index[rank_desc == 1]
    val_idx = pos.index[rank_desc == 2]
    ratings.loc[test_idx, "split"] = "test"
    ratings.loc[val_idx, "split"] = "val"
    return ratings


def main() -> None:
    ratings, users, movies = load_all()

    # ---------- 1. 清洗 ----------
    ratings, clean_log = clean(ratings, users, movies)
    clean_log.to_csv(TABLES_DIR / "08_cleaning_comparison.csv", index=False)
    print("=== cleaning ===")
    print(clean_log.to_string(index=False))

    # ---------- 2. 标签与编码 ----------
    ratings["label"] = (ratings["Rating"] >= POSITIVE_THRESHOLD).astype(int)
    users_feat = build_users_features(users)
    movies_feat = build_movies_features(movies)
    users_feat.to_csv(PROCESSED_DIR / "users_features.csv", index=False)
    movies_feat.to_csv(PROCESSED_DIR / "movies_features.csv", index=False)

    # ---------- 3. 划分 ----------
    splitted = leave_one_out_split(ratings)
    for name in ["train", "val", "test"]:
        part = splitted[splitted["split"] == name]
        part[["UserID", "MovieID", "Rating", "Timestamp", "label"]].to_csv(
            PROCESSED_DIR / f"{name}.csv", index=False)

    # ---------- 4. 仅训练集统计特征（防泄漏） ----------
    train = splitted[splitted["split"] == "train"]
    popularity = (train.groupby("MovieID")
                  .agg(n_ratings_train=("Rating", "size"),
                       n_positive_train=("label", "sum"),
                       mean_rating_train=("Rating", "mean"))
                  .reset_index())
    popularity.to_csv(PROCESSED_DIR / "train_item_popularity.csv", index=False)

    # ---------- 5. 统计与校验 ----------
    stats = []
    for name in ["train", "val", "test"]:
        part = splitted[splitted["split"] == name]
        ts = pd.to_datetime(part["Timestamp"], unit="s")
        stats.append({
            "split": name, "n_rows": len(part),
            "n_users": part["UserID"].nunique(),
            "n_positive": int(part["label"].sum()),
            "time_min": str(ts.min()), "time_max": str(ts.max()),
        })
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(TABLES_DIR / "09_split_stats.csv", index=False)

    # 校验：同一 (UserID,MovieID) 不跨 split；每用户 val/test 各至多 1 条
    pair_split = splitted.groupby(["UserID", "MovieID"])["split"].nunique()
    assert int((pair_split > 1).sum()) == 0, "存在跨 split 的重复 user-movie 对"
    for name in ["val", "test"]:
        per_user = splitted[splitted["split"] == name].groupby("UserID").size()
        assert int((per_user > 1).sum()) == 0, f"{name} 每用户应至多 1 条"
    assert len(splitted) == len(ratings), "划分后总行数不一致"

    n_users = ratings["UserID"].nunique()
    print("\n=== split stats ===")
    print(stats_df.to_string(index=False))
    print(f"\nusers total = {n_users}")
    print(f"users without test (0 positive) = "
          f"{n_users - splitted[splitted['split'] == 'test']['UserID'].nunique()}")
    print(f"users without val  (<=1 positive) = "
          f"{n_users - splitted[splitted['split'] == 'val']['UserID'].nunique()}")
    print("all split sanity checks passed")
    print(f"\nprocessed files saved -> {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
