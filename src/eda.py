"""任务二（上）：数据探索与可视化（EDA）。

生成 results/figures/ 下 6 张图 + results/tables/07_eda_summary.csv 统计摘要。
图表使用英文标签，避免字体问题；中文解读见 reports/A_data_report.md。

运行：python src/eda.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURES_DIR, POSITIVE_THRESHOLD, TABLES_DIR
from load_data import AGE_LABELS, OCCUPATION_LABELS, load_all

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})


def main() -> None:
    ratings, users, movies = load_all()

    # ---------- 图 1：评分分布 ----------
    rating_counts = ratings["Rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(rating_counts.index.astype(str), rating_counts.values, color="#4C78A8")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title("Rating Distribution (MovieLens 1M)")
    for i, v in enumerate(rating_counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig01_rating_distribution.png")
    plt.close(fig)

    # ---------- 图 2：每用户评分数分布 ----------
    user_counts = ratings.groupby("UserID").size()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(user_counts, bins=60, color="#4C78A8", log=True)
    ax.set_xlabel("Ratings per user")
    ax.set_ylabel("Number of users (log scale)")
    ax.set_title("User Activity Distribution")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig02_user_activity.png")
    plt.close(fig)

    # ---------- 图 3：每电影评分数分布（长尾） ----------
    item_counts = ratings.groupby("MovieID").size()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(item_counts, bins=60, color="#F58518", log=True)
    axes[0].set_xlabel("Ratings per movie")
    axes[0].set_ylabel("Number of movies (log scale)")
    axes[0].set_title("Movie Popularity Distribution")
    sorted_counts = item_counts.sort_values(ascending=False).to_numpy()
    axes[1].plot(range(1, len(sorted_counts) + 1), sorted_counts, color="#F58518")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Movie rank (log)")
    axes[1].set_ylabel("Ratings (log)")
    axes[1].set_title("Long-tail: Popularity by Rank")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig03_item_popularity.png")
    plt.close(fig)

    # ---------- 图 4：评分随时间变化 ----------
    ts = pd.to_datetime(ratings["Timestamp"], unit="s")
    monthly = ts.dt.to_period("M").value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(monthly.index.astype(str), monthly.values, color="#54A24B")
    ax.set_xlabel("Month")
    ax.set_ylabel("Ratings")
    ax.set_title("Ratings over Time")
    ax.set_xticks(ax.get_xticks()[::6])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig04_ratings_over_time.png")
    plt.close(fig)

    # ---------- 图 5：电影类型分布（一部电影可属多个类型） ----------
    genre_counts = movies["Genres"].str.split("|").explode().value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    genre_counts.sort_values().plot.barh(ax=ax, color="#B279A2")
    ax.set_xlabel("Number of movies")
    ax.set_title("Genre Distribution")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig05_genre_distribution.png")
    plt.close(fig)

    # ---------- 图 6：用户人口统计 ----------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    users["Gender"].value_counts().plot.bar(
        ax=axes[0], color=["#4C78A8", "#E45756"], rot=0)
    axes[0].set_title("Gender")
    age_order = list(AGE_LABELS.keys())
    age_counts = users["Age"].value_counts().reindex(age_order)
    axes[1].bar([AGE_LABELS[a] for a in age_order], age_counts.values,
                color="#4C78A8")
    axes[1].set_title("Age Group")
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", fontsize=8)
    occ = users["Occupation"].map(OCCUPATION_LABELS).value_counts()
    occ.sort_values().plot.barh(ax=axes[2], color="#72B7B2")
    axes[2].set_title("Occupation")
    axes[2].tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig06_demographics.png")
    plt.close(fig)

    # ---------- 统计摘要表 ----------
    n_users = ratings["UserID"].nunique()
    n_movies_rated = ratings["MovieID"].nunique()
    n_ratings = len(ratings)
    sparsity = 1 - n_ratings / (n_users * movies.shape[0])
    summary = pd.DataFrame([
        ("n_users", n_users, "评分用户数"),
        ("n_movies_total", movies.shape[0], "电影总数"),
        ("n_movies_rated", n_movies_rated, "有评分记录的电影数"),
        ("n_ratings", n_ratings, "评分总数"),
        ("rating_mean", round(ratings["Rating"].mean(), 4), "评分均值"),
        ("rating_median", float(ratings["Rating"].median()), "评分中位数"),
        ("positive_ratio(>=4)", round((ratings["Rating"] >= POSITIVE_THRESHOLD).mean(), 4),
         "正例（评分>=4）占比"),
        ("user_item_sparsity", round(sparsity, 6), "用户-电影矩阵稀疏度"),
        ("ratings_per_user_mean", round(user_counts.mean(), 2), "每用户平均评分数"),
        ("ratings_per_user_median", float(user_counts.median()), "每用户评分数中位数"),
        ("ratings_per_user_min", int(user_counts.min()), "每用户最少评分数"),
        ("ratings_per_user_max", int(user_counts.max()), "每用户最多评分数"),
        ("ratings_per_item_mean", round(item_counts.mean(), 2), "每电影平均评分数"),
        ("ratings_per_item_median", float(item_counts.median()), "每电影评分数中位数"),
        ("ratings_per_item_max", int(item_counts.max()), "每电影最多评分数"),
        ("first_rating_time", str(ts.min()), "最早评分时间"),
        ("last_rating_time", str(ts.max()), "最晚评分时间"),
    ], columns=["metric", "value", "description_zh"])
    summary.to_csv(TABLES_DIR / "07_eda_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nfigures saved -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
