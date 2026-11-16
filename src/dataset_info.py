"""任务一：数据集信息提取（评分占比 20%）。

输出（results/tables/）：
  01_shapes.csv                    各数据集 shape
  02_null_summary.csv              每列缺失值数量 + 各表缺失总计 + 缺失列检测
  03_dtypes.csv                    每列数据类型
  04_nunique.csv                   每列唯一值数量
  05_info.txt                      各数据集 info() 全文
  06_comparison_dataset_summary.csv  合并对比数据集与各原始表的对比
并生成合并对比数据集 data/processed/comparison_dataset.csv
（ratings ⋈ users ⋈ movies 的全量分析视图，本地文件，不提交仓库）。

运行：python src/dataset_info.py
"""
import io

import pandas as pd

from config import PROCESSED_DIR, TABLES_DIR
from load_data import load_all


def main() -> None:
    ratings, users, movies = load_all()
    datasets = {"ratings": ratings, "users": users, "movies": movies}

    # 合并对比数据集：评分记录关联用户属性与电影属性
    comparison = (
        ratings.merge(users, on="UserID", how="left", validate="many_to_one")
        .merge(movies, on="MovieID", how="left", validate="many_to_one")
    )
    # 合并后若出现缺失，说明存在无法关联的用户/电影，需要单独报告
    datasets["comparison"] = comparison

    # 1. shape
    shapes = pd.DataFrame(
        [{"dataset": name, "n_rows": df.shape[0], "n_cols": df.shape[1]}
         for name, df in datasets.items()]
    )
    shapes.to_csv(TABLES_DIR / "01_shapes.csv", index=False)

    # 2. 缺失值：每列数量 + 每表总计 + 缺失列检测
    null_rows = []
    for name, df in datasets.items():
        nulls = df.isna().sum()
        for col, n in nulls.items():
            null_rows.append({"dataset": name, "column": col, "n_nulls": int(n)})
        null_cols = [c for c, n in nulls.items() if n > 0]
        null_rows.append({
            "dataset": name, "column": "__TOTAL__",
            "n_nulls": int(nulls.sum()),
        })
        print(f"[{name}] total nulls = {int(nulls.sum())}; "
              f"null columns = {null_cols if null_cols else '无'}")
    pd.DataFrame(null_rows).to_csv(TABLES_DIR / "02_null_summary.csv", index=False)

    # 3. dtypes
    dtypes = pd.DataFrame([
        {"dataset": name, "column": col, "dtype": str(df[col].dtype)}
        for name, df in datasets.items() for col in df.columns
    ])
    dtypes.to_csv(TABLES_DIR / "03_dtypes.csv", index=False)

    # 4. 每列唯一值数量
    nunique = pd.DataFrame([
        {"dataset": name, "column": col, "n_unique": int(df[col].nunique())}
        for name, df in datasets.items() for col in df.columns
    ])
    nunique.to_csv(TABLES_DIR / "04_nunique.csv", index=False)

    # 5. info() 全文
    with open(TABLES_DIR / "05_info.txt", "w", encoding="utf-8") as f:
        for name, df in datasets.items():
            buf = io.StringIO()
            df.info(buf=buf)
            f.write(f"===== {name} =====\n{buf.getvalue()}\n")
            f.write(f"--- {name}: head(3) ---\n{df.head(3).to_string()}\n\n")

    # 6. 对比数据集摘要
    summary = pd.DataFrame([
        {
            "dataset": name,
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "total_nulls": int(df.isna().sum().sum()),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
            "description": desc,
        }
        for (name, df), desc in zip(datasets.items(), [
            "原始评分表 UserID::MovieID::Rating::Timestamp",
            "原始用户表 UserID::Gender::Age::Occupation::Zip-code",
            "原始电影表 MovieID::Title::Genres",
            "合并对比数据集 ratings ⋈ users ⋈ movies（全量分析视图）",
        ])
    ])
    summary.to_csv(TABLES_DIR / "06_comparison_dataset_summary.csv", index=False)

    # 合并视图落盘（大文件，仅供本地后续分析，不入库）
    comparison_path = PROCESSED_DIR / "comparison_dataset.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\n=== shapes ===")
    print(shapes.to_string(index=False))
    print(f"\ncomparison dataset saved -> {comparison_path}")
    print(f"tables saved -> {TABLES_DIR}")


if __name__ == "__main__":
    main()
