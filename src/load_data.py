"""MovieLens 1M 原始数据加载。

三个 .dat 文件均为 `::` 分隔、无表头；movies.dat 含非 UTF-8 字符，使用 latin-1。
"""
import pandas as pd

from config import CFG, RAW_DIR

RATINGS_COLS = ["UserID", "MovieID", "Rating", "Timestamp"]
USERS_COLS = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]
MOVIES_COLS = ["MovieID", "Title", "Genres"]

# README 中的编码表（供预处理与报告使用）
AGE_LABELS = {
    1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44",
    45: "45-49", 50: "50-55", 56: "56+",
}
OCCUPATION_LABELS = {
    0: "other / not specified", 1: "academic/educator", 2: "artist",
    3: "clerical/admin", 4: "college/grad student", 5: "customer service",
    6: "doctor/health care", 7: "executive/managerial", 8: "farmer",
    9: "homemaker", 10: "K-12 student", 11: "lawyer", 12: "programmer",
    13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed",
    17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed",
    20: "writer",
}
GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def load_ratings() -> pd.DataFrame:
    return pd.read_csv(
        RAW_DIR / CFG["files"]["ratings"], sep="::", engine="python",
        header=None, names=RATINGS_COLS,
    )


def load_users() -> pd.DataFrame:
    return pd.read_csv(
        RAW_DIR / CFG["files"]["users"], sep="::", engine="python",
        header=None, names=USERS_COLS, dtype={"Zip-code": str},
    )


def load_movies() -> pd.DataFrame:
    return pd.read_csv(
        RAW_DIR / CFG["files"]["movies"], sep="::", engine="python",
        header=None, names=MOVIES_COLS, encoding="latin-1",
    )


def load_all():
    return load_ratings(), load_users(), load_movies()
