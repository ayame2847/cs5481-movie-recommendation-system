"""全局配置加载：路径、随机种子、标签阈值、划分方式。

所有脚本通过 `from config import CFG, PATHS, SEED` 使用统一配置。
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

SEED = int(CFG["seed"])
POSITIVE_THRESHOLD = int(CFG["labels"]["positive_threshold"])


def _p(key: str) -> Path:
    path = ROOT / CFG["paths"][key]
    path.mkdir(parents=True, exist_ok=True)
    return path


RAW_DIR = ROOT / CFG["paths"]["raw_dir"]
PROCESSED_DIR = _p("processed_dir")
TABLES_DIR = _p("tables_dir")
FIGURES_DIR = _p("figures_dir")
REPORTS_DIR = _p("reports_dir")
