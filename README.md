# 电影推荐系统

CS5481 小组项目，使用 MovieLens 1M 数据集，为用户推荐没有评分过的电影，并比较三种推荐算法的效果。

## 目前进度

- **已完成（A：数据部分）**：数据集信息提取、EDA 可视化、清洗与特征编码、统一 leave-one-out 划分，全部脚本已跑通并通过校验，结果见 `results/`，报告草稿见 `reports/A_data_report.md`。
- **已完成（LRURec）**：完成项目数据适配、单配置训练、验证集选模、统一过滤与逐用户测试评估，以及测试/最终展示两类推荐导出。测试6035名用户，Recall@10为0.243413，NDCG@10为0.128483；见 [运行说明](docs/LRURec运行说明.md) 和 [实验记录](docs/实验记录_lrurec.md)。
- **已完成（MF）**：基于 A 的统一数据实现带偏置矩阵分解，验证集选择 32 维、正则化 10、3 轮。测试 6035 名用户，Recall@10 为 0.042088，NDCG@10 为 0.022095；见 [运行说明](docs/MF运行说明.md)、[实验记录](docs/实验记录_mf.md) 和 [报告](reports/B_MF_report.md)。
- **待做**：BPR 的同口径评估与整合（实现见 PR #1），三个算法的推荐结果对比、展示界面、最终报告与5分钟展示。LRURec当前只有单种子固定配置结果，后续比较须对齐用户范围和候选集合。

## 要做什么

- 了解数据：查看用户、电影和评分数据，统计数据量、缺失值、字段类型和分布。✅
- 处理数据：整理特征，固定训练集、验证集和测试集的划分，避免把验证和测试信息用进训练。✅
- 比较算法：按课程要求选 **正好三个算法**，分别来自课堂内容、指定参考算法和指定近期算法。
- 给出推荐：针对选定用户，每种算法推荐至少 5 部不在已观察评分历史中的电影，展示片名和类型，并比较结果。
- 完成展示：用网页、移动端、Notebook 或等效交互界面接收用户或示例档案，展示推荐。

课程要求和待确认事项见 [课程要求](docs/课程要求.md)。

## 目录怎么放

```text
.
├── README.md                 # 项目介绍和使用入口
├── requirements.txt          # Python 依赖
├── configs/config.yaml       # 全局配置：种子 42、正例阈值 4、路径
├── data/
│   └── processed/            # 划分与特征文件（本地生成，不入库）
├── docs/                     # 课程要求、协作约定、实验记录模板
├── ml-1m/                    # 原始数据（本地，不入库）
├── notebooks/                # 数据探索和临时实验
├── src/                      # 数据处理、推荐算法、评估和展示代码
│   ├── config.py             # 配置加载
│   ├── load_data.py          # 原始数据加载与编码表
│   ├── dataset_info.py       # 任务一：数据集信息提取
│   ├── eda.py                # 任务二上：EDA 图与统计
│   └── preprocess.py         # 任务二下：清洗、编码、划分、校验
├── results/                  # 结果表和图
│   ├── tables/               # 01–09 结果表
│   └── figures/              # fig01–fig06
└── reports/                  # 技术报告和展示材料
    └── A_data_report.md      # A 部分报告草稿（数据信息 + 探索预处理）
```

## 怎么开始

1. 克隆仓库并进入项目目录（私有仓库，需要访问权限）：

   ```bash
   git clone https://github.com/ayame2847/cs5481-movie-recommendation-system.git
   cd cs5481-movie-recommendation-system
   ```

2. 从 [GroupLens 官方页面](https://grouplens.org/datasets/movielens/1m/) 下载 MovieLens 1M，解压为项目根目录下的 `ml-1m/`（或修改 `configs/config.yaml` 的 `raw_dir`）。原始数据不提交仓库。

3. 安装依赖并复现 A 部分全部结果：

   ```bash
   pip install -r requirements.txt
   python src/dataset_info.py   # 数据集信息 + comparison dataset
   python src/eda.py            # EDA 图与统计
   python src/preprocess.py     # 清洗、编码、leave-one-out 划分与校验
   ```

## 软件环境和运行方法

- 语言：Python 3（已在 Windows + Python 3.x 实际跑通上述三条命令）。
- 依赖：`requirements.txt`（pandas / numpy / matplotlib / pyyaml）。
- 输入：`ml-1m/` 下三个 `.dat` 文件；输出：`results/tables/`、`results/figures/`、`data/processed/`。
- LRURec使用CUDA GPU，依赖与训练命令见 [LRURec运行说明](docs/LRURec运行说明.md)，Colab过程见 [`notebooks/cs5481_LRURec.ipynb`](notebooks/cs5481_LRURec.ipynb)。MF 使用 CPU，依赖与训练、交互推荐命令见 [MF运行说明](docs/MF运行说明.md)。BPR 和最终展示界面的整合待完成。

## 统一实验口径（三个算法必须共用）

- **相关性标签**：Rating ≥ 4 → 正例（label=1），正例占比 57.52%；train 文件同时保留原始评分供显式模型使用。
- **划分**：leave-one-out，按用户正例时间排序，最后一次正例 → test（6,038 行）、倒数第二次 → val（6,037 行）、其余 → train（988,134 行）。
- **防泄漏**：行为统计特征仅用 train 计算（示例：`train_item_popularity.csv`）；评估 val 过滤 train 已观察电影，评估 test 过滤 train+val；最终展示推荐过滤全部已观察电影；调参只用 val。
- **评估指标**：Recall@10、NDCG@10，全量候选排序（评估脚本实现时固定候选集合并记录）。
- 详细论证见 `reports/A_data_report.md`。

## 随机种子和实验复现

全局随机种子 **42**（`configs/config.yaml`）。当前数据划分按时间确定性执行，无随机性；种子供后续负采样、模型初始化与数据抽样使用，使用时需在实验记录中注明生效位置。数据版本：MovieLens 1M 官方 stable 包（ratings 1,000,209 行）。

## 结果在哪

- 结果表：[`results/tables/`](results/tables/)（01–06 数据集信息表、07 EDA 摘要、08 清洗对比、09 划分统计）
- 结果图：[`results/figures/`](results/figures/)（评分分布、用户活跃度、电影长尾、时间趋势、类型、人口统计）
- A 部分报告草稿：[`reports/A_data_report.md`](reports/A_data_report.md)
- LRURec测试指标：[`results/tables/10_lrurec_metrics.csv`](results/tables/10_lrurec_metrics.csv)。
- 测试场景推荐：[`results/tables/11_lrurec_test_top5.csv`](results/tables/11_lrurec_test_top5.csv)；最终展示推荐：[`results/tables/12_lrurec_final_top5.csv`](results/tables/12_lrurec_final_top5.csv)。
- 用户范围、逐用户测试结果、验证历史和运行元数据：[`results/lrurec/`](results/lrurec/)。

- MF 测试指标：[mf_metrics.csv](results/tables/mf_metrics.csv)；[测试推荐](results/tables/mf_test_top5.csv)、[最终展示推荐](results/tables/mf_final_top5.csv)。
- MF 验证历史、逐用户结果和元数据：[results/mf/](results/mf/)；报告：[B_MF_report.md](reports/B_MF_report.md)。
- [MF / LRURec 对比](results/tables/mf_lrurec_comparison.csv)使用相同用户、候选集合和过滤规则，但训练信号及测试时验证项上下文不同。共享划分可能保留晚于目标的低分训练记录，不能视为严格时间回测；详见 MF 报告。BPR 对比需先对齐口径。

## 数据来源与引用

数据来源：F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems, 5(4), Article 19. [https://doi.org/10.1145/2827872](https://doi.org/10.1145/2827872)

数据页面：[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)。原始数据不随本仓库提交或再分发。采用论文和第三方实现后，在报告中补充对应引用及修改说明。

