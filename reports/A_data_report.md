# A 部分：数据集信息、探索与预处理报告

> CS5481 Group Project — Movie Recommendation（MovieLens 1M）
> 本文件为技术报告中"Dataset information（20%）+ Data exploration and preprocessing（20%）"部分的草稿，所有数字均由 `src/` 下脚本实际运行生成，对应结果表与图见 `results/`。

## 1. 数据集信息（Dataset Information）

### 1.1 数据来源

MovieLens 1M（GroupLens），官方下载 `https://files.grouplens.org/datasets/ml-1m.zip`，包含三张表：`ratings.dat`、`users.dat`、`movies.dat`，均为 `::` 分隔、无表头。引用：Harper & Konstan (2015), *The MovieLens Datasets: History and Context*, ACM TiiS 5(4), Article 19, DOI: 10.1145/2827872。原始数据不再分发、不入库。

### 1.2 形状（shape）

| 数据集 | 行数 | 列数 |
|---|---:|---:|
| ratings | 1,000,209 | 4 |
| users | 6,040 | 5 |
| movies | 3,883 | 3 |
| comparison（合并视图） | 1,000,209 | 10 |

结果表：`results/tables/01_shapes.csv`。

### 1.3 缺失值（null values）

- 三张原始表及合并视图的**缺失值总数均为 0**，**不存在含缺失值的列**。
- 检测结果：`results/tables/02_null_summary.csv`（每列缺失数 + 各表 `__TOTAL__` 行）。

因此本项目无需缺失值插补；预处理重点转为类型处理、编码与划分（见第 3 节）。

### 1.4 对比数据集（comparison dataset）

通过 `ratings ⋈ users ⋈ movies`（均为主键关联，`many_to_one` 校验）构建全量分析视图 **comparison dataset**：每条评分记录同时携带用户人口统计属性（Gender/Age/Occupation/Zip-code）与电影属性（Title/Genres），共 1,000,209 行 × 10 列，合并后无缺失，说明所有评分均可完整关联到用户与电影。

- 文件：`data/processed/comparison_dataset.csv`（约 73 MB，本地文件，不入库，可由 `src/dataset_info.py` 重建）
- 对比摘要：`results/tables/06_comparison_dataset_summary.csv`（各表行/列/缺失/内存对比）

该视图用于后续的联合探索分析，并作为三个算法共用同一数据基础的"对比基准数据集"。

### 1.5 基本信息、唯一值与数据类型

- `info()` 全文与各表前 3 行样例：`results/tables/05_info.txt`
- 每列唯一值数量：`results/tables/04_nunique.csv`。要点：ratings 中 UserID 6,040 个、MovieID 3,706 个（即 3,883 部电影中 3,706 部有评分）、Rating 5 种取值（1–5）。
- 每列数据类型：`results/tables/03_dtypes.csv`。原始类型：UserID/MovieID/Rating/Timestamp 为 int64，Gender/Title/Genres 为 object，Age/Occupation 为 int64 编码的类别，Zip-code 按字符串读取（含前导零）。

## 2. 数据探索（Data Exploration）

统计摘要见 `results/tables/07_eda_summary.csv`，图见 `results/figures/`。

### 2.1 评分分布（fig01）

评分均值 **3.5816**、中位数 **4**。4 分最多（348,971 条，34.9%），其次 3 分（261,197）与 5 分（226,310）；1 分最少（56,174）。整体明显左偏（高分偏多），**评分 ≥ 4 的正例占比为 57.52%**——这是标签阈值取 4 的直接依据（见 3.2）。

### 2.2 用户活跃度（fig02）

每用户评分数：最少 20、最多 2,314、均值 165.6、中位数 96。均值远大于中位数，呈右偏长尾：少数重度用户贡献大量评分。每用户至少 20 条评分保证了 leave-one-out 划分对所有用户可行。

### 2.3 电影流行度与长尾（fig03）

每电影评分数：均值 269.9、中位数 123.5、最高 3,428。按流行度排序的 rank-count 曲线在 log-log 坐标下近似幂律衰减，呈明显长尾。这提示后续评估需关注**流行度偏差**（popular items 主导推荐），报告中可结合推荐结果讨论。

### 2.4 时间特征（fig04）

评分时间跨度 **2000-04-25 至 2003-02-28**。评分量集中在 2000 年下半年（2000 年 11 月达到峰值，单月 290,793 条），2001 年后显著下降并趋于平稳。评分时间分布极不均匀，因此**按用户时间戳排序划分**比全局随机划分更符合真实推荐场景。

### 2.5 电影类型（fig05）

一部电影可属于多个类型（`|` 分隔，共 18 类）。最多的是 Drama（1,603 部）与 Comedy（1,200 部），其次 Action（503）、Thriller（492）、Romance（471）。

### 2.6 用户人口统计（fig06）

- 性别：男性 4,331（71.7%）、女性 1,709（28.3%），性别不均衡。
- 年龄：25–34 岁最多（2,096，34.7%），其次 35–44（1,193）与 18–24（1,103）；Under 18 最少（222）。
- 职业：college/grad student 最多（759），其次 other/not specified（711）、executive/managerial（679）。

### 2.7 稀疏度

用户-电影评分矩阵稀疏度 = 1 − 1,000,209 / (6,040 × 3,883) = **95.74%**。高稀疏度是协同过滤类方法的主要挑战，也是选择矩阵分解/序列建模的动机之一。

## 3. 数据预处理（Preprocessing）

脚本：`src/preprocess.py`；清洗对比表：`results/tables/08_cleaning_comparison.csv`。

### 3.1 清洗校验

对评分表依次执行五项校验，**本数据集实际剔除行数均为 0**（脚本保留处理逻辑，保证流程可复用于其他数据）：

| 校验项 | 处理规则 | 剔除行数 |
|---|---|---:|
| 重复 user-movie 对 | 保留时间戳最新一条 | 0 |
| 非法评分 | 仅保留 1–5 | 0 |
| 非法时间戳 | 仅保留正数 | 0 |
| 无法关联用户表 | 剔除 | 0 |
| 无法关联电影表 | 剔除（保证推荐可展示片名/类型） | 0 |

### 3.2 相关性标签（relevance labels）

采用显式评分转隐式相关性的常用做法：**Rating ≥ 4 → label = 1（正例），Rating ≤ 3 → label = 0**。理由：

1. 4–5 分在语义上代表"喜欢/推荐"，1–3 分不代表正向偏好；
2. 该阈值下正例占比 57.52%，标签均衡，评估稳定；
3. 三个算法共用同一标签定义，保证公平对比。

train/val/test 文件中同时保留原始 `Rating` 与 `label`：显式模型（如 MF）可用 1–5 分回归，隐式/排序模型（BPR、LRURec）使用正例。

### 3.3 特征编码与标准化

- **users**（`data/processed/users_features.csv`）：Gender 二值编码（`Gender_F`）；Age 保留原始有序类别码（1<18<25<35<45<50<56，顺序即年龄顺序）并附文本标签；Occupation 保留类别码并附文本标签；Zip-code 为字符串标识，不作为数值特征。
- **movies**（`data/processed/movies_features.csv`，23 列）：从 Title 提取上映年份 `Year`（1919–2000，无缺失）；`Genres` 拆分为 **18 维 multi-hot**（`genre_Action` … `genre_Western`）；数值特征标准化示例为 `Year_zscore`（z-score）。年份属于电影固有内容属性，与评分行为无关，不涉及标签泄漏。
- **ratings**：`Timestamp` 可转 datetime（EDA 中已用于时间分析）；划分文件保留原始秒级时间戳，便于序列模型复用。

### 3.4 可复现 train / validation / test 划分

采用 **leave-one-out（按用户、按时间）**，脚本确定性执行，无随机性（全局种子 42 记录于 `configs/config.yaml`，供后续抽样与模型使用）：

1. 每个用户的**正例评分**按时间戳升序排列；
2. 最后一次正例 → **test**，倒数第二次正例 → **val**；
3. 其余**全部**评分记录（含低分与剩余正例）→ **train**。

特殊情况：2 个用户没有任何正例（全部记录进 train，不参与评估）；1 个用户仅 1 条正例（进 test，val 为空）。

同时间戳处理：ML-1M 用户常在一次会话中同秒评多部电影，排序键为 `(UserID, Timestamp, MovieID)`，同秒记录按 MovieID 决定先后次序；因此 train 最后一条正例与 val/test 可能时间戳相等（并列），属正常现象。

低分记录的时间位置：train 保留低分记录供显式模型使用，其中 15,295 条晚于 val 正例、8,263 条晚于 test 正例，**这些记录全部 label=0**；即 train 中不存在任何晚于 val/test 的正例（同秒并列除外），评估目标无未来正例泄漏。隐式/序列模型仅使用 train 正例，严格满足时间方向；显式 MF 使用低分作为辅助信号，且三个模型共用同一 train，保证公平。

划分结果（`results/tables/09_split_stats.csv`）：

| split | 行数 | 用户数 | 正例数 |
|---|---:|---:|---:|
| train | 988,134 | 6,040 | 563,206 |
| val | 6,037 | 6,037 | 6,037 |
| test | 6,038 | 6,038 | 6,038 |

合计 1,000,209 行，与原始评分总数一致。脚本内置断言校验：同一 (UserID, MovieID) 不跨 split；每用户 val/test 各至多 1 条；划分前后总行数一致——全部通过。

### 3.5 防止数据泄漏（leakage prevention）

1. **时间方向**：val/test 均为用户时间上最后的正例，train 中的正例全部不晚于 val/test（同秒并列除外）；晚于 val/test 的 train 记录均为低分辅助信号（见 3.4 量化说明），held-out 正例从未进入训练，模拟"用过去预测未来"。
2. **统计特征只从 train 计算**：电影流行度（`data/processed/train_item_popularity.csv`：train 评分数、train 正例数、train 平均评分）等任何行为统计特征不得混入 val/test 信息；后续新增用户/物品特征必须遵守同一规则。
3. **评估过滤**：评估 val 时，候选集过滤该用户 train 中已观察电影；评估 test 时，过滤 train + val 中已观察电影；最终演示推荐时，过滤该用户全部已观察电影（train+val+test），保证"推荐未评分电影"。
4. **调参与选模只用 val**，test 仅用于最终一次评估。

## 4. 复现方式与文件清单

```bash
pip install -r requirements.txt   # pandas / numpy / matplotlib / pyyaml
python src/dataset_info.py        # 任务一：信息提取 + comparison dataset
python src/eda.py                 # 任务二上：EDA 图与统计
python src/preprocess.py          # 任务二下：清洗、编码、划分、校验
```

环境：Windows + Python 3（内部托管 3.x），全局种子 42。原始数据放于 `ml-1m/`（或修改 `configs/config.yaml` 的 `raw_dir`）。

| 文件 | 内容 |
|---|---|
| `results/tables/01–06` | shape / 缺失 / dtypes / nunique / info / 对比数据集摘要 |
| `results/tables/07_eda_summary.csv` | EDA 统计摘要 |
| `results/tables/08_cleaning_comparison.csv` | 清洗前后对比 |
| `results/tables/09_split_stats.csv` | 划分统计 |
| `results/figures/fig01–fig06` | 评分/用户/电影/时间/类型/人口统计图 |
| `data/processed/train.csv, val.csv, test.csv` | 统一划分（UserID,MovieID,Rating,Timestamp,label） |
| `data/processed/users_features.csv, movies_features.csv` | 编码后特征 |
| `data/processed/train_item_popularity.csv` | 仅训练集统计的流行度 |
| `data/processed/comparison_dataset.csv` | 合并对比数据集（73 MB，不入库） |

## 5. 给 B / C / D / E 的接口约定

- **B（MF）**：读 `train.csv` 的显式评分训练；用 `val.csv` 调参；评估时按 3.5 的过滤规则对候选电影打分排序。
- **C（BPR）**：用 `train.csv` 中 `label=1` 的记录作为隐式正反馈，负采样种子 42。
- **D（LRURec）**：按 `UserID, Timestamp` 排序构造正例序列，序列仅来自 train（评估 val/test 时按标准 leave-one-out 协议拼接对应历史）。
- **E（评估与展示）**：统一指标 Recall@10 / NDCG@10，全量候选（3,706 部有评分电影或 3,883 部全量，需在评估脚本中固定其一并记录）；推荐展示关联 `movies_features.csv` 获取片名与类型。

> 待办（不属于 A）：三个算法的训练调参、统一评估脚本实现、推荐结果对比、展示界面与最终报告整合。
