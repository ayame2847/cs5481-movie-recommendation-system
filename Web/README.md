# Web 交互式展示界面

课程要求中的「Web/mobile presentation (10%)」环节：**输入一个用户（或一个人工构造的偏好档案），页面展示三种算法给出的推荐结果**。

页面是纯静态的 HTML + CSS + JavaScript，不依赖任何第三方库、不依赖后端服务，双击 `index.html` 即可打开。

## 快速开始

```bash
# 方式一：直接双击（file:// 也完全可用）
start webapp/index.html          # Windows

# 方式二：启动一个本地静态服务器
python webapp/serve.py           # 默认 http://127.0.0.1:8000/
python webapp/serve.py --port 8123 --no-browser
```

浏览器要求：Chrome / Edge / Firefox 任一较新版本。页面在 `file://` 下也能正常工作，因为数据是以 `<script src="data/demo_data.js">` 的形式加载的，没有用到 `fetch`（`fetch` 在 `file://` 下会被 CORS 拦截）。

## 页面包含什么

| 标签页 | 内容 |
| --- | --- |
| **已有用户** | 输入 UserID（1–6040）后显示性别/年龄段/职业、训练集评分条数与好评率、偏好类型、最近的好评电影；并排展示 **MF**、**BPR**（LRURec 仅对 5 位示例用户可用）的 Top-10，含分数、类型标签、以及该用户留出正例（leave-one-out test 目标）在全排序中的名次；最后给出三种算法 Top-10 的重合度。 |
| **示例档案（冷启动）** | 勾选若干类型、设定最低平均分与最少评分数，构造一个没有历史评分的新用户。左列是**内容 / 热度基线**（贝叶斯平滑好评率），中列是 **MF 的冷启动回退 `μ + bᵢ`**，用于演示协同过滤在冷启动下会退化成一张与个人无关的榜单。 |
| **结果对比与说明** | 三种算法在测试集上的 Recall@5 / NDCG@5 / Recall@10 / NDCG@10 / MRR@10、候选集规模、测试用户数，以及口径说明（三者同口径，可直接横向比较）。 |

URL 支持深链，便于演示时直接跳转：`index.html#user=42`、`index.html#tab=panel-metrics`。

## 目录结构

```
webapp/
├── index.html                 # 页面结构
├── assets/
│   ├── styles.css             # 样式（响应式，窄屏自动单列）
│   └── app.js                 # 交互逻辑（原生 JS，无构建步骤）
├── data/
│   └── demo_data.js           # 预计算数据包（window.DEMO_DATA，约 2.3 MiB）
├── tools/
│   └── export_demo_data.py    # 生成上面的数据包
└── serve.py                   # 可选的本地静态服务器（标准库）
```

## 重新生成 `data/demo_data.js`

数据包是离线预计算的：页面本身不做任何模型推理。修改模型或数据后，在仓库根目录重新导出即可：

```bash
python webapp/tools/export_demo_data.py
```

脚本会加载 `src/mf.py` 的共享划分与 `runs/mf-formal/best_model.npz`、`artifacts/bpr.npz`，然后写入：

* `meta`：种子、正例阈值、候选集规模、测试用户数、生成时间、运行环境版本；
* `users` / `movies`：用户画像与电影元数据（标题、类型、年份、训练集评分统计、MF 的电影偏置）；
* `recs.mf` / `recs.bpr`：**全部 6,040 位用户**的 Top-10；
* `recs.lrurec`：`results/tables/12_lrurec_final_top5.csv` 中 5 位示例用户的 Top-5；
* `eval_ranks`：三种算法每位测试用户的目标电影名次（MF 来自 `runs/mf-formal/test_per_user.csv`，
  BPR 来自 `results/tables/bpr_shared_test_per_user.csv`，LRURec 来自 `results/lrurec/test_per_user.csv`）；
* `metrics`：MF（`runs/mf-formal/mf_metrics.csv`）、BPR 同口径（`results/tables/bpr_shared_metrics.csv`）、
  LRURec（`results/tables/10_lrurec_metrics.csv`）三份指标表。

导出时的排序规则与评估代码一致：**分数降序、并列时 MovieID 小的在前**，并过滤掉该用户在 train / val / test 中已观察过的电影。
BPR 的 Top-10 与打分都在**共享候选集（3,706 部）**上计算，即把已训练好的 d=64 / 第 30 轮权重限制到共享电影集合后重新排序
（`tools/bpr_shared_eval.py`），因此页面三个算法的列表与指标完全同口径。

## 复现性说明（写报告时请保留）

* **三种算法同口径**：同一份划分、同一候选集（3,706 部电影）、同一 6,035 位测试用户、同样的 Top-N 过滤规则，
  指标可直接比较：MF NDCG@10 = 0.02209450、BPR-MF 0.04581568、LRURec 0.12848255。
* **BPR 的同口径与独立口径**：BPR 独立实验使用其自带划分（候选集 3,883 部、测试用户 6,038 位），
  测试集同样命中 539 人，Recall@10 = 0.08926797、NDCG@10 = 0.04579292；同口径下为
  Recall@10 = 0.08931234、NDCG@10 = 0.04581568，两者相对差异约 0.05%，仅来自分母。
  页面统一采用**同口径**结果，并在卡片与说明区标注。
* **LRURec 只内置 5 位用户**：官方仓库训练需要 GPU + PyTorch，本机无法重训，页面展示的是报告归档的离线运行结果（UserID 1–5）。
* 重新导出的数据文件与最初提交的实验记录在**序列化字节**上不同（行序/换行差异），但语义一致：重新训练 MF 得到的 Recall@10 = 0.042087821、NDCG@10 = 0.022094504，与记录值逐位相同；`final_top5.csv`、`bpr_recommendations.csv`、`12_lrurec_final_top5.csv` 中的示例推荐也全部逐条一致。导出脚本用 `CS5481_ALLOW_INPUT_HASH_MISMATCH=1` 跳过输入哈希校验时，仍会校验用户总体与电影映射。

## 配套的对比材料

* `reports/三算法推荐对比.txt`：三算法推荐结果的文字对比（含截图引用、指标、列表特征、优缺点评价）；
* `reports/figures/fig1 ~ fig4`：用户 #1、用户 #42、指标面板、冷启动示例档案的界面截图；
* `tools/three_way_compare.py`：复算该文档中引用的全部统计量（覆盖度、热度、类型分布、列表一致性、重合度、名次分布）。

## 不使用 Web 界面时的备选展示

`results/tables/` 下的 CSV 与 `reports/` 下的报告已经覆盖同样的内容，界面只是把结果可视化，不引入任何新的计算。
