# LRURec 怎么运行

本模块复用 [官方 LRURec](https://github.com/yueqirex/LRURec) 模型，项目入口自行实现数据适配、训练窗口、验证、测试和推荐导出。不会读取或覆盖官方 `dataset.pkl`。旧版 `lrurec_pipeline.py` 和旧指标未纳入本次提交。

## 环境

实际实验环境：Python 3.13.16、PyTorch 2.14.1+cu130、numpy 2.4.6、pandas 2.2.3、Tesla T4；其他依赖见根目录 requirements.txt。上述版本来自实际运行元数据，不保证不同设备和版本逐位一致。

当前入口要求 CUDA GPU。Colab 中已有可用 PyTorch 时不必重装；其余依赖执行 `pip install -r requirements.txt`。本地需按设备安装支持 CUDA 的 PyTorch，附加依赖入口为 `requirements-lrurec.txt`。

## 准备数据与官方代码

从项目根目录执行，先按主 README 下载 MovieLens 1M 至 `ml-1m/`，然后：

```bash
python src/preprocess.py
git clone https://github.com/yueqirex/LRURec.git ../LRURec_project
git -C ../LRURec_project checkout --detach e02a559bd8c42342f5c53d69cefea2589becb22c
```

不要在官方目录中运行旧版训练入口。项目脚本会核对官方提交编号及模型文件内容。原始电影词表和评分不提交仓库。

## 短跑与正式训练

在项目根目录执行。输出目录必须为空或不存在，避免覆盖已有实验。

```bash
python -u src/lrurec_project.py --project . --official ../LRURec_project --output runs/lrurec-smoke --epochs 2 --smoke
python -u src/lrurec_project.py --project . --official ../LRURec_project --output runs/lrurec-formal --epochs 50
```

短跑每轮最多三个 batch，仅检查前128名验证用户，不生成正式测试指标。正式运行每轮完整验证，以验证 Recall@10 选最佳权重；连续10次验证未改善即早停，再冻结权重做一次最终测试。

训练仅使用训练集正例，窗口最多201项、步长200，构造长度最多200的移位下一项样本；不使用padding预测第一部电影，每个相邻转移只训练一次。这与官方训练入口的样本构造和验证频率不同，结果不能当作官方默认协议的原样复现。

固定共享配置 `seed=42`、正例阈值4；保留三个split中出现的3706部电影作为词表，不用留出评分训练。181部电影没有训练正例，但仍在候选词表中；其他模型对比时必须对齐候选口径，不能自行删除它们。

## 输出文件

- `10_lrurec_metrics.csv`：测试指标，由逐用户结果计算。
- `11_lrurec_test_top5.csv`：测试场景推荐，历史输入为train正例+val，过滤train全部评分+val，允许命中test目标。
- `12_lrurec_final_top5.csv`：最终展示推荐，输入全部正例历史，并过滤train+val+test所有已评分电影。
- `test_per_user.csv`：每名测试用户的目标编号、排名和各项指标。目标编号为模型内部编号，通过 `item_mapping.json` 反查原MovieID。
- `user_population.csv`：每名用户是否具备训练、验证和测试资格，供其他模型对齐。
- `validation_history.csv`：逐轮训练损失和验证指标。
- `run_metadata.json`：参数、环境、数据与代码哈希、实际参数量、耗时、最佳轮次和模型校验值。
- `best_model.pth`：最佳权重，仅保存在本地或个人备份，不提交GitHub。

已提交的小表在 `results/tables/`；逐用户结果与元数据在 `results/lrurec/`；实际执行的Notebook在 `notebooks/cs5481_LRURec.ipynb`。Notebook清除了上传字节和控件，未修改算法结果。

Colab 中 `/content` 是临时存储。训练完成后下载整个实验目录和执行后的Notebook。权重不包含在复核包中，应另行保存完整备份。

## 比较范围与时间限制

本次训练、验证、测试有效用户均为6035人。不伪造验证答案，不因训练正例为空而回退使用低分。MF、BPR需要使用同一测试用户名单、候选词表和过滤规则，才能进行直接比较；本次提交不代表其他算法已对齐或完成。

验证过滤全部train评分，测试过滤全部train+val评分，序列截断不影响过滤集合。指标逐用户等权平均；每人一个测试目标，因此 Recall@K 等于命中率。并列得分按原MovieID升序处理。

共享划分将未留出的低分记录保留在train，其中可能有晚于验证或测试目标的低分。本模型不把这些低分输入训练序列，但完整历史过滤仍按split定义。这是本次实验的限制，不能声称采用严格现实时间前缀；若改成时间截断协议，须由三种算法统一重跑。
