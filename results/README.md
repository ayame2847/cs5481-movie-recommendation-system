# 实验结果

- `tables/`：整理后的评价结果和推荐对比表。
- `figures/`：数据分布、算法对比和推荐示例图。

`tables/` 和 `figures/` 保存 A 部分结果；`lrurec/`、`mf/` 保存相应算法的详细记录。MF 指标与推荐表以 `tables/mf_` 开头，实验配置和复现过程见 `docs/实验记录_mf.md`。

提交结果时，附上对应实验记录、配置、代码提交编号和生成方法。表格注明指标、数据划分和算法名称；图注明标题、坐标和单位（如适用）。不要填写没有实际跑出的数值。

原始日志、大体积中间文件和模型权重放本地 `runs/`、`logs/` 或 `checkpoints/`，这些目录默认不提交。

BPR 的独立实验见 [指标](tables/bpr_metrics.csv)和[推荐示例](tables/bpr_recommendations.csv)。组内比较使用 [同口径验证](tables/bpr_shared_validation.csv)和[同口径测试](tables/bpr_shared_metrics.csv)，由 `python src/bpr.py train --shared` 生成。共同口径来自 LRURec 的用户资格表与电影映射，验证和测试各 6,035 人，电影词表 3,706 部。
