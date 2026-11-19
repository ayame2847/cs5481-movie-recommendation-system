# 源代码

数据处理入口为 `dataset_info.py`、`eda.py` 和 `preprocess.py`。推荐算法包括 `lrurec_project.py`，以及 MF 的 `mf.py`、`run_mf.py` 和 `mf_demo.py`；运行命令见根目录 README。

确定语言和技术方案后，可按职责拆分数据处理、三个推荐算法、评估和交互展示。每个模块都应说明输入和输出；三个算法尽量共用数据划分与评估逻辑。

入口和模块确定后，同步更新根目录 README 中的安装和运行说明。
