# 数据放置说明

本项目使用 MovieLens 1M。请从 [GroupLens 官方页面](https://grouplens.org/datasets/movielens/1m/) 下载，并先阅读压缩包中的 README。

解压后，把原始文件保留在下面的位置，不直接修改：

```text
data/raw/ml-1m/
├── README
├── ratings.dat
├── movies.dat
└── users.dat
```

这三个数据文件分别存放评分、电影和用户信息，使用 `::` 分隔字段。读取时按数据包说明核对字段和编码，不把它们当成带表头的普通逗号分隔 CSV。

清洗、编码和数据划分等产物放到 `data/processed/`。实际实现后，在这里补充产物名称、字段含义和生成命令，并记录原始数据版本或校验值。

`raw/` 和 `processed/` 的内容默认不会提交，只有 `.gitkeep` 用于保留目录。原始数据不在仓库内再分发；不要用 `git add -f` 绕过忽略规则上传。
