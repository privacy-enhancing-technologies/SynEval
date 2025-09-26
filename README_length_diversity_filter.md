# 长度-多样性数据筛选器

这个工具专门用于筛选出文本长度最长和语义多样性最丰富的数据，基于 SynEval 框架的 Fidelity 和 Diversity 维度。

## 功能特点

- **Fidelity维度 (50%权重)**：专注于文本长度，筛选出 title 和 text 最长的数据
- **Diversity维度 (50%权重)**：专注于语义多样性，包括：
  - 词汇丰富度 (Type-Token Ratio) - 30%
  - 句子复杂度 - 30%
  - 词汇复杂度 (长词比例) - 20%
  - 标点符号多样性 - 20%
- **Utility和Privacy维度**：不参与评分 (0%权重)

## 文件说明

- `length_diversity_filter.py`：主要的筛选器实现
- `example_length_diversity_filter.py`：使用示例脚本
- `README_length_diversity_filter.md`：本文档

## 使用方法

### 方法1：使用命令行

```bash
cd SynEval
python length_diversity_filter.py \
    --data real_10k.csv \
    --metadata metadata.json \
    --output filtered_length_diversity_data.csv \
    --scores-output length_diversity_scores.csv \
    --top-n 50 \
    --text-columns title text
```

### 方法2：使用示例脚本

```bash
cd SynEval
python example_length_diversity_filter.py
```

### 方法3：在代码中使用

```python
from length_diversity_filter import LengthDiversityFilter
import pandas as pd
import json

# 加载数据
data = pd.read_csv("real_10k.csv")
with open("metadata.json", "r") as f:
    metadata = json.load(f)

# 创建筛选器
config = {
    'text_columns': ['title', 'text']
}
data_filter = LengthDiversityFilter(data, metadata)

# 筛选数据
filtered_data, filtered_scores = data_filter.filter_top_data(config, top_n=50)

# 保存结果
filtered_data.to_csv("filtered_length_diversity_data.csv", index=False)
filtered_scores.to_csv("length_diversity_scores.csv", index=False)
```

## 参数说明

### 命令行参数

- `--data`：输入数据文件路径 (必需)
- `--metadata`：元数据文件路径 (必需)
- `--output`：筛选后数据输出路径 (必需)
- `--scores-output`：得分数据输出路径 (可选)
- `--top-n`：要筛选的数据条数，默认50
- `--text-columns`：要分析的文本列名，如 `title text`
- `--log-level`：日志级别，默认INFO

### 配置参数

- `text_columns`：要分析的文本列列表，如 `['title', 'text']`

## 输出文件

### 筛选后的数据文件
包含筛选出的前N条数据，格式与原始数据相同。

### 得分文件
包含每条数据的详细得分信息：
- `fidelity_score`：文本长度得分
- `diversity_score`：语义多样性得分
- `total_score`：加权总分

## 评分算法

### Fidelity维度 (50%权重)
- 计算每个文本列的长度
- 标准化到0-1范围
- 多个文本列取平均值

### Diversity维度 (50%权重)
综合计算以下指标：
- **词汇丰富度 (30%)**：Type-Token Ratio = 唯一词汇数 / 总词汇数
- **句子复杂度 (30%)**：平均句子长度
- **词汇复杂度 (20%)**：长词(>6字符)比例
- **标点多样性 (20%)**：标点符号种类和数量

### 总分计算
```
总分 = 0.50 × Fidelity得分 + 0.50 × Diversity得分
```

## 示例输出

```
📊 长度-多样性数据筛选结果摘要
==================================================
原始数据条数: 10000
筛选后数据条数: 50

权重配置:
  Fidelity维度: 50% (文本长度)
  Diversity维度: 50% (语义多样性)
  Utility维度: 0% (不参与评分)
  Privacy维度: 0% (不参与评分)

筛选后数据的基本统计:
平均总分: 0.847
最高总分: 0.923
最低总分: 0.812
Fidelity (文本长度) 平均得分: 0.856
Diversity (语义多样性) 平均得分: 0.838
```

## 注意事项

1. **依赖包**：需要安装 `pandas`, `numpy`, `textblob`
2. **文本列**：确保指定的文本列存在于数据中
3. **数据格式**：支持CSV格式，需要对应的metadata.json文件
4. **性能**：对于大数据集，语义多样性计算可能较慢

## 与原始SynEval筛选器的区别

| 特性 | 原始筛选器 | 长度-多样性筛选器 |
|------|------------|-------------------|
| Fidelity权重 | 25% | 50% |
| Diversity权重 | 25% | 50% |
| Utility权重 | 25% | 0% |
| Privacy权重 | 25% | 0% |
| 主要目标 | 综合评估 | 文本长度+语义多样性 |
| 适用场景 | 通用数据筛选 | 文本质量优化 | 