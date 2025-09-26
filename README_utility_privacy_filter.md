# Utility-Privacy数据筛选器

这个工具专门用于筛选出情感评分匹配度高且PII/实体数量少的数据，基于 SynEval 框架的 Utility 和 Privacy 维度。

## 功能特点

- **Utility维度 (50%权重)**：专注于情感与评分的匹配度，确保：
  - 正面情感 → 高评分 (4-5分)
  - 中性情感 → 中等评分 (3分)
  - 负面情感 → 低评分 (1-2分)
- **Privacy维度 (50%权重)**：专注于减少PII/实体数量，包括：
  - 使用Flair NER模型检测命名实体
  - 实体数量越少，得分越高
  - 支持简化计算模式（当NER模型不可用时）
- **Fidelity和Diversity维度**：不参与评分 (0%权重)

## 文件说明

- `utility_privacy_filter.py`：主要的筛选器实现
- `example_utility_privacy_filter.py`：使用示例脚本
- `README_utility_privacy_filter.md`：本文档

## 使用方法

### 方法1：使用命令行

```bash
cd SynEval
python utility_privacy_filter.py \
    --data real_10k.csv \
    --metadata metadata.json \
    --output filtered_utility_privacy_data.csv \
    --scores-output utility_privacy_scores.csv \
    --top-n 50 \
    --text-columns title text \
    --rating-column rating
```

### 方法2：使用示例脚本

```bash
cd SynEval
python example_utility_privacy_filter.py
```

### 方法3：在代码中使用

```python
from utility_privacy_filter import UtilityPrivacyFilter
import pandas as pd
import json

# 加载数据
data = pd.read_csv("real_10k.csv")
with open("metadata.json", "r") as f:
    metadata = json.load(f)

# 创建筛选器
config = {
    'text_columns': ['title', 'text'],
    'rating_column': 'rating'
}
data_filter = UtilityPrivacyFilter(data, metadata)

# 筛选数据
filtered_data, filtered_scores = data_filter.filter_top_data(config, top_n=50)

# 保存结果
filtered_data.to_csv("filtered_utility_privacy_data.csv", index=False)
filtered_scores.to_csv("utility_privacy_scores.csv", index=False)
```

## 参数说明

### 命令行参数

- `--data`：输入数据文件路径 (必需)
- `--metadata`：元数据文件路径 (必需)
- `--output`：筛选后数据输出路径 (必需)
- `--scores-output`：得分数据输出路径 (可选)
- `--top-n`：要筛选的数据条数，默认50
- `--text-columns`：要分析的文本列名，如 `title text`
- `--rating-column`：评分列名，默认 `rating`
- `--log-level`：日志级别，默认INFO

### 配置参数

- `text_columns`：要分析的文本列列表，如 `['title', 'text']`
- `rating_column`：评分列名，如 `'rating'`

## 输出文件

### 筛选后的数据文件
包含筛选出的前N条数据，格式与原始数据相同。

### 得分文件
包含每条数据的详细得分信息：
- `utility_score`：情感评分匹配度得分
- `privacy_score`：PII/实体数量得分
- `total_score`：加权总分

## 评分算法

### Utility维度 (50%权重)
1. **情感分析**：使用TextBlob计算文本的情感极性 (-1到1)
2. **期望评分映射**：
   - 负面情感(-1到0) → 期望评分1-3
   - 中性情感(0) → 期望评分3
   - 正面情感(0到1) → 期望评分3-5
3. **匹配度计算**：`1.0 - abs(实际评分 - 期望评分) / 4.0`

### Privacy维度 (50%权重)
1. **NER实体检测**：使用Flair模型检测命名实体
2. **实体计数**：统计每个文本中的实体数量
3. **得分计算**：`1.0 - (实体数量 / 最大实体数量)`
4. **简化模式**：当NER不可用时，基于文本特征估算：
   - 数字、大写字母、特殊符号的存在性
   - 特征越少，实体可能越少

### 总分计算
```
总分 = 0.50 × Utility得分 + 0.50 × Privacy得分
```

## 示例输出

```
📊 Utility-Privacy数据筛选结果摘要
==================================================
原始数据条数: 10000
筛选后数据条数: 50

权重配置:
  Utility维度: 50% (情感评分匹配度)
  Privacy维度: 50% (PII/实体数量)
  Fidelity维度: 0% (不参与评分)
  Diversity维度: 0% (不参与评分)

筛选后数据的基本统计:
平均总分: 0.823
最高总分: 0.912
最低总分: 0.756
Utility (情感评分匹配度) 平均得分: 0.847
Privacy (PII/实体数量) 平均得分: 0.799

情感评分匹配度分析:
------------------------------
最高匹配度: 0.923
最低匹配度: 0.756
平均匹配度: 0.847

PII/实体数量分析:
------------------------------
最高隐私得分: 0.912
最低隐私得分: 0.723
平均隐私得分: 0.799
```

## 注意事项

1. **依赖包**：需要安装 `pandas`, `numpy`, `textblob`, `flair`
2. **NER模型**：首次运行时会下载Flair NER模型
3. **性能**：NER计算较慢，大数据集处理时间较长
4. **简化模式**：当NER不可用时自动切换到基于特征的估算
5. **情感分析**：基于TextBlob的英文情感分析

## 与原始SynEval筛选器的区别

| 特性 | 原始筛选器 | Utility-Privacy筛选器 |
|------|------------|----------------------|
| Fidelity权重 | 25% | 0% |
| Diversity权重 | 25% | 0% |
| Utility权重 | 25% | 50% |
| Privacy权重 | 25% | 50% |
| 主要目标 | 综合评估 | 情感匹配+隐私保护 |
| 适用场景 | 通用数据筛选 | 情感一致性+隐私安全 |

## 应用场景

1. **情感分析训练数据**：确保情感标签与文本内容一致
2. **隐私敏感应用**：减少包含个人信息的训练数据
3. **数据质量优化**：提高数据的一致性和安全性
4. **合规性要求**：满足数据隐私保护法规要求 