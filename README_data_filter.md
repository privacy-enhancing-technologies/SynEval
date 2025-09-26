# 基于SynEval框架的数据筛选工具

这个工具基于SynEval评估框架，根据多个维度的评估指标筛选出得分最高的前N条数据。特别适用于训练集筛选、数据质量优化等场景。

## 主要特性

- **基于SynEval框架**: 使用专业的评估指标和算法
- **多维度评估**: 涵盖Fidelity、Utility、Diversity、Privacy四个维度
- **固定权重算法**: 每个大类占25%权重，小类平均分配
- **智能评分**: 自动计算各维度的综合得分

## 评估维度与权重

### 1. Fidelity维度 (25%权重)
- **文本长度** (8.33%): 文本越长得分越高
- **诊断得分** (8.33%): 数据诊断的overall score，分数越高越好
- **质量得分** (8.33%): 数据质量的overall score，分数越高越好

### 2. Utility维度 (25%权重)
- **情感评分匹配度** (25%): rating与review的情感匹配程度
  - 正面评价 → 期望高评分
  - 负面评价 → 期望低评分
  - 中性评价 → 期望中等评分(3.0)

### 3. Diversity维度 (25%权重)
- **分类分布相似性** (12.5%): 每个categorical data的distribution_similarity总和
- **语义多样性** (12.5%): 文本的semantic_diversity中的total_mst_weight

### 4. Privacy维度 (25%权重)
- **命名实体数量** (25%): named_entities中的total_entities，越少越好

## 基本用法

### 1. 基本筛选

```bash
python data_filter.py \
    --data your_dataset.csv \
    --metadata metadata.json \
    --output filtered_data.csv \
    --top-n 50
```

### 2. 指定列名

```bash
python data_filter.py \
    --data your_dataset.csv \
    --metadata metadata.json \
    --output filtered_data.csv \
    --text-columns review comment \
    --rating-column rating \
    --categorical-columns category brand \
    --top-n 50
```

### 3. 保存得分信息

```bash
python data_filter.py \
    --data your_dataset.csv \
    --metadata metadata.json \
    --output filtered_data.csv \
    --scores-output scores.csv \
    --text-columns review \
    --rating-column rating \
    --top-n 50
```

## 评分算法详解

### 1. 文本长度评分
```python
# 标准化到0-1范围
normalized_lengths = (text_lengths - text_lengths.min()) / (text_lengths.max() - text_lengths.min())
```

### 2. 情感评分匹配度
```python
# 计算情感极性
sentiment_scores = TextBlob(text).sentiment.polarity

# 转换为期望评分
expected_ratings = 3.0 + sentiment_scores * 2.0

# 计算匹配度
alignment_scores = 1.0 - abs(actual_ratings - expected_ratings) / 4.0
```

### 3. 分类分布相似性
```python
# 计算稀有度得分
rarity_scores = 1 - (category_frequency / total_count)
```

### 4. 语义多样性
```python
# 词汇丰富度
vocabulary_richness = unique_words / total_words

# 句子复杂度
avg_sentence_length = text_length / sentence_count

# 综合得分
semantic_scores = (vocabulary_richness + avg_sentence_length) / 2
```

### 5. 命名实体评分
```python
# 实体越少得分越高
entity_scores = 1.0 - (entity_counts / max_entity_count)
```

## 输出文件说明

### 1. 筛选后的数据文件
- 包含得分最高的前N条数据
- 保持原始数据的所有列
- 按总分从高到低排序

### 2. 得分文件（可选）
包含每条数据在各个维度上的详细得分：

```csv
,score_text_length,score_diagnostic,score_quality,score_sentiment_alignment,score_categorical_distribution,score_semantic_diversity,score_named_entities,total_score
0,0.083,0.083,0.083,0.250,0.125,0.125,0.250,1.000
1,0.075,0.080,0.082,0.245,0.120,0.122,0.248,0.972
...
```

## 使用场景示例

### 场景1: 筛选高质量评论数据

```bash
python data_filter.py \
    --data reviews.csv \
    --metadata metadata.json \
    --output high_quality_reviews.csv \
    --text-columns review \
    --rating-column rating \
    --categorical-columns category \
    --top-n 100
```

### 场景2: 筛选多样化产品数据

```bash
python data_filter.py \
    --data products.csv \
    --metadata metadata.json \
    --output diverse_products.csv \
    --text-columns description \
    --rating-column user_rating \
    --categorical-columns brand category subcategory \
    --top-n 50
```

### 场景3: 筛选隐私友好的文本数据

```bash
python data_filter.py \
    --data documents.csv \
    --metadata metadata.json \
    --output privacy_safe_documents.csv \
    --text-columns content \
    --rating-column relevance_score \
    --categorical-columns topic \
    --top-n 30
```

## 权重配置说明

### 固定权重算法
- **每个大类权重相等**: Fidelity、Utility、Diversity、Privacy各占25%
- **小类平均分配**: 大类内的小类权重相等
- **总分计算**: 所有得分直接相加，然后标准化到0-1范围

### 权重分配示例
```
Fidelity (25%):
  ├── 文本长度: 8.33%
  ├── 诊断得分: 8.33%
  └── 质量得分: 8.33%

Utility (25%):
  └── 情感评分匹配度: 25%

Diversity (25%):
  ├── 分类分布相似性: 12.5%
  └── 语义多样性: 12.5%

Privacy (25%):
  └── 命名实体数量: 25%
```

## 性能优化建议

1. **大数据集处理**: 对于大型数据集，建议先进行采样测试
2. **内存使用**: 确保有足够的内存处理整个数据集
3. **依赖项**: 确保安装了TextBlob和Flair等依赖

## 故障排除

### 常见错误

1. **TextBlob不可用**: 情感分析功能会降级到默认得分
2. **Flair模型不可用**: 命名实体检测会使用简化算法
3. **列名不存在**: 系统会使用元数据自动推断列类型

### 调试模式

```bash
python data_filter.py \
    --data your_dataset.csv \
    --metadata metadata.json \
    --output filtered_data.csv \
    --text-columns review \
    --log-level DEBUG
```

## 扩展功能

### 自定义权重
可以通过修改代码中的权重配置来调整各维度的重要性：

```python
# 在 calculate_syn_eval_scores 方法中修改权重
fidelity_weight = 0.30  # 增加Fidelity权重
utility_weight = 0.20   # 减少Utility权重
diversity_weight = 0.25
privacy_weight = 0.25
```

### 添加新指标
可以通过继承 `SynEvalDataFilter` 类来添加新的评估指标：

```python
class CustomDataFilter(SynEvalDataFilter):
    def calculate_custom_scores(self, columns):
        # 实现自定义评分逻辑
        pass
```

## 最佳实践

1. **明确目标**: 根据具体需求选择合适的列配置
2. **数据预处理**: 确保数据质量和列名正确
3. **结果验证**: 检查筛选结果是否符合预期
4. **迭代优化**: 根据结果反馈调整配置参数 