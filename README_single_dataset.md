# 单数据集评估框架

这个框架用于评估单个数据集的质量，帮助筛选出合适的训练集。与原始的 `run.py` 不同，这个版本不需要对比合成数据和原始数据，而是专注于评估单个数据集的各种质量指标。

## 主要特性

- **单数据集评估**: 只需要提供一个数据集，不需要对比数据
- **多维度评估**: 支持保真度、多样性、隐私性、实用性等多个维度的评估
- **数据集摘要**: 自动生成数据集的基本统计信息
- **灵活配置**: 可以选择特定的评估指标和维度
- **中文界面**: 提供中文的日志输出和结果展示

## 安装依赖

确保已安装 SynEval 的所有依赖：

```bash
pip install -r requirements.txt
```

## 基本用法

### 1. 基本评估（推荐）

```bash
python run_single_dataset.py \
    --data your_dataset.csv \
    --metadata metadata.json
```

这将运行默认的评估维度：摘要、保真度、多样性、隐私性。

### 2. 指定评估维度

```bash
python run_single_dataset.py \
    --data your_dataset.csv \
    --metadata metadata.json \
    --dimensions summary fidelity diversity
```

### 3. 实用评估（需要指定输入输出列）

```bash
python run_single_dataset.py \
    --data your_dataset.csv \
    --metadata metadata.json \
    --dimensions utility \
    --utility-input feature1 feature2 \
    --utility-output target
```

### 4. 选择特定指标

```bash
python run_single_dataset.py \
    --data your_dataset.csv \
    --metadata metadata.json \
    --fidelity-metrics diagnostic quality \
    --diversity-metrics tabular_diversity \
    --privacy-metrics membership_inference exact_matches
```

## 评估维度说明

### 1. Summary（摘要）
- 数据集基本信息（形状、内存使用、重复行数等）
- 各列的基本统计信息
- 缺失值分析

### 2. Fidelity（保真度）
- **diagnostic**: 数据诊断（有效性、结构等）
- **quality**: 数据质量评估
- **text**: 文本质量评估
- **numerical_statistics**: 数值统计评估

### 3. Diversity（多样性）
- **tabular_diversity**: 表格数据多样性（覆盖率、唯一性、熵等）
- **text_diversity**: 文本多样性（词汇、语义、情感多样性）

### 4. Privacy（隐私性）
- **exact_matches**: 精确匹配风险
- **membership_inference**: 成员推理攻击风险
- **tabular_privacy**: 表格隐私指标
- **text_privacy**: 文本隐私指标
- **anonymeter**: Anonymeter 隐私评估

### 5. Utility（实用性）
- **tstr_accuracy**: 训练合成测试真实（TSTR）准确率
- **correlation_analysis**: 跨模态相关性分析

## 元数据文件格式

元数据文件应该是一个 JSON 文件，定义数据集的列类型：

```json
{
  "columns": {
    "review": {"sdtype": "text"},
    "rating": {"sdtype": "numerical"},
    "category": {"sdtype": "categorical"},
    "user_id": {"sdtype": "pii"},
    "price": {"sdtype": "numerical"}
  }
}
```

支持的列类型：
- `text`: 文本数据
- `numerical`: 数值数据
- `categorical`: 分类数据
- `pii`: 个人身份信息
- `time`: 时间数据

## 输出结果

评估结果将保存为 JSON 文件，包含以下信息：

1. **数据集摘要**: 基本统计信息
2. **各维度评估结果**: 详细的评估指标和分数
3. **错误信息**: 如果某个评估失败，会记录错误信息

## 使用场景

### 1. 训练集筛选
```bash
# 评估候选训练集的质量
python run_single_dataset.py \
    --data candidate_training_set.csv \
    --metadata metadata.json \
    --dimensions summary fidelity diversity
```

### 2. 数据质量检查
```bash
# 检查数据集的整体质量
python run_single_dataset.py \
    --data dataset_to_check.csv \
    --metadata metadata.json \
    --fidelity-metrics diagnostic quality
```

### 3. 隐私风险评估
```bash
# 评估数据集的隐私风险
python run_single_dataset.py \
    --data sensitive_dataset.csv \
    --metadata metadata.json \
    --dimensions privacy \
    --privacy-metrics membership_inference exact_matches anonymeter
```

### 4. 机器学习任务评估
```bash
# 评估数据集在特定任务上的实用性
python run_single_dataset.py \
    --data ml_dataset.csv \
    --metadata metadata.json \
    --dimensions utility \
    --utility-input feature1 feature2 feature3 \
    --utility-output target_label
```

## 高级选项

### 生成图表
```bash
python run_single_dataset.py \
    --data your_dataset.csv \
    --metadata metadata.json \
    --plot
```

### 设置日志级别
```bash
python run_single_dataset.py \
    --data your_dataset.csv \
    --metadata metadata.json \
    --log-level DEBUG
```

### 自定义输出文件
```bash
python run_single_dataset.py \
    --data your_dataset.csv \
    --metadata metadata.json \
    --output custom_results.json
```

## 注意事项

1. **内存使用**: 大型数据集可能需要较多内存，建议在评估前检查数据集大小
2. **隐私评估**: 某些隐私评估可能需要较长时间，特别是大型数据集
3. **依赖项**: 确保所有必要的依赖项都已正确安装
4. **GPU 加速**: 如果可用，框架会自动使用 GPU 加速某些计算

## 故障排除

### 常见错误

1. **导入错误**: 确保在 SynEval 目录下运行脚本
2. **内存不足**: 对于大型数据集，考虑使用数据采样
3. **依赖缺失**: 运行 `pip install -r requirements.txt` 安装所有依赖

### 获取帮助

```bash
python run_single_dataset.py --help
```

这将显示所有可用的命令行选项和说明。 