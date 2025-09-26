#!/usr/bin/env python3
"""
快速测试数据筛选功能
"""

import pandas as pd
import json
import os
from data_filter import DataFilter

def create_test_data():
    """创建测试数据"""
    data = {
        'review': [
            "这是一个很长的评论，包含了很多详细信息。这个产品非常好用，质量很棒！我非常满意这次购买，强烈推荐给大家。",
            "短评论",
            "这是一个中等长度的评论，包含了一些基本信息。",
            "另一个很长的评论，详细描述了产品的各个方面，包括使用体验、质量评价、性价比分析等。",
            "非常短的评论"
        ],
        'rating': [5, 2, 3, 5, 1],
        'category': ['电子产品', '服装', '食品', '家居', '电子产品'],
        'price': [299.99, 89.50, 15.99, 599.99, 45.00]
    }
    
    df = pd.DataFrame(data)
    
    metadata = {
        "columns": {
            "review": {"sdtype": "text"},
            "rating": {"sdtype": "numerical"},
            "category": {"sdtype": "categorical"},
            "price": {"sdtype": "numerical"}
        }
    }
    
    return df, metadata

def test_text_length_filter():
    """测试文本长度筛选"""
    print("🔍 测试文本长度筛选...")
    
    df, metadata = create_test_data()
    filter_tool = DataFilter(df, metadata)
    
    # 配置指标
    metrics_config = {
        'text_length': {
            'columns': ['review']
        }
    }
    
    # 筛选前3条
    filtered_data, filtered_scores = filter_tool.filter_top_data(metrics_config, top_n=3)
    
    print(f"原始数据: {len(df)} 条")
    print(f"筛选后: {len(filtered_data)} 条")
    print("\n筛选结果:")
    for i, (_, row) in enumerate(filtered_data.iterrows()):
        print(f"{i+1}. 评论长度: {len(row['review'])} 字符")
        print(f"   内容: {row['review'][:30]}...")
        print(f"   得分: {filtered_scores.loc[row.name, 'total_score']:.3f}")
        print()

def test_multi_metric_filter():
    """测试多指标筛选"""
    print("🔍 测试多指标筛选...")
    
    df, metadata = create_test_data()
    filter_tool = DataFilter(df, metadata)
    
    # 配置多个指标
    metrics_config = {
        'text_length': {
            'columns': ['review']
        },
        'text_quality': {
            'columns': ['review']
        }
    }
    
    # 设置权重
    weights = {
        'text_length': 0.7,
        'text_quality': 0.3
    }
    
    # 筛选前3条
    filtered_data, filtered_scores = filter_tool.filter_top_data(
        metrics_config, 
        top_n=3, 
        weights=weights
    )
    
    print(f"原始数据: {len(df)} 条")
    print(f"筛选后: {len(filtered_data)} 条")
    print(f"权重配置: {weights}")
    print("\n筛选结果:")
    for i, (_, row) in enumerate(filtered_data.iterrows()):
        print(f"{i+1}. 评论: {row['review'][:30]}...")
        print(f"   长度得分: {filtered_scores.loc[row.name, 'score_text_length']:.3f}")
        print(f"   质量得分: {filtered_scores.loc[row.name, 'score_text_quality']:.3f}")
        print(f"   总分: {filtered_scores.loc[row.name, 'total_score']:.3f}")
        print()

def test_numerical_filter():
    """测试数值筛选"""
    print("🔍 测试数值筛选...")
    
    df, metadata = create_test_data()
    filter_tool = DataFilter(df, metadata)
    
    # 配置数值指标
    metrics_config = {
        'numerical_value': {
            'columns': ['rating', 'price'],
            'type': 'value'
        }
    }
    
    # 筛选前3条
    filtered_data, filtered_scores = filter_tool.filter_top_data(metrics_config, top_n=3)
    
    print(f"原始数据: {len(df)} 条")
    print(f"筛选后: {len(filtered_data)} 条")
    print("\n筛选结果:")
    for i, (_, row) in enumerate(filtered_data.iterrows()):
        print(f"{i+1}. 评分: {row['rating']}, 价格: {row['price']}")
        print(f"   数值得分: {filtered_scores.loc[row.name, 'score_numerical_value']:.3f}")
        print(f"   总分: {filtered_scores.loc[row.name, 'total_score']:.3f}")
        print()

def main():
    """主函数"""
    print("🚀 数据筛选功能快速测试")
    print("=" * 40)
    
    # 测试文本长度筛选
    test_text_length_filter()
    
    # 测试多指标筛选
    test_multi_metric_filter()
    
    # 测试数值筛选
    test_numerical_filter()
    
    print("✅ 测试完成!")

if __name__ == "__main__":
    main() 