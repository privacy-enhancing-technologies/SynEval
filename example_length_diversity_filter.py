#!/usr/bin/env python3
"""
长度-多样性数据筛选器使用示例

这个脚本展示了如何使用 LengthDiversityFilter 来筛选出
文本长度最长和语义多样性最丰富的数据。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from length_diversity_filter import LengthDiversityFilter
import pandas as pd
import json
import logging

def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 配置参数
    data_file = "real_10k.csv"  # 原始数据文件
    metadata_file = "metadata.json"  # 元数据文件
    output_file = "filtered_length_diversity_data.csv"  # 输出文件
    scores_output_file = "length_diversity_scores.csv"  # 得分文件
    top_n = 50  # 筛选前50条
    
    # 检查文件是否存在
    if not os.path.exists(data_file):
        logger.error(f"数据文件不存在: {data_file}")
        return
    
    if not os.path.exists(metadata_file):
        logger.error(f"元数据文件不存在: {metadata_file}")
        return
    
    try:
        # 加载数据
        logger.info(f"正在加载数据集: {data_file}")
        data = pd.read_csv(data_file)
        
        logger.info(f"正在加载元数据: {metadata_file}")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # 删除未在元数据中定义的 _id
        metadata_columns = metadata.get("columns", {})
        if '_id' in data.columns and '_id' not in metadata_columns:
            logger.warning("在数据中发现 '_id' 列但元数据中未定义 — 正在删除")
            data.drop(columns=['_id'], inplace=True)
        
        # 构建配置 - 使用 title 和 text 列
        config = {
            'text_columns': ['title', 'text'],  # 指定要分析的文本列
        }
        
        # 创建数据筛选器
        logger.info("正在创建长度-多样性数据筛选器...")
        data_filter = LengthDiversityFilter(data, metadata)
        
        # 筛选数据
        logger.info(f"正在筛选前{top_n}条数据...")
        filtered_data, filtered_scores = data_filter.filter_top_data(config, top_n=top_n)
        
        # 保存结果
        logger.info(f"正在保存筛选后的数据到: {output_file}")
        filtered_data.to_csv(output_file, index=False)
        
        logger.info(f"正在保存得分数据到: {scores_output_file}")
        filtered_scores.to_csv(scores_output_file, index=False)
        
        # 显示结果摘要
        print(f"\n📊 长度-多样性数据筛选结果摘要")
        print("=" * 50)
        print(f"原始数据条数: {len(data)}")
        print(f"筛选后数据条数: {len(filtered_data)}")
        
        print(f"\n权重配置:")
        print(f"  Fidelity维度: 50% (文本长度)")
        print(f"  Diversity维度: 50% (语义多样性)")
        print(f"  Utility维度: 0% (不参与评分)")
        print(f"  Privacy维度: 0% (不参与评分)")
        
        print(f"\n筛选后数据的基本统计:")
        print(f"平均总分: {filtered_scores['total_score'].mean():.3f}")
        print(f"最高总分: {filtered_scores['total_score'].max():.3f}")
        print(f"最低总分: {filtered_scores['total_score'].min():.3f}")
        
        print(f"Fidelity (文本长度) 平均得分: {filtered_scores['fidelity_score'].mean():.3f}")
        print(f"Diversity (语义多样性) 平均得分: {filtered_scores['diversity_score'].mean():.3f}")
        
        # 显示前几条数据的详细信息
        print(f"\n前5条筛选数据的详细信息:")
        print("-" * 50)
        for i, (idx, row) in enumerate(filtered_data.head().iterrows()):
            score_row = filtered_scores.loc[idx]
            print(f"第{i+1}条 (总分: {score_row['total_score']:.3f}):")
            print(f"  Fidelity得分: {score_row['fidelity_score']:.3f}")
            print(f"  Diversity得分: {score_row['diversity_score']:.3f}")
            print(f"  Title长度: {len(str(row['title']))}")
            print(f"  Text长度: {len(str(row['text']))}")
            print(f"  Title: {str(row['title'])[:50]}...")
            print(f"  Text: {str(row['text'])[:100]}...")
            print()
        
        logger.info("数据筛选完成!")
        
    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 