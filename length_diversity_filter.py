#!/usr/bin/env python3

import argparse
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from textblob import TextBlob


class LengthDiversityFilter:
    def __init__(self, data: pd.DataFrame, metadata: Dict):
        """
        基于文本长度和语义多样性的数据筛选器
        
        Args:
            data: 要筛选的数据集
            metadata: 数据集的元数据信息
        """
        self.data = data.copy()
        self.metadata = metadata
        self.logger = logging.getLogger(__name__)

    def calculate_text_length_scores(self, text_columns: List[str]) -> pd.Series:
        """计算文本长度得分（Fidelity维度）"""
        scores = pd.Series(0.0, index=self.data.index)
        
        if not text_columns:
            self.logger.warning("没有指定文本列，文本长度得分将为0")
            return scores
        
        for col in text_columns:
            if col in self.data.columns:
                # 计算文本长度
                text_lengths = self.data[col].astype(str).str.len()
                self.logger.debug(f"列 {col} 的文本长度范围: {text_lengths.min()} - {text_lengths.max()}")
                
                # 标准化到0-1范围
                if text_lengths.max() > text_lengths.min():
                    normalized_lengths = (text_lengths - text_lengths.min()) / (text_lengths.max() - text_lengths.min())
                else:
                    normalized_lengths = pd.Series(0.5, index=text_lengths.index)
                scores += normalized_lengths
            else:
                self.logger.warning(f"文本列 {col} 不存在于数据中")
        
        return scores / len(text_columns) if text_columns else scores

    def calculate_semantic_diversity_scores(self, text_columns: List[str]) -> pd.Series:
        """计算语义多样性得分（Diversity维度）"""
        scores = pd.Series(0.0, index=self.data.index)
        
        if not text_columns:
            self.logger.warning("没有指定文本列，语义多样性得分将为0")
            return scores
        
        for col in text_columns:
            if col in self.data.columns:
                texts = self.data[col].astype(str)
                
                # 语义多样性计算
                # 1. 词汇丰富度 (Type-Token Ratio)
                unique_words = texts.str.split().apply(lambda x: len(set(x)) if x else 0)
                total_words = texts.str.split().apply(lambda x: len(x) if x else 1)
                vocabulary_richness = unique_words / total_words
                
                # 2. 句子复杂度
                sentences = texts.str.split('[.!?]')
                sentence_count = sentences.apply(lambda x: len([s for s in x if s.strip()]) if x else 1)
                avg_sentence_length = texts.str.len() / sentence_count
                
                # 3. 词汇复杂度 (长词比例)
                words = texts.str.split()
                long_words = words.apply(lambda x: sum(1 for w in x if len(w) > 6) if x else 0)
                total_words_for_complexity = words.apply(lambda x: len(x) if x else 1)
                complexity_ratio = long_words / total_words_for_complexity
                
                # 4. 标点符号多样性
                punctuation_variety = texts.str.count(r'[.!?,;:()]')
                
                # 5. 综合语义多样性得分
                if avg_sentence_length.max() > 0:
                    sentence_scores = avg_sentence_length / avg_sentence_length.max()
                else:
                    sentence_scores = pd.Series(0.5, index=avg_sentence_length.index)
                
                if punctuation_variety.max() > 0:
                    punctuation_scores = punctuation_variety / punctuation_variety.max()
                else:
                    punctuation_scores = pd.Series(0.5, index=punctuation_variety.index)
                
                semantic_scores = (
                    vocabulary_richness * 0.3 +  # 词汇丰富度权重30%
                    sentence_scores * 0.3 +      # 句子复杂度权重30%
                    complexity_ratio * 0.2 +     # 词汇复杂度权重20%
                    punctuation_scores * 0.2     # 标点多样性权重20%
                )
                
                scores += semantic_scores.fillna(0.5)
                
                self.logger.debug(f"列 {col} 的语义多样性范围: {semantic_scores.min():.3f} - {semantic_scores.max():.3f}")
            else:
                self.logger.warning(f"文本列 {col} 不存在于数据中")
        
        return scores / len(text_columns) if text_columns else scores

    def get_column_types(self) -> Dict[str, List[str]]:
        """获取按类型分组的列名"""
        column_types = {
            'text': [],
            'numerical': [],
            'categorical': [],
            'pii': []
        }
        
        for col, info in self.metadata.get('columns', {}).items():
            if col in self.data.columns:
                col_type = info.get('sdtype', 'unknown')
                if col_type in column_types:
                    column_types[col_type].append(col)
                
                # 检查是否为PII
                if info.get('pii', False):
                    column_types['pii'].append(col)
        
        self.logger.info(f"检测到的列类型: {column_types}")
        return column_types

    def get_metadata_config(self) -> Dict:
        """从metadata中获取配置信息"""
        config = {}
        
        # 获取文本列配置
        if 'text_columns' in self.metadata:
            config['text_columns'] = self.metadata['text_columns']
            self.logger.info(f"从metadata获取文本列: {config['text_columns']}")
        
        return config

    def calculate_scores(self, config: Dict) -> pd.DataFrame:
        """
        计算基于文本长度和语义多样性的得分
        
        Args:
            config: 配置字典，包含各维度的参数
        
        Returns:
            包含所有得分的DataFrame
        """
        scores_df = pd.DataFrame(index=self.data.index)
        column_types = self.get_column_types()
        metadata_config = self.get_metadata_config()
        
        # 获取文本列配置，优先使用metadata中的配置，然后是指定的配置，最后是自动检测
        text_cols = config.get('text_columns') or metadata_config.get('text_columns') or column_types['text']
        
        self.logger.info(f"使用的文本列: {text_cols}")
        
        # 1. Fidelity维度 (50%权重) - 文本长度
        text_length_scores = self.calculate_text_length_scores(text_cols)
        self.logger.info(f"文本长度得分范围: {text_length_scores.min():.3f} - {text_length_scores.max():.3f}")
        scores_df['fidelity_score'] = text_length_scores
        
        # 2. Diversity维度 (50%权重) - 语义多样性
        semantic_scores = self.calculate_semantic_diversity_scores(text_cols)
        self.logger.info(f"语义多样性得分范围: {semantic_scores.min():.3f} - {semantic_scores.max():.3f}")
        scores_df['diversity_score'] = semantic_scores
        
        return scores_df

    def filter_top_data(self, config: Dict, top_n: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        筛选出得分最高的前N条数据
        
        Args:
            config: 配置字典
            top_n: 要筛选的数据条数
        
        Returns:
            (筛选后的数据, 得分DataFrame)
        """
        # 计算所有指标得分
        scores_df = self.calculate_scores(config)
        
        # 计算加权总分 (Fidelity 50%, Diversity 50%)
        total_score = (
            0.50 * scores_df['fidelity_score'] +
            0.50 * scores_df['diversity_score']
        )
        
        self.logger.info(f"加权总分范围: {total_score.min():.3f} - {total_score.max():.3f}")
        
        scores_df['total_score'] = total_score
        
        # 按总分排序并选择前N条
        top_indices = total_score.nlargest(top_n).index
        filtered_data = self.data.loc[top_indices].copy()
        filtered_scores = scores_df.loc[top_indices].copy()
        
        return filtered_data, filtered_scores


def parse_args():
    parser = argparse.ArgumentParser(description='基于文本长度和语义多样性的数据筛选工具')

    parser.add_argument('--data', required=True, help='要筛选的数据集CSV文件路径')
    parser.add_argument('--metadata', required=True, help='元数据JSON文件路径')
    parser.add_argument('--output', required=True, help='筛选后数据的输出路径')
    parser.add_argument('--scores-output', help='得分数据的输出路径（可选）')
    parser.add_argument('--top-n', type=int, default=50, help='要筛选的数据条数（默认: 50）')

    # 列配置
    parser.add_argument('--text-columns', nargs='+', help='文本列名（如title, text）')

    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='设置日志级别')
    
    return parser

def main():
    parser = parse_args()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # 加载数据
    logger.info(f"正在加载数据集: {args.data}")
    data = pd.read_csv(args.data)

    logger.info(f"正在加载元数据: {args.metadata}")
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)

    # 删除未在元数据中定义的 _id
    metadata_columns = metadata.get("columns", {})
    if '_id' in data.columns and '_id' not in metadata_columns:
        logger.warning("在数据中发现 '_id' 列但元数据中未定义 — 正在删除")
        data.drop(columns=['_id'], inplace=True)

    # 构建配置
    config = {
        'text_columns': args.text_columns,
    }

    # 创建数据筛选器
    logger.info("正在创建长度-多样性数据筛选器...")
    data_filter = LengthDiversityFilter(data, metadata)

    # 筛选数据
    logger.info(f"正在筛选前{args.top_n}条数据...")
    filtered_data, filtered_scores = data_filter.filter_top_data(config, top_n=args.top_n)

    # 保存结果
    logger.info(f"正在保存筛选后的数据到: {args.output}")
    filtered_data.to_csv(args.output, index=False)
    
    if args.scores_output:
        logger.info(f"正在保存得分数据到: {args.scores_output}")
        filtered_scores.to_csv(args.scores_output, index=False)

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

    logger.info("数据筛选完成!")

if __name__ == "__main__":
    main() 