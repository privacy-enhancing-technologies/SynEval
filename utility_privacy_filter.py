#!/usr/bin/env python3

import argparse
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from textblob import TextBlob


class UtilityPrivacyFilter:
    def __init__(self, data: pd.DataFrame, metadata: Dict):
        """
        基于情感评分匹配度和PII隐私保护的数据筛选器
        
        Args:
            data: 要筛选的数据集
            metadata: 数据集的元数据信息
        """
        self.data = data.copy()
        self.metadata = metadata
        self.logger = logging.getLogger(__name__)

    def calculate_sentiment_rating_alignment_scores(self, text_columns: List[str], rating_column: str) -> pd.Series:
        """计算情感与评分的匹配度得分（Utility维度）"""
        scores = pd.Series(0.0, index=self.data.index)
        
        if rating_column not in self.data.columns:
            self.logger.warning(f"评分列 {rating_column} 不存在，跳过情感评分匹配计算")
            return scores
        
        if not text_columns:
            self.logger.warning("没有指定文本列，情感评分匹配度得分将为0")
            return scores
        
        try:
            for col in text_columns:
                if col in self.data.columns:
                    texts = self.data[col].astype(str)
                    ratings = pd.to_numeric(self.data[rating_column], errors='coerce').fillna(3.0)
                    
                    # 计算情感极性
                    sentiment_scores = texts.apply(lambda x: TextBlob(x).sentiment.polarity)
                    
                    # 将情感极性转换为期望的评分范围
                    # 负面情感(-1到0) -> 期望评分1-3
                    # 中性情感(0) -> 期望评分3
                    # 正面情感(0到1) -> 期望评分3-5
                    expected_ratings = 3.0 + sentiment_scores * 2.0
                    
                    # 计算实际评分与期望评分的匹配度
                    # 使用负的绝对差值，差值越小得分越高
                    alignment_scores = 1.0 - abs(ratings - expected_ratings) / 4.0  # 4.0是最大可能差值
                    alignment_scores = alignment_scores.clip(0.0, 1.0)  # 限制在0-1范围
                    
                    self.logger.debug(f"列 {col} 的情感评分匹配度范围: {alignment_scores.min():.3f} - {alignment_scores.max():.3f}")
                    scores += alignment_scores
                else:
                    self.logger.warning(f"文本列 {col} 不存在于数据中")
            
            return scores / len(text_columns) if text_columns else scores
            
        except Exception as e:
            self.logger.error(f"计算情感评分匹配度时出错: {e}")
            return pd.Series(0.5, index=self.data.index)

    def calculate_named_entities_scores(self, text_columns: List[str]) -> pd.Series:
        """计算命名实体得分（Privacy维度，实体越少得分越高）"""
        scores = pd.Series(0.0, index=self.data.index)
        
        if not text_columns:
            self.logger.warning("没有指定文本列，命名实体得分将为0")
            return scores
        
        try:
            from flair.models import SequenceTagger
            from flair.data import Sentence
            
            # 初始化NER模型
            tagger = SequenceTagger.load('ner')
            
            for col in text_columns:
                if col in self.data.columns:
                    texts = self.data[col].astype(str)
                    
                    # 计算每个文本的实体数量
                    entity_counts = []
                    for text in texts:
                        if pd.isna(text) or text.strip() == '':
                            entity_counts.append(0)
                        else:
                            sentence = Sentence(text)
                            tagger.predict(sentence)
                            entity_count = len(sentence.get_spans('ner'))
                            entity_counts.append(entity_count)
                    
                    # 转换为得分（实体越少得分越高）
                    entity_counts = pd.Series(entity_counts)
                    if entity_counts.max() > 0:
                        # 标准化：实体数量越少，得分越高
                        entity_scores = 1.0 - (entity_counts / entity_counts.max())
                    else:
                        entity_scores = pd.Series(1.0, index=entity_counts.index)
                    
                    scores += entity_scores
                    self.logger.debug(f"列 {col} 的命名实体得分范围: {entity_scores.min():.3f} - {entity_scores.max():.3f}")
                else:
                    self.logger.warning(f"文本列 {col} 不存在于数据中")
            
            return scores / len(text_columns) if text_columns else scores
            
        except Exception as e:
            self.logger.warning(f"NER模型不可用，使用简化计算: {e}")
            # 简化版本：基于文本特征估算实体数量
            for col in text_columns:
                if col in self.data.columns:
                    texts = self.data[col].astype(str)
                    
                    # 基于文本特征估算实体数量
                    # 包含数字、大写字母、特殊符号的文本可能包含更多实体
                    has_numbers = texts.str.contains(r'\d')
                    has_uppercase = texts.str.contains(r'[A-Z]')
                    has_special_chars = texts.str.contains(r'[^\w\s]')
                    
                    # 综合特征得分（特征越少，实体可能越少）
                    feature_scores = 1.0 - (has_numbers.astype(int) + has_uppercase.astype(int) + has_special_chars.astype(int)) / 3.0
                    scores += feature_scores
                    
                    self.logger.debug(f"列 {col} 的简化命名实体得分范围: {feature_scores.min():.3f} - {feature_scores.max():.3f}")
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
        
        # 获取utility配置
        if 'utility' in self.metadata:
            utility_config = self.metadata['utility']
            if 'input_columns' in utility_config and 'output_columns' in utility_config:
                config['utility_input'] = utility_config['input_columns']
                config['utility_output'] = utility_config['output_columns']
                self.logger.info(f"从metadata获取utility配置: 输入={config['utility_input']}, 输出={config['utility_output']}")
        
        return config

    def calculate_scores(self, config: Dict) -> pd.DataFrame:
        """
        计算基于情感评分匹配度和PII隐私保护的得分
        
        Args:
            config: 配置字典，包含各维度的参数
        
        Returns:
            包含所有得分的DataFrame
        """
        scores_df = pd.DataFrame(index=self.data.index)
        column_types = self.get_column_types()
        metadata_config = self.get_metadata_config()
        
        # 获取列配置，优先使用metadata中的配置，然后是指定的配置，最后是自动检测
        text_cols = config.get('text_columns') or metadata_config.get('text_columns') or column_types['text']
        rating_col = config.get('rating_column', 'rating')
        
        # 对于utility，使用metadata中的配置
        utility_input_cols = metadata_config.get('utility_input', text_cols)
        utility_output_cols = metadata_config.get('utility_output', [rating_col])
        
        self.logger.info(f"使用的文本列: {text_cols}")
        self.logger.info(f"使用的评分列: {rating_col}")
        self.logger.info(f"Utility输入列: {utility_input_cols}")
        self.logger.info(f"Utility输出列: {utility_output_cols}")
        
        # 1. Utility维度 (50%权重) - 情感与评分匹配度
        if utility_input_cols and utility_output_cols:
            # 使用第一个输入列和第一个输出列
            input_col = utility_input_cols[0] if utility_input_cols else text_cols[0]
            output_col = utility_output_cols[0] if utility_output_cols else rating_col
            sentiment_alignment_scores = self.calculate_sentiment_rating_alignment_scores([input_col], output_col)
        else:
            sentiment_alignment_scores = self.calculate_sentiment_rating_alignment_scores(text_cols, rating_col)
        
        self.logger.info(f"情感评分匹配度得分范围: {sentiment_alignment_scores.min():.3f} - {sentiment_alignment_scores.max():.3f}")
        scores_df['utility_score'] = sentiment_alignment_scores
        
        # 2. Privacy维度 (50%权重) - 命名实体数量（越少越好）
        named_entities_scores = self.calculate_named_entities_scores(text_cols)
        self.logger.info(f"命名实体得分范围: {named_entities_scores.min():.3f} - {named_entities_scores.max():.3f}")
        scores_df['privacy_score'] = named_entities_scores
        
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
        
        # 计算加权总分 (Utility 50%, Privacy 50%)
        total_score = (
            0.50 * scores_df['utility_score'] +
            0.50 * scores_df['privacy_score']
        )
        
        self.logger.info(f"加权总分范围: {total_score.min():.3f} - {total_score.max():.3f}")
        
        scores_df['total_score'] = total_score
        
        # 按总分排序并选择前N条
        top_indices = total_score.nlargest(top_n).index
        filtered_data = self.data.loc[top_indices].copy()
        filtered_scores = scores_df.loc[top_indices].copy()
        
        return filtered_data, filtered_scores


def parse_args():
    parser = argparse.ArgumentParser(description='基于情感评分匹配度和PII隐私保护的数据筛选工具')

    parser.add_argument('--data', required=True, help='要筛选的数据集CSV文件路径')
    parser.add_argument('--metadata', required=True, help='元数据JSON文件路径')
    parser.add_argument('--output', required=True, help='筛选后数据的输出路径')
    parser.add_argument('--scores-output', help='得分数据的输出路径（可选）')
    parser.add_argument('--top-n', type=int, default=50, help='要筛选的数据条数（默认: 50）')

    # 列配置
    parser.add_argument('--text-columns', nargs='+', help='文本列名（如title, text）')
    parser.add_argument('--rating-column', default='rating', help='评分列名（默认: rating）')

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
        'rating_column': args.rating_column,
    }

    # 创建数据筛选器
    logger.info("正在创建Utility-Privacy数据筛选器...")
    data_filter = UtilityPrivacyFilter(data, metadata)

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
    print(f"\n📊 Utility-Privacy数据筛选结果摘要")
    print("=" * 50)
    print(f"原始数据条数: {len(data)}")
    print(f"筛选后数据条数: {len(filtered_data)}")
    
    print(f"\n权重配置:")
    print(f"  Utility维度: 50% (情感评分匹配度)")
    print(f"  Privacy维度: 50% (PII/实体数量)")
    print(f"  Fidelity维度: 0% (不参与评分)")
    print(f"  Diversity维度: 0% (不参与评分)")
    
    print(f"\n筛选后数据的基本统计:")
    print(f"平均总分: {filtered_scores['total_score'].mean():.3f}")
    print(f"最高总分: {filtered_scores['total_score'].max():.3f}")
    print(f"最低总分: {filtered_scores['total_score'].min():.3f}")
    
    print(f"Utility (情感评分匹配度) 平均得分: {filtered_scores['utility_score'].mean():.3f}")
    print(f"Privacy (PII/实体数量) 平均得分: {filtered_scores['privacy_score'].mean():.3f}")

    logger.info("数据筛选完成!")

if __name__ == "__main__":
    main() 