#!/usr/bin/env python3

import pandas as pd
import argparse
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HotelDataFilter:
    def __init__(self, data: pd.DataFrame):
        """
        初始化酒店数据筛选器
        
        Args:
            data: 酒店数据DataFrame
        """
        self.data = data.copy()
        self.logger = logging.getLogger(__name__)
        
    def filter_complete_data(self) -> pd.DataFrame:
        """
        筛选出所有列都不为空的数据
        
        Returns:
            筛选后的DataFrame
        """
        # 记录原始数据量
        original_count = len(self.data)
        self.logger.info(f"原始数据量: {original_count}")
        
        # 移除任何包含空值的行
        complete_data = self.data.dropna()
        
        # 记录筛选后的数据量
        filtered_count = len(complete_data)
        self.logger.info(f"完整数据量（无空值）: {filtered_count}")
        self.logger.info(f"移除了 {original_count - filtered_count} 条包含空值的数据")
        
        return complete_data
    
    def filter_valid_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        筛选出checkin_date早于checkout_date的数据
        
        Args:
            data: 输入数据
            
        Returns:
            筛选后的DataFrame
        """
        # 记录筛选前数据量
        before_count = len(data)
        self.logger.info(f"日期筛选前数据量: {before_count}")
        
        # 转换日期格式
        def parse_date(date_str):
            try:
                # 尝试解析 "27 Dec 2020" 格式
                return datetime.strptime(date_str.strip(), "%d %b %Y")
            except:
                try:
                    # 尝试解析其他常见格式
                    return pd.to_datetime(date_str)
                except:
                    return None
        
        # 转换日期列
        data['checkin_date_parsed'] = data['checkin_date'].apply(parse_date)
        data['checkout_date_parsed'] = data['checkout_date'].apply(parse_date)
        
        # 筛选出日期解析成功且checkin早于checkout的数据
        valid_dates = data[
            (data['checkin_date_parsed'].notna()) & 
            (data['checkout_date_parsed'].notna()) &
            (data['checkin_date_parsed'] < data['checkout_date_parsed'])
        ].copy()
        
        # 移除临时的日期解析列
        valid_dates = valid_dates.drop(['checkin_date_parsed', 'checkout_date_parsed'], axis=1)
        
        # 记录筛选后的数据量
        after_count = len(valid_dates)
        self.logger.info(f"有效日期数据量: {after_count}")
        self.logger.info(f"移除了 {before_count - after_count} 条日期无效的数据")
        
        return valid_dates
    
    def calculate_data_quality_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算数据质量得分
        
        Args:
            data: 输入数据
            
        Returns:
            包含质量得分的DataFrame
        """
        scores = pd.DataFrame(index=data.index)
        
        # 1. 数据完整性得分 (30%)
        completeness_score = 1.0  # 已经筛选过，所以都是完整的
        scores['completeness_score'] = completeness_score * 0.3
        
        # 2. 数据一致性得分 (30%)
        # 检查数值字段的合理性
        rate_score = np.where(
            (data['room_rate'] > 0) & (data['room_rate'] < 10000), 
            1.0, 0.0
        )
        fee_score = np.where(
            (data['amenities_fee'] >= 0) & (data['amenities_fee'] < 1000), 
            1.0, 0.0
        )
        consistency_score = (rate_score + fee_score) / 2
        scores['consistency_score'] = consistency_score * 0.3
        
        # 3. 数据多样性得分 (20%)
        # 房间类型的多样性
        room_type_diversity = data['room_type'].nunique() / len(data)
        rewards_diversity = data['has_rewards'].nunique() / 2  # 布尔值最多2种
        diversity_score = (room_type_diversity + rewards_diversity) / 2
        scores['diversity_score'] = diversity_score * 0.2
        
        # 4. 数据实用性得分 (20%)
        # 地址信息的长度（适中为好）
        address_length = data['billing_address'].str.len()
        address_score = 1 - abs(address_length - address_length.median()) / address_length.max()
        scores['utility_score'] = address_score * 0.2
        
        # 计算总分
        scores['total_score'] = scores.sum(axis=1)
        
        return scores
    
    def filter_top_data(self, data: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        """
        筛选出得分最高的前N条数据
        
        Args:
            data: 输入数据
            top_n: 要筛选的数据条数
            
        Returns:
            筛选后的DataFrame
        """
        # 计算质量得分
        scores = self.calculate_data_quality_scores(data)
        
        # 按总分排序并选择前N条
        top_indices = scores['total_score'].nlargest(top_n).index
        filtered_data = data.loc[top_indices].copy()
        
        self.logger.info(f"筛选出前{top_n}条数据")
        self.logger.info(f"平均总分: {scores['total_score'].mean():.3f}")
        self.logger.info(f"最高总分: {scores['total_score'].max():.3f}")
        self.logger.info(f"最低总分: {scores['total_score'].min():.3f}")
        
        return filtered_data

def parse_args():
    parser = argparse.ArgumentParser(description='酒店数据筛选工具')
    
    parser.add_argument('--input', required=True, help='输入CSV文件路径')
    parser.add_argument('--output', required=True, help='输出CSV文件路径')
    parser.add_argument('--top-n', type=int, default=50, help='要筛选的数据条数（默认: 50）')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       help='日志级别')
    
    return parser

def main():
    parser = parse_args()
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 加载数据
    logger.info(f"正在加载数据: {args.input}")
    data = pd.read_csv(args.input)
    logger.info(f"加载了 {len(data)} 条数据")
    
    # 显示数据基本信息
    logger.info("数据列信息:")
    for col in data.columns:
        null_count = data[col].isnull().sum()
        logger.info(f"  {col}: {len(data)} 条记录, {null_count} 个空值")
    
    # 创建数据筛选器
    filter = HotelDataFilter(data)
    
    # 1. 筛选完整数据（无空值）
    complete_data = filter.filter_complete_data()
    
    # 2. 筛选有效日期数据
    valid_data = filter.filter_valid_dates(complete_data)
    
    # 3. 筛选前N条高质量数据
    final_data = filter.filter_top_data(valid_data, args.top_n)
    
    # 保存结果
    logger.info(f"正在保存筛选后的数据到: {args.output}")
    final_data.to_csv(args.output, index=False)
    
    # 显示结果统计
    logger.info("\n" + "="*50)
    logger.info("筛选结果统计")
    logger.info("="*50)
    logger.info(f"原始数据量: {len(data)}")
    logger.info(f"完整数据量（无空值）: {len(complete_data)}")
    logger.info(f"有效日期数据量: {len(valid_data)}")
    logger.info(f"最终筛选数据量: {len(final_data)}")
    
    # 显示各列的基本统计
    logger.info("\n最终数据各列统计:")
    for col in final_data.columns:
        if final_data[col].dtype in ['int64', 'float64']:
            logger.info(f"  {col}: 平均值={final_data[col].mean():.2f}, 范围={final_data[col].min()}-{final_data[col].max()}")
        else:
            unique_count = final_data[col].nunique()
            logger.info(f"  {col}: {unique_count} 个唯一值")
    
    logger.info(f"\n筛选完成！结果已保存到: {args.output}")

if __name__ == "__main__":
    main() 