#!/usr/bin/env python3
"""
数据筛选工具使用示例

这个脚本展示了如何使用 data_filter.py 来筛选高质量数据。
"""

import pandas as pd
import json
import os
from pathlib import Path

def create_sample_data():
    """创建示例数据集和元数据"""
    
    # 创建示例数据
    data = {
        'review': [
            "这个产品非常好用，质量很棒！我非常满意这次购买，强烈推荐给大家。",
            "不太满意，价格太贵了，性价比不高。",
            "一般般，还可以接受，但是有改进空间。",
            "非常推荐，性价比很高，质量也不错，值得购买。",
            "质量不错，但是包装需要改进，整体来说还可以。",
            "这个产品非常好用，质量很棒！",  # 短评论
            "不太满意，价格太贵了",         # 短评论
            "一般般，还可以接受",           # 短评论
            "非常推荐，性价比很高",         # 短评论
            "质量不错，但是包装需要改进"    # 短评论
        ],
        'rating': [5, 2, 3, 5, 4, 5, 2, 3, 5, 4],
        'category': ['电子产品', '服装', '食品', '电子产品', '家居', 
                    '电子产品', '服装', '食品', '电子产品', '家居'],
        'user_id': ['user_001', 'user_002', 'user_003', 'user_004', 'user_005',
                   'user_006', 'user_007', 'user_008', 'user_009', 'user_010'],
        'price': [299.99, 89.50, 15.99, 599.99, 45.00,
                 299.99, 89.50, 15.99, 599.99, 45.00]
    }
    
    df = pd.DataFrame(data)
    
    # 创建元数据
    metadata = {
        "columns": {
            "review": {"sdtype": "text"},
            "rating": {"sdtype": "numerical"},
            "category": {"sdtype": "categorical"},
            "user_id": {"sdtype": "pii"},
            "price": {"sdtype": "numerical"}
        }
    }
    
    return df, metadata

def save_sample_files(df, metadata):
    """保存示例文件"""
    
    # 保存数据
    df.to_csv('sample_dataset.csv', index=False)
    print("✅ 示例数据集已保存: sample_dataset.csv")
    
    # 保存元数据
    with open('sample_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("✅ 示例元数据已保存: sample_metadata.json")

def run_text_length_filter():
    """运行文本长度筛选"""
    print("\n🔍 运行文本长度筛选...")
    
    # 检查文件是否存在
    if not os.path.exists('sample_dataset.csv') or not os.path.exists('sample_metadata.json'):
        print("❌ 示例文件不存在，请先运行 create_sample_files()")
        return
    
    # 运行筛选
    import subprocess
    cmd = [
        'python', 'data_filter.py',
        '--data', 'sample_dataset.csv',
        '--metadata', 'sample_metadata.json',
        '--output', 'filtered_by_length.csv',
        '--scores-output', 'length_scores.csv',
        '--metrics', 'text_length',
        '--text-columns', 'review',
        '--top-n', '5'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 文本长度筛选完成")
            print("📊 结果已保存到: filtered_by_length.csv")
            print("📊 得分已保存到: length_scores.csv")
        else:
            print("❌ 筛选失败:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ 运行筛选时出错: {e}")

def run_multi_metric_filter():
    """运行多指标筛选"""
    print("\n🔍 运行多指标筛选...")
    
    # 检查文件是否存在
    if not os.path.exists('sample_dataset.csv') or not os.path.exists('sample_metadata.json'):
        print("❌ 示例文件不存在，请先运行 create_sample_files()")
        return
    
    # 运行筛选
    import subprocess
    cmd = [
        'python', 'data_filter.py',
        '--data', 'sample_dataset.csv',
        '--metadata', 'sample_metadata.json',
        '--output', 'filtered_multi_metric.csv',
        '--scores-output', 'multi_metric_scores.csv',
        '--metrics', 'text_length', 'text_quality', 'sentiment',
        '--text-columns', 'review',
        '--weights', '0.5', '0.3', '0.2',
        '--top-n', '5'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 多指标筛选完成")
            print("📊 结果已保存到: filtered_multi_metric.csv")
            print("📊 得分已保存到: multi_metric_scores.csv")
        else:
            print("❌ 筛选失败:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ 运行筛选时出错: {e}")

def run_numerical_filter():
    """运行数值筛选"""
    print("\n🔍 运行数值筛选...")
    
    # 检查文件是否存在
    if not os.path.exists('sample_dataset.csv') or not os.path.exists('sample_metadata.json'):
        print("❌ 示例文件不存在，请先运行 create_sample_files()")
        return
    
    # 运行筛选
    import subprocess
    cmd = [
        'python', 'data_filter.py',
        '--data', 'sample_dataset.csv',
        '--metadata', 'sample_metadata.json',
        '--output', 'filtered_by_numerical.csv',
        '--scores-output', 'numerical_scores.csv',
        '--metrics', 'numerical_value',
        '--numerical-columns', 'rating', 'price',
        '--numerical-metric-type', 'value',
        '--top-n', '5'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 数值筛选完成")
            print("📊 结果已保存到: filtered_by_numerical.csv")
            print("📊 得分已保存到: numerical_scores.csv")
        else:
            print("❌ 筛选失败:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ 运行筛选时出错: {e}")

def run_diversity_filter():
    """运行多样性筛选"""
    print("\n🔍 运行多样性筛选...")
    
    # 检查文件是否存在
    if not os.path.exists('sample_dataset.csv') or not os.path.exists('sample_metadata.json'):
        print("❌ 示例文件不存在，请先运行 create_sample_files()")
        return
    
    # 运行筛选
    import subprocess
    cmd = [
        'python', 'data_filter.py',
        '--data', 'sample_dataset.csv',
        '--metadata', 'sample_metadata.json',
        '--output', 'filtered_by_diversity.csv',
        '--scores-output', 'diversity_scores.csv',
        '--metrics', 'categorical_diversity',
        '--categorical-columns', 'category',
        '--top-n', '5'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 多样性筛选完成")
            print("📊 结果已保存到: filtered_by_diversity.csv")
            print("📊 得分已保存到: diversity_scores.csv")
        else:
            print("❌ 筛选失败:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ 运行筛选时出错: {e}")

def display_results():
    """显示筛选结果"""
    print("\n📊 筛选结果概览:")
    
    result_files = [
        ('filtered_by_length.csv', '文本长度筛选'),
        ('filtered_multi_metric.csv', '多指标筛选'),
        ('filtered_by_numerical.csv', '数值筛选'),
        ('filtered_by_diversity.csv', '多样性筛选')
    ]
    
    for file_path, description in result_files:
        if os.path.exists(file_path):
            print(f"\n📄 {description} ({file_path}):")
            try:
                df = pd.read_csv(file_path)
                print(f"  筛选出 {len(df)} 条数据")
                
                # 显示前几条数据的关键信息
                if 'review' in df.columns:
                    print("  前3条评论:")
                    for i, review in enumerate(df['review'].head(3)):
                        print(f"    {i+1}. {review[:50]}...")
                
                if 'rating' in df.columns:
                    avg_rating = df['rating'].mean()
                    print(f"  平均评分: {avg_rating:.2f}")
                
            except Exception as e:
                print(f"  ❌ 读取结果文件失败: {e}")
        else:
            print(f"\n📄 {description} ({file_path}): 文件不存在")

def display_scores():
    """显示得分信息"""
    print("\n📊 得分信息概览:")
    
    score_files = [
        ('length_scores.csv', '文本长度得分'),
        ('multi_metric_scores.csv', '多指标得分'),
        ('numerical_scores.csv', '数值得分'),
        ('diversity_scores.csv', '多样性得分')
    ]
    
    for file_path, description in score_files:
        if os.path.exists(file_path):
            print(f"\n📄 {description} ({file_path}):")
            try:
                df = pd.read_csv(file_path)
                
                # 显示得分统计
                if 'total_score' in df.columns:
                    print(f"  总分统计:")
                    print(f"    平均分: {df['total_score'].mean():.3f}")
                    print(f"    最高分: {df['total_score'].max():.3f}")
                    print(f"    最低分: {df['total_score'].min():.3f}")
                
                # 显示各指标得分
                score_columns = [col for col in df.columns if col.startswith('score_')]
                for col in score_columns:
                    metric_name = col.replace('score_', '')
                    mean_score = df[col].mean()
                    print(f"    {metric_name}: {mean_score:.3f}")
                
            except Exception as e:
                print(f"  ❌ 读取得分文件失败: {e}")
        else:
            print(f"\n📄 {description} ({file_path}): 文件不存在")

def main():
    """主函数"""
    print("🚀 数据筛选工具示例")
    print("=" * 50)
    
    # 1. 创建示例数据
    print("\n1️⃣ 创建示例数据...")
    df, metadata = create_sample_data()
    save_sample_files(df, metadata)
    
    # 2. 运行文本长度筛选
    print("\n2️⃣ 运行文本长度筛选...")
    run_text_length_filter()
    
    # 3. 运行多指标筛选
    print("\n3️⃣ 运行多指标筛选...")
    run_multi_metric_filter()
    
    # 4. 运行数值筛选
    print("\n4️⃣ 运行数值筛选...")
    run_numerical_filter()
    
    # 5. 运行多样性筛选
    print("\n5️⃣ 运行多样性筛选...")
    run_diversity_filter()
    
    # 6. 显示结果
    print("\n6️⃣ 显示筛选结果...")
    display_results()
    
    # 7. 显示得分信息
    print("\n7️⃣ 显示得分信息...")
    display_scores()
    
    print("\n✅ 示例运行完成!")
    print("\n💡 提示:")
    print("   - 查看生成的 CSV 文件了解详细结果")
    print("   - 尝试调整权重参数来优化筛选效果")
    print("   - 参考 README_data_filter.md 了解更多用法")
    print("\n🎯 实际应用场景:")
    print("   - 筛选高质量长评论用于训练")
    print("   - 选择高价值产品数据")
    print("   - 筛选多样化的分类数据")
    print("   - 选择复杂文本用于特定任务")

if __name__ == "__main__":
    main() 