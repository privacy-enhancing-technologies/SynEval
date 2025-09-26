#!/usr/bin/env python3
"""
单数据集评估框架使用示例

这个脚本展示了如何使用 run_single_dataset.py 来评估单个数据集的质量。
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
            "这个产品非常好用，质量很棒！",
            "不太满意，价格太贵了",
            "一般般，还可以接受",
            "非常推荐，性价比很高",
            "质量不错，但是包装需要改进",
            "这个产品非常好用，质量很棒！",  # 重复
            "不太满意，价格太贵了",         # 重复
            "一般般，还可以接受",           # 重复
            "非常推荐，性价比很高",         # 重复
            "质量不错，但是包装需要改进"    # 重复
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

def run_basic_evaluation():
    """运行基本评估"""
    print("\n🔍 运行基本评估...")
    
    # 检查文件是否存在
    if not os.path.exists('sample_dataset.csv') or not os.path.exists('sample_metadata.json'):
        print("❌ 示例文件不存在，请先运行 create_sample_files()")
        return
    
    # 运行评估
    import subprocess
    cmd = [
        'python', 'run_single_dataset.py',
        '--data', 'sample_dataset.csv',
        '--metadata', 'sample_metadata.json',
        '--output', 'basic_evaluation_results.json'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 基本评估完成")
            print("📊 结果已保存到: basic_evaluation_results.json")
        else:
            print("❌ 评估失败:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ 运行评估时出错: {e}")

def run_custom_evaluation():
    """运行自定义评估"""
    print("\n🔍 运行自定义评估...")
    
    # 检查文件是否存在
    if not os.path.exists('sample_dataset.csv') or not os.path.exists('sample_metadata.json'):
        print("❌ 示例文件不存在，请先运行 create_sample_files()")
        return
    
    # 运行自定义评估
    import subprocess
    cmd = [
        'python', 'run_single_dataset.py',
        '--data', 'sample_dataset.csv',
        '--metadata', 'sample_metadata.json',
        '--dimensions', 'summary', 'fidelity', 'diversity',
        '--fidelity-metrics', 'diagnostic', 'quality',
        '--diversity-metrics', 'tabular_diversity',
        '--output', 'custom_evaluation_results.json'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 自定义评估完成")
            print("📊 结果已保存到: custom_evaluation_results.json")
        else:
            print("❌ 评估失败:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ 运行评估时出错: {e}")

def run_utility_evaluation():
    """运行实用性评估"""
    print("\n🔍 运行实用性评估...")
    
    # 检查文件是否存在
    if not os.path.exists('sample_dataset.csv') or not os.path.exists('sample_metadata.json'):
        print("❌ 示例文件不存在，请先运行 create_sample_files()")
        return
    
    # 运行实用性评估
    import subprocess
    cmd = [
        'python', 'run_single_dataset.py',
        '--data', 'sample_dataset.csv',
        '--metadata', 'sample_metadata.json',
        '--dimensions', 'utility',
        '--utility-input', 'review', 'category', 'price',
        '--utility-output', 'rating',
        '--output', 'utility_evaluation_results.json'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 实用性评估完成")
            print("📊 结果已保存到: utility_evaluation_results.json")
        else:
            print("❌ 评估失败:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ 运行评估时出错: {e}")

def display_results():
    """显示评估结果"""
    print("\n📊 评估结果概览:")
    
    result_files = [
        'basic_evaluation_results.json',
        'custom_evaluation_results.json', 
        'utility_evaluation_results.json'
    ]
    
    for file_path in result_files:
        if os.path.exists(file_path):
            print(f"\n📄 {file_path}:")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                # 显示基本信息
                if 'summary' in results:
                    info = results['summary'].get('dataset_info', {})
                    print(f"  数据集形状: {info.get('shape', 'N/A')}")
                    print(f"  内存使用: {info.get('memory_usage_mb', 'N/A'):.2f} MB")
                    print(f"  重复行数: {info.get('duplicate_rows', 'N/A')}")
                
                # 显示评估维度
                dimensions = [k for k in results.keys() if k != 'summary']
                print(f"  评估维度: {', '.join(dimensions)}")
                
            except Exception as e:
                print(f"  ❌ 读取结果文件失败: {e}")
        else:
            print(f"\n📄 {file_path}: 文件不存在")

def main():
    """主函数"""
    print("🚀 单数据集评估框架示例")
    print("=" * 50)
    
    # 1. 创建示例数据
    print("\n1️⃣ 创建示例数据...")
    df, metadata = create_sample_data()
    save_sample_files(df, metadata)
    
    # 2. 运行基本评估
    print("\n2️⃣ 运行基本评估...")
    run_basic_evaluation()
    
    # 3. 运行自定义评估
    print("\n3️⃣ 运行自定义评估...")
    run_custom_evaluation()
    
    # 4. 运行实用性评估
    print("\n4️⃣ 运行实用性评估...")
    run_utility_evaluation()
    
    # 5. 显示结果
    print("\n5️⃣ 显示评估结果...")
    display_results()
    
    print("\n✅ 示例运行完成!")
    print("\n💡 提示:")
    print("   - 查看生成的 JSON 文件了解详细结果")
    print("   - 使用 --plot 参数生成可视化图表")
    print("   - 参考 README_single_dataset.md 了解更多用法")

if __name__ == "__main__":
    main() 