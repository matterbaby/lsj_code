import pandas as pd
import random
from itertools import product
import numpy as np

def generate_balanced_dataset(df, protein_col='enzyme', ligand_col='smile', label_col='flag', random_state=42):
    """生成平衡的数据集，通过从未标记的配对中随机采样负样本
    
    Args:
        df (pd.DataFrame): 原始数据框
        protein_col (str): 蛋白质列名
        ligand_col (str): 配体列名
        label_col (str): 标签列名
        random_state (int): 随机种子
        
    Returns:
        pd.DataFrame: 平衡后的数据集
    """
    # 设置随机种子
    random.seed(random_state)
    np.random.seed(random_state)
    
    # 获取正样本
    positive_samples = df[df[label_col] == 1].copy()
    n_positive = len(positive_samples)
    print(f"正样本数量: {n_positive}")
    
    # 获取所有唯一的蛋白质和配体
    all_proteins = set(df[protein_col].unique())
    all_ligands = set(df[ligand_col].unique())
    print(f"唯一蛋白质数量: {len(all_proteins)}")
    print(f"唯一配体数量: {len(all_ligands)}")
    
    # 获取所有已知的配对
    known_pairs = set(zip(df[protein_col], df[ligand_col]))
    print(f"已知配对数量: {len(known_pairs)}")
    
    # 生成负样本
    negative_pairs = []
    all_possible_pairs = list(product(all_proteins, all_ligands))
    print(f"所有可能的配对数量: {len(all_possible_pairs)}")
    
    # 随机打乱所有可能的配对
    random.shuffle(all_possible_pairs)
    
    # 收集负样本
    for pair in all_possible_pairs:
        if pair not in known_pairs and len(negative_pairs) < n_positive:
            negative_pairs.append(pair)
    
    print(f"采样的负样本数量: {len(negative_pairs)}")
    
    # 创建负样本数据框
    negative_df = pd.DataFrame(negative_pairs, columns=[protein_col, ligand_col])
    negative_df[label_col] = 0  # 标记为负样本
    
    # 合并正负样本
    balanced_df = pd.concat([positive_samples, negative_df], ignore_index=True)
    
    # 打乱数据
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print("\n最终数据集统计:")
    print(f"总样本数: {len(balanced_df)}")
    print(f"正样本比例: {len(balanced_df[balanced_df[label_col] == 1]) / len(balanced_df):.2%}")
    print(f"负样本比例: {len(balanced_df[balanced_df[label_col] == 0]) / len(balanced_df):.2%}")
    
    return balanced_df

def save_dataset(df, output_path):
    """保存数据集到文件
    
    Args:
        df (pd.DataFrame): 数据框
        output_path (str): 输出文件路径
    """
    df.to_csv(output_path, index=False)
    print(f"\n数据集已保存到: {output_path}")

if __name__ == '__main__':
    # 示例使用
    input_path = 'combined_data.csv'
    output_path = 'balanced_data.csv'
    
    print("读取原始数据...")
    df = pd.read_csv(input_path)
    
    print("\n生成平衡数据集...")
    balanced_df = generate_balanced_dataset(df)
    
    print("\n保存平衡数据集...")
    save_dataset(balanced_df, output_path)
