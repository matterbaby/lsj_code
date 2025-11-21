from rdkit import Chem
import torch
from torch_geometric.data import Data
from ..utils.constants import aa_properties
import torch
from torch_geometric.data import Data
import numpy as np
from biopandas.pdb import PandasPdb
import os
from tqdm import tqdm
from scipy.spatial.distance import cdist
import hashlib
def smiles_to_graph(smiles):
    """将SMILES字符串转换为图结构
    
    Args:
        smiles (str): SMILES分子表示
        
    Returns:
        Data: PyTorch Geometric数据对象
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),  # 原子序数
            atom.GetDegree(),  # 原子度
            atom.GetFormalCharge(),  # 原子的形式电荷
            atom.GetIsAromatic(),  # 是否芳香性
            atom.GetHybridization(),  # 杂化轨道类型
        ])

    bonds = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bonds.append([start, end])

    edge_index = torch.tensor(bonds, dtype=torch.long).t().contiguous()
    x = torch.tensor(atom_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index)


def sequence_to_graph(sequence):
    """将蛋白质序列转换为图结构
    
    Args:
        sequence (str): 氨基酸序列
        
    Returns:
        Data: PyTorch Geometric数据对象
    """
    features = []
    for aa in sequence:
        if aa in aa_properties:
            features.append(aa_properties[aa])

    if not features:
        raise ValueError(f"No valid amino acids found in sequence")

    bonds = []
    for i in range(len(features) - 1):
        bonds.append([i, i + 1])  # 连接相邻氨基酸
        bonds.append([i + 1, i])  # 双向连接

    edge_index = torch.tensor(bonds, dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index)

def extract_sequence_from_pdb(ppdb):
    """从PDB对象中提取1字母氨基酸序列"""
    atoms = ppdb.df['ATOM']
    ca_atoms = atoms[atoms['atom_name'] == 'CA']
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    sequence = []
    for res_name in ca_atoms['residue_name']:
        if res_name in three_to_one:
            sequence.append(three_to_one[res_name])
        else:
            sequence.append('X')  # 非标准氨基酸用X表示
    return ''.join(sequence)

# 预定义的5个PDB文件路径（假设放在同一目录下的pdbs文件夹中）
PDB_FILES = {
    "P1": "O42275.pdb",  # 对应序列1
    "P2": "P04058.pdb",  # 对应序列2
    "P3": "P21836.pdb",  # 对应序列3
    "P4": "P23795.pdb",  # 对应序列4
    "P5": "P22303.pdb",  # 对应序列5
}
# 修改预加载逻辑
preloaded_pdb_data = {}  # 改为存储完整信息的字典

for pdb_id, pdb_path in tqdm(PDB_FILES.items(), desc="预加载PDB文件"):
    if os.path.exists(pdb_path):
        ppdb = PandasPdb().read_pdb(pdb_path)
        pdb_sequence = extract_sequence_from_pdb(ppdb)
        ca_atoms = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] == 'CA']
        coords = ca_atoms[['x_coord', 'y_coord', 'z_coord']].values

        # 统一使用MD5哈希
        sequence_hash = hashlib.md5(pdb_sequence.encode()).hexdigest()

        preloaded_pdb_data[sequence_hash] = {
            'pdb_path': pdb_path,
            'sequence': pdb_sequence,
            'coords': coords
        }

def sequence_to_graph_1(sequence, distance_threshold=8.0, verbose=True):
    """将蛋白质序列转换为图结构，并尝试匹配预加载的PDB结构

    参数:
        sequence (str): 氨基酸序列 (如 "MSTP...")
        distance_threshold (float): 距离阈值 (Å)，用于确定空间邻近的边
        verbose (bool): 是否打印处理过程信息

    返回:
        Data: PyG 图数据结构 (包含节点特征和边索引)
    """
    if verbose:
        print(f"\n=== 正在处理序列 (长度: {len(sequence)}) ===")
        print(f"序列: {sequence[:10]}...{sequence[-10:]}")  # 显示序列首尾

    # 生成节点特征
    features = []
    for aa in sequence:
        if aa in aa_properties:
            features.append(aa_properties[aa])
    if not features:
        raise ValueError("错误: 序列中未找到有效的氨基酸")
    x = torch.tensor(features, dtype=torch.float)

    # 尝试匹配PDB文件
    seq_hash = hashlib.md5(sequence.encode()).hexdigest()
    matched_data = preloaded_pdb_data.get(seq_hash)

    # 定义边
    bonds = []
    if matched_data:
        if verbose:
            print(f"✅ 匹配到预加载的PDB文件: {os.path.basename(matched_data['pdb_path'])}")
            print(f"   - 序列验证: {'匹配' if sequence == matched_data['sequence'] else '不匹配'}")
        coords = matched_data['coords']
        if verbose:
            print(f"   - 坐标形状: {coords.shape} (CA原子数: {len(coords)})")

        # 计算距离矩阵并构建边
        dist_matrix = cdist(coords, coords)
        n_contacts = 0
        for i in range(len(dist_matrix)):
            for j in range(i + 1, len(dist_matrix)):
                if dist_matrix[i, j] < distance_threshold:
                    bonds.append([i, j])
                    bonds.append([j, i])
                    n_contacts += 1

        if verbose:
            print(f"   - 基于距离阈值 {distance_threshold}Å 构建了 {n_contacts} 个空间邻近边")
    else:
        if verbose:
            print("⚠️ 未匹配到PDB结构，仅使用序列相邻边")

    # 添加序列相邻边 (确保图连通)
    n_sequential = 0
    for i in range(len(sequence) - 1):
        bonds.append([i, i + 1])
        bonds.append([i + 1, i])
        n_sequential += 1

    if verbose:
        print(f"→ 添加了 {n_sequential} 个序列相邻边")
        print(f"→ 总边数: {len(bonds)}")

    if bonds:
        edge_index = torch.tensor(bonds, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        if verbose:
            print("警告: 未构建任何边")

    return Data(x=x, edge_index=edge_index)
#
# def sequence_to_graph_1(sequence):
#     """将蛋白质序列转换为图结构，自动匹配预存的PDB文件以定义3D边
#
#     Args:
#         sequence (str): 氨基酸序列
#         distance_threshold (float): 距离阈值（Å），默认8.0
#
#     Returns:
#         Data: PyTorch Geometric数据对象
#     """
#     distance_threshold = 8.0
#     # 1. 生成节点特征（与之前相同）
#     features = []
#     for aa in sequence:
#         if aa in aa_properties:
#             features.append(aa_properties[aa])
#     if not features:
#         raise ValueError("No valid amino acids found")
#     x = torch.tensor(features, dtype=torch.float)
#
#     # 2. 尝试匹配PDB文件
#     matched_pdb = None
#     for pdb_id, pdb_path in PDB_FILES.items():
#         if not os.path.exists(pdb_path):
#             continue
#         # 从PDB提取序列
#         ppdb = PandasPdb().read_pdb(pdb_path)
#         pdb_sequence = extract_sequence_from_pdb(ppdb)
#         if pdb_sequence == sequence:
#             matched_pdb = pdb_path
#             break
#
#     # 3. 定义边
#     bonds = []
#     if matched_pdb:
#         # 使用PDB的3D结构定义边
#         ppdb = PandasPdb().read_pdb(matched_pdb)
#         ca_atoms = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] == 'CA']
#         coords = ca_atoms[['x_coord', 'y_coord', 'z_coord']].values
#         dist_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
#         for i in range(len(dist_matrix)):
#             for j in range(i + 1, len(dist_matrix)):
#                 if dist_matrix[i, j] < distance_threshold:
#                     bonds.append([i, j])
#                     bonds.append([j, i])
#     # 无论是否匹配PDB，都保留序列相邻边
#     for i in range(len(sequence) - 1):
#         if [i, i + 1] not in bonds:
#             bonds.append([i, i + 1])
#             bonds.append([i + 1, i])
#
#     edge_index = torch.tensor(bonds, dtype=torch.long).t().contiguous()
#     return Data(x=x, edge_index=edge_index)


