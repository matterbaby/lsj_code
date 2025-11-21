import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.utils.data import DataLoader

class GCNFeatureExtractor:
    @staticmethod
    def smiles_to_graph(smiles):
        """
        将SMILES转换为图结构
        
        Args:
            smiles (str): SMILES分子表示
        
        Returns:
            Data: PyTorch Geometric图数据对象
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append([
                atom.GetAtomicNum(),      # 原子序数
                atom.GetDegree(),         # 原子度
                atom.GetFormalCharge(),   # 形式电荷
                atom.GetIsAromatic(),     # 是否芳香
                atom.GetHybridization(),  # 杂化类型
            ])
        
        # 边信息
        edges = []
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            edges.append([start, end])
            edges.append([end, start])  # 无向图
        
        # 创建图数据对象
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)
    
    @staticmethod
    def sequence_to_graph(sequence):
        """
        将蛋白质序列转换为图结构
        
        Args:
            sequence (str): 氨基酸序列
        
        Returns:
            Data: PyTorch Geometric图数据对象
        """
        from ..utils.constants import aa_properties
        
        # 节点特征
        node_features = []
        for aa in sequence:
            if aa in aa_properties:
                hydrophobicity, charge, volume = aa_properties[aa]
                node_features.append([hydrophobicity, charge, volume])
            else:
                node_features.append([0, 0, 0])
        
        # 边信息（简单的线性连接）
        edges = []
        for i in range(len(sequence) - 1):
            edges.append([i, i+1])
            edges.append([i+1, i])  # 无向图
        
        # 创建图数据对象
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)

class GCNModel(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=64):
        """
        图卷积网络(GCN)模型
        
        Args:
            num_features (int): 节点特征维度
            num_classes (int): 分类数量
            hidden_channels (int): 隐藏层通道数
        """
        super(GCNModel, self).__init__()
        
        self.feature_extractor = GCNFeatureExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # 图卷积层
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def preprocess_data(self, df):
        """
        数据预处理
        
        Args:
            df (pd.DataFrame): 原始数据框
        
        Returns:
            tuple: 处理后的图数据和标签
        """
        # 提取分子图
        mol_graphs = df['smile'].apply(self.feature_extractor.smiles_to_graph)
        
        # 提取蛋白质图
        protein_graphs = df['enzyme'].apply(self.feature_extractor.sequence_to_graph)
        
        # 合并图特征
        graph_pairs = list(zip(mol_graphs, protein_graphs))
        
        # 编码标签
        y = self.label_encoder.fit_transform(df['flag'])
        
        return graph_pairs, y
    
    def forward(self, data):
        """
        前向传播
        
        Args:
            data (torch_geometric.data.Data): 图数据对象
        
        Returns:
            torch.Tensor: 输出张量
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 图卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # 全局池化
        x = global_mean_pool(x, batch)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def fit(self, df, epochs=50, batch_size=32):
        """
        训练模型
        
        Args:
            df (pd.DataFrame): 训练数据
            epochs (int): 训练轮数
            batch_size (int): 批次大小
        """
        graph_pairs, y = self.preprocess_data(df)
        
        # 创建数据加载器
        train_loader = DataLoader(
            [Data(x=mol_graph.x, edge_index=mol_graph.edge_index, 
                  batch=torch.zeros(mol_graph.x.size(0), dtype=torch.long),
                  y=torch.tensor([label])) 
             for mol_graph, _ in graph_pairs for label in y],
            batch_size=batch_size,
            shuffle=True
        )
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters())
        
        # 训练循环
        for epoch in range(epochs):
            self.train()
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = self(batch)
                loss = criterion(outputs, batch.y.squeeze())
                loss.backward()
                optimizer.step()
    
    def predict(self, df):
        """
        预测
        
        Args:
            df (pd.DataFrame): 测试数据
        
        Returns:
            np.ndarray: 预测结果
        """
        graph_pairs, _ = self.preprocess_data(df)
        
        test_loader = DataLoader(
            [Data(x=mol_graph.x, edge_index=mol_graph.edge_index, 
                  batch=torch.zeros(mol_graph.x.size(0), dtype=torch.long)) 
             for mol_graph, _ in graph_pairs],
            batch_size=32
        )
        
        self.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                outputs = self(batch)
                _, pred = torch.max(outputs, 1)
                predictions.extend(pred.numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, df):
        """
        预测概率
        
        Args:
            df (pd.DataFrame): 测试数据
        
        Returns:
            np.ndarray: 预测概率
        """
        graph_pairs, _ = self.preprocess_data(df)
        
        test_loader = DataLoader(
            [Data(x=mol_graph.x, edge_index=mol_graph.edge_index, 
                  batch=torch.zeros(mol_graph.x.size(0), dtype=torch.long)) 
             for mol_graph, _ in graph_pairs],
            batch_size=32
        )
        
        self.eval()
        probabilities = []
        with torch.no_grad():
            for batch in test_loader:
                outputs = self(batch)
                proba = F.softmax(outputs, dim=1)
                probabilities.extend(proba.numpy())
        
        return np.array(probabilities)
