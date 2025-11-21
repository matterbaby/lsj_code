import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

class CNN3DFeatureExtractor:
    @staticmethod
    def smiles_to_3d_tensor(smiles, grid_size=(10, 10, 10)):
        """
        将SMILES转换为3D体素网格
        
        Args:
            smiles (str): SMILES分子表示
            grid_size (tuple): 3D网格大小
        
        Returns:
            np.ndarray: 3D体素网格
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(grid_size)
        
        # 生成3D构象
        mol_3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol_3d)
        
        # 计算分子特征
        conf = mol_3d.GetConformer()
        grid = np.zeros(grid_size)
        
        for atom in mol_3d.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            atomic_num = atom.GetAtomicNum()
            
            # 将原子映射到网格
            x = int((pos.x + 10) / 20 * grid_size[0])
            y = int((pos.y + 10) / 20 * grid_size[1])
            z = int((pos.z + 10) / 20 * grid_size[2])
            
            if 0 <= x < grid_size[0] and 0 <= y < grid_size[1] and 0 <= z < grid_size[2]:
                grid[x, y, z] = atomic_num
        
        return grid
    
    @staticmethod
    def sequence_to_3d_tensor(sequence, grid_size=(10, 10, 10)):
        """
        将蛋白质序列转换为3D特征张量
        
        Args:
            sequence (str): 氨基酸序列
            grid_size (tuple): 3D网格大小
        
        Returns:
            np.ndarray: 3D特征张量
        """
        from ..utils.constants import aa_properties
        
        grid = np.zeros(grid_size)
        
        for i, aa in enumerate(sequence):
            if aa in aa_properties:
                # 使用氨基酸的物理化学性质填充网格
                hydrophobicity, charge, volume = aa_properties[aa]
                
                # 将序列映射到3D网格
                x = i % grid_size[0]
                y = (i // grid_size[0]) % grid_size[1]
                z = i // (grid_size[0] * grid_size[1])
                
                if z < grid_size[2]:
                    grid[x, y, z] = hydrophobicity
        
        return grid

    @staticmethod
    def extract_molecule_features(smiles):
        """
        提取分子特征
        
        Args:
            smiles (str): SMILES分子表示
        
        Returns:
            list: 分子特征
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        # 计算分子特征
        features = []
        for atom in mol.GetAtoms():
            features.append(atom.GetAtomicNum())
        
        return features
    
    @staticmethod
    def extract_protein_features(sequence):
        """
        提取蛋白质特征
        
        Args:
            sequence (str): 氨基酸序列
        
        Returns:
            list: 蛋白质特征
        """
        from ..utils.constants import aa_properties
        
        features = []
        for aa in sequence:
            if aa in aa_properties:
                hydrophobicity, charge, volume = aa_properties[aa]
                features.append(hydrophobicity)
        
        return features

class CNN3DModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        """
        3D卷积神经网络模型
        
        Args:
            input_shape (tuple): 输入特征形状 (channels, depth, height, width)
            num_classes (int): 分类数量
        """
        super(CNN3DModel, self).__init__()
        
        self.feature_extractor = CNN3DFeatureExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # 3D卷积层
        self.conv1 = nn.Conv3d(input_shape[0], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        
        # 池化层
        self.pool = nn.MaxPool3d(kernel_size=2)
        
        # 全连接层
        def _get_conv_output(shape):
            x = torch.rand(1, *shape)
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            return x.view(1, -1).size(1)
        
        self.fc_input_dim = _get_conv_output(input_shape)
        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def preprocess_data(self, df):
        """
        数据预处理
        
        Args:
            df (pd.DataFrame, np.ndarray, torch.Tensor): 输入数据
        
        Returns:
            tuple: 处理后的特征和标签
        """
        # 如果输入是 torch.Tensor，直接返回
        if isinstance(df, torch.Tensor):
            return df, None
        
        # 如果输入是 NumPy 数组
        if isinstance(df, np.ndarray):
            # 如果是 3D 张量，直接返回
            if df.ndim == 4:
                return torch.FloatTensor(df), None
            
            # 假设 df 是特征矩阵，最后一列是标签
            features = df[:, :-1]
            labels = df[:, -1]
            
            # 创建临时 DataFrame
            temp_df = pd.DataFrame(features)
            temp_df['flag'] = labels
            df = temp_df
        
        # 如果输入是 DataFrame 但已经是特征矩阵
        if 'flag' not in df.columns:
            return torch.FloatTensor(df.values), None
        
        # 提取分子特征
        mol_features = df['smile'].apply(self.feature_extractor.extract_molecule_features)
        mol_features = pd.DataFrame(mol_features.tolist())
        
        # 提取蛋白质特征
        protein_features = df['enzyme'].apply(self.feature_extractor.extract_protein_features)
        protein_features = pd.DataFrame(protein_features.tolist())
        
        # 合并特征
        X = pd.concat([mol_features, protein_features], axis=1)
        
        # 编码标签
        y = self.label_encoder.fit_transform(df['flag'])
        
        return torch.FloatTensor(X.values), torch.LongTensor(y)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量
        
        Returns:
            torch.Tensor: 输出张量
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        x = x.view(x.size(0), -1)
        
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
        X, y = self.preprocess_data(df)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters())
        
        # 训练循环
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
    
    def predict(self, df):
        """
        预测标签
        
        Args:
            df (torch.Tensor, np.ndarray, pd.DataFrame): 输入数据
        
        Returns:
            np.ndarray: 预测结果
        """
        # 预处理数据
        X_tensor, _ = self.preprocess_data(df)
        
        self.eval()
        with torch.no_grad():
            outputs = self(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.numpy()
    
    def predict_proba(self, df):
        """
        预测概率
        
        Args:
            df (torch.Tensor, np.ndarray, pd.DataFrame): 输入数据
        
        Returns:
            np.ndarray: 预测概率
        """
        # 预处理数据
        X_tensor, _ = self.preprocess_data(df)
        
        self.eval()
        with torch.no_grad():
            outputs = self(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.numpy()
