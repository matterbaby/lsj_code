import torch
import torch.nn as nn
from .gnn import GNNModelWithResidual
from .attention import BiDirectionalCrossAttention

class ClassificationModel(nn.Module):
    """最终的分类模型"""
    
    def __init__(self, hidden_dim, output_dim):
        """
        Args:
            hidden_dim (int): 隐藏层维度
            output_dim (int): 输出维度
        """
        super(ClassificationModel, self).__init__()
        self.protein_gnn = GNNModelWithResidual(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.ligand_gnn = GNNModelWithResidual(input_dim=5, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.cross_attention = BiDirectionalCrossAttention(hidden_dim, hidden_dim)
        
        # 只在分类器中添加dropout和batchnorm
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # 使用较小的dropout率
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, protein_data, ligand_data):
        """前向传播
        
        Args:
            protein_data: 蛋白质图数据
            ligand_data: 配体图数据
            
        Returns:
            torch.Tensor: 分类预测结果
        """
        # 获取蛋白质和配体的特征
        protein_features = self.protein_gnn(protein_data)
        ligand_features = self.ligand_gnn(ligand_data)
        
        # 应用交叉注意力
        protein_attended, ligand_attended = self.cross_attention(protein_features, ligand_features)
        
        # 连接特征并进行分类
        combined_features = torch.cat([protein_attended, ligand_attended], dim=1)
        output = self.classifier(combined_features)
        
        return output
