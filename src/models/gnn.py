import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool

class GNNModelWithResidual(nn.Module):
    """带残差连接的混合图神经网络模型（浅层GAT + 深层SAGE）"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        """
        Args:
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度
            output_dim (int): 输出维度
            dropout (float, optional): Dropout率. Defaults to 0.1.
        """
        super(GNNModelWithResidual, self).__init__()
        # 浅层使用GAT，使用4个注意力头
        self.conv1 = GATConv(input_dim, hidden_dim // 4, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True)
        
        # 深层使用SAGE
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.conv4 = SAGEConv(hidden_dim, hidden_dim)
        self.conv5 = SAGEConv(hidden_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # 添加批标准化
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.bn5 = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        # 调整残差连接的维度
        self.residual_fc1 = nn.Linear(input_dim, hidden_dim)
        self.residual_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, data):
        """前向传播
        
        Args:
            data: PyTorch Geometric数据对象
            
        Returns:
            torch.Tensor: 模型输出
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch
        
        # 第一层 (GAT)
        res1 = self.residual_fc1(x)
        x = self.conv1(x, edge_index)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = x + res1
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        
        # 第二层 (GAT)
        res2 = self.residual_fc2(x)
        x = self.conv2(x, edge_index)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = x + res2
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        
        # 第三层 (SAGE)
        x = self.conv3(x, edge_index)
        if x.size(0) > 1:
            x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        
        # 第四层 (SAGE)
        x = self.conv4(x, edge_index)
        if x.size(0) > 1:
            x = self.bn4(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dropout(x)
        
        # 第五层 (SAGE)
        x = self.conv5(x, edge_index)
        if x.size(0) > 1:
            x = self.bn5(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        
        # 全局池化
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x
