import torch
import torch.nn as nn
import torch.nn.functional as F

class BiDirectionalCrossAttention(nn.Module):
    """双向交叉注意力机制"""
    
    def __init__(self, input_dim, output_dim, num_heads=4):
        """
        Args:
            input_dim (int): 输入特征维度
            output_dim (int): 输出特征维度
            num_heads (int, optional): 注意力头数量. Defaults to 4.
        """
        super(BiDirectionalCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        assert output_dim % num_heads == 0

        self.query_fc1 = nn.Linear(input_dim, output_dim)
        self.key_fc1 = nn.Linear(input_dim, output_dim)
        self.value_fc1 = nn.Linear(input_dim, output_dim)

        self.query_fc2 = nn.Linear(input_dim, output_dim)
        self.key_fc2 = nn.Linear(input_dim, output_dim)
        self.value_fc2 = nn.Linear(input_dim, output_dim)

        self.output_fc1 = nn.Linear(output_dim, output_dim)
        self.output_fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, protein_features, ligand_features):
        """前向传播
        
        Args:
            protein_features (torch.Tensor): 蛋白质特征
            ligand_features (torch.Tensor): 配体特征
            
        Returns:
            tuple: (protein_output, ligand_output)
        """
        batch_size = protein_features.size(0)
        
        # 蛋白质到配体的注意力
        q1 = self.query_fc1(protein_features).view(batch_size, -1, self.num_heads, self.head_dim)
        k1 = self.key_fc1(ligand_features).view(batch_size, -1, self.num_heads, self.head_dim)
        v1 = self.value_fc1(ligand_features).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # 配体到蛋白质的注意力
        q2 = self.query_fc2(ligand_features).view(batch_size, -1, self.num_heads, self.head_dim)
        k2 = self.key_fc2(protein_features).view(batch_size, -1, self.num_heads, self.head_dim)
        v2 = self.value_fc2(protein_features).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # 计算注意力分数
        scores1 = torch.matmul(q1, k1.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores2 = torch.matmul(q2, k2.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 应用softmax
        attn1 = F.softmax(scores1, dim=-1)
        attn2 = F.softmax(scores2, dim=-1)
        
        # 计算输出
        out1 = torch.matmul(attn1, v1)
        out2 = torch.matmul(attn2, v2)
        
        # 重塑并通过输出层
        protein_output = self.output_fc1(out1.reshape(batch_size, -1))
        ligand_output = self.output_fc2(out2.reshape(batch_size, -1))
        
        return protein_output, ligand_output
