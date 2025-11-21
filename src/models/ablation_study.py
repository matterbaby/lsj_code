import torch
import torch.nn as nn
from .classification import ClassificationModel
from .gnn import GNNModelWithResidual
from .attention import BiDirectionalCrossAttention

class AblationClassificationModel(nn.Module):
    """用于消融实验的分类模型变体"""
    
    def __init__(self, hidden_dim, output_dim, ablation_config):
        """
        Args:
            hidden_dim (int): 隐藏层维度
            output_dim (int): 输出维度
            ablation_config (dict): 消融实验配置
                - use_gnn (bool): 是否使用GNN（如果False则使用简单MLP）
                - use_attention (bool): 是否使用注意力机制
                - use_residual (bool): 是否使用残差连接
                - gnn_layers (int): GNN层数 (1-5)
                - attention_heads (int): 注意力头数量
        """
        super(AblationClassificationModel, self).__init__()
        self.config = ablation_config
        
        # 根据配置决定是使用GNN还是MLP
        if self.config['use_gnn']:
            self.protein_encoder = self._create_gnn(3, hidden_dim, hidden_dim)
            self.ligand_encoder = self._create_gnn(5, hidden_dim, hidden_dim)
        else:
            self.protein_encoder = self._create_mlp(3, hidden_dim)
            self.ligand_encoder = self._create_mlp(5, hidden_dim)
        
        # 根据配置决定是否使用注意力机制
        if self.config['use_attention']:
            self.cross_attention = BiDirectionalCrossAttention(
                hidden_dim, 
                hidden_dim, 
                num_heads=self.config['attention_heads']
            )
        
        # 分类器保持不变
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def _create_gnn(self, input_dim, hidden_dim, output_dim):
        """创建GNN编码器"""
        return GNNModelWithResidual(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_residual=self.config['use_residual']
        )
    
    def _create_mlp(self, input_dim, output_dim):
        """创建MLP编码器"""
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, protein_data, ligand_data):
        """前向传播"""
        # 获取蛋白质和配体的特征
        protein_features = self.protein_encoder(protein_data)
        ligand_features = self.ligand_encoder(ligand_data)
        
        # 根据配置决定是否使用注意力机制
        if self.config['use_attention']:
            protein_features, ligand_features = self.cross_attention(
                protein_features, ligand_features
            )
            
        # 连接特征并进行分类
        combined_features = torch.cat([protein_features, ligand_features], dim=1)
        output = self.classifier(combined_features)
        
        return output

def run_ablation_experiments(train_loader, val_loader, test_loader, device, num_epochs=100):
    """运行一系列消融实验
    
    Args:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        device: 计算设备
        num_epochs: 训练轮数
    
    Returns:
        dict: 包含所有实验结果的字典
    """
    # 定义要测试的不同配置
    experiments = [
        {
            'name': 'Full Model',
            'config': {
                'use_gnn': True,
                'use_attention': True,
                'use_residual': True,
                'gnn_layers': 5,
                'attention_heads': 4
            }
        },
        {
            'name': 'No Attention',
            'config': {
                'use_gnn': True,
                'use_attention': False,
                'use_residual': True,
                'gnn_layers': 5,
                'attention_heads': 0
            }
        },
        {
            'name': 'No GNN (MLP only)',
            'config': {
                'use_gnn': False,
                'use_attention': True,
                'use_residual': False,
                'gnn_layers': 0,
                'attention_heads': 4
            }
        },
        {
            'name': 'No Residual',
            'config': {
                'use_gnn': True,
                'use_attention': True,
                'use_residual': False,
                'gnn_layers': 5,
                'attention_heads': 4
            }
        },
        {
            'name': 'Reduced GNN Layers (3)',
            'config': {
                'use_gnn': True,
                'use_attention': True,
                'use_residual': True,
                'gnn_layers': 3,
                'attention_heads': 4
            }
        },
        {
            'name': 'Reduced Attention Heads (2)',
            'config': {
                'use_gnn': True,
                'use_attention': True,
                'use_residual': True,
                'gnn_layers': 5,
                'attention_heads': 2
            }
        }
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\n{'='*50}")
        print(f"Running experiment: {exp['name']}")
        print(f"Configuration: {exp['config']}")
        print('='*50)
        
        # 创建模型
        model = AblationClassificationModel(
            hidden_dim=64,
            output_dim=2,
            ablation_config=exp['config']
        ).to(device)
        
        # 训练和评估模型
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
        
        best_val_acc = 0
        exp_results = {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': []
        }
        
        for epoch in range(num_epochs):
            # 训练
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for protein_data, ligand_data, labels in train_loader:
                protein_data = protein_data.to(device)
                ligand_data = ligand_data.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(protein_data, ligand_data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_acc = 100. * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # 验证
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for protein_data, ligand_data, labels in val_loader:
                    protein_data = protein_data.to(device)
                    ligand_data = ligand_data.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(protein_data, ligand_data)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = 100. * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            # 保存结果
            exp_results['train_acc'].append(train_acc)
            exp_results['val_acc'].append(val_acc)
            exp_results['train_loss'].append(avg_train_loss)
            exp_results['val_loss'].append(avg_val_loss)
            
            # 更新最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'ablation_{exp["name"].lower().replace(" ", "_")}_best.pt')
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}]:')
                print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 测试最佳模型
        model.load_state_dict(torch.load(f'ablation_{exp["name"].lower().replace(" ", "_")}_best.pt'))
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for protein_data, ligand_data, labels in test_loader:
                protein_data = protein_data.to(device)
                ligand_data = ligand_data.to(device)
                labels = labels.to(device)
                
                outputs = model(protein_data, ligand_data)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        exp_results['test_acc'] = test_acc
        
        print(f"\nFinal Results for {exp['name']}:")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        results[exp['name']] = exp_results
    
    return results

def print_ablation_summary(results):
    """打印消融实验结果总结
    
    Args:
        results (dict): 消融实验结果
    """
    print("\n" + "="*70)
    print("消融实验结果总结")
    print("="*70)
    
    # 创建结果表格
    headers = ["Model Variant", "Best Val Acc", "Test Acc", "Δ from Full"]
    row_format = "{:<25} {:<15} {:<15} {:<15}"
    
    print(row_format.format(*headers))
    print("-"*70)
    
    # 获取完整模型的测试准确率作为基准
    full_model_acc = results['Full Model']['test_acc']
    
    # 打印每个变体的结果
    for name, exp_results in results.items():
        best_val_acc = max(exp_results['val_acc'])
        test_acc = exp_results['test_acc']
        delta = test_acc - full_model_acc
        
        print(row_format.format(
            name,
            f"{best_val_acc:.2f}%",
            f"{test_acc:.2f}%",
            f"{delta:+.2f}%"
        ))
    
    print("\n注：Δ from Full 表示相对于完整模型的性能变化")
    print("正值表示性能提升，负值表示性能下降")
