import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, 
    classification_report
)

from .random_forest import RandomForestModel
from .cnn_3d import CNN3DModel
from .gcn import GCNModel
from .attention import BiDirectionalCrossAttention

class ExperimentManager:
    def __init__(self, data, labels):
        """
        实验管理器，用于进行消融实验和对比实验
        
        Args:
            data (np.ndarray): 特征数据
            labels (np.ndarray): 标签
        """
        self.data = data
        self.labels = labels
        
        # 数据预处理
        self.scaler = StandardScaler()
        self.data_scaled = self.scaler.fit_transform(data)
        
        # 数据分割
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data_scaled, self.labels, test_size=0.2, random_state=42
        )
    
    def _evaluate_model(self, y_true, y_pred, y_pred_proba, model_name):
        """
        统一的模型评估方法
        
        Args:
            y_true (np.ndarray): 真实标签
            y_pred (np.ndarray): 预测标签
            y_pred_proba (np.ndarray): 预测概率
            model_name (str): 模型名称
        
        Returns:
            dict: 模型性能指标
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred)
        }
        
        # 打印性能指标
        print(f"\n{model_name} Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
        
        return metrics
    
    def run_random_forest(self):
        """
        运行随机森林实验
        
        Returns:
            dict: 随机森林模型性能指标
        """
        # 将 NumPy 数组转换为 DataFrame
        train_df = pd.DataFrame(self.X_train, columns=[f'feature_{i}' for i in range(self.X_train.shape[1])])
        train_df['flag'] = self.y_train
        
        test_df = pd.DataFrame(self.X_test, columns=[f'feature_{i}' for i in range(self.X_test.shape[1])])
        test_df['flag'] = self.y_test
        
        rf_model = RandomForestModel()
        rf_model.fit(train_df)
        
        y_pred = rf_model.predict(test_df)
        y_pred_proba = rf_model.predict_proba(test_df)[:, 1]
        
        return self._evaluate_model(self.y_test, y_pred, y_pred_proba, 'Random Forest')
    
    def run_3d_cnn(self, input_shape=(1, 10, 10, 10)):
        """
        运行3D-CNN实验
        
        Args:
            input_shape (tuple): 输入张量形状
        
        Returns:
            dict: 3D-CNN模型性能指标
        """
        def features_to_3d_tensor(X, target_shape):
            """
            将特征矩阵转换为3D张量
            
            Args:
                X (np.ndarray): 输入特征矩阵
                target_shape (tuple): 目标张量形状
            
            Returns:
                np.ndarray: 3D张量
            """
            batch_size = X.shape[0]
            total_tensor_size = np.prod(target_shape)
            
            # 初始化3D张量
            X_3d = np.zeros((batch_size, *target_shape), dtype=np.float32)
            
            for i in range(batch_size):
                # 获取当前样本的特征
                sample_features = X[i]
                
                # 标准化特征到 [-1, 1]
                sample_features = (sample_features - sample_features.mean()) / (sample_features.std() + 1e-7)
                
                # 如果特征数少于张量大小，重复特征
                if len(sample_features) < total_tensor_size:
                    repeated_features = np.tile(sample_features, 
                                                (total_tensor_size // len(sample_features) + 1))
                    repeated_features = repeated_features[:total_tensor_size]
                else:
                    # 如果特征数多于张量大小，截断
                    repeated_features = sample_features[:total_tensor_size]
                
                # 重塑为目标形状
                X_3d[i] = repeated_features.reshape(target_shape)
            
            return X_3d
        
        # 转换训练和测试数据
        X_train_3d = features_to_3d_tensor(self.X_train, input_shape)
        X_test_3d = features_to_3d_tensor(self.X_test, input_shape)
        
        model = CNN3DModel(input_shape, num_classes=2)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # 训练
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            outputs = model(torch.FloatTensor(X_train_3d))
            loss = criterion(outputs, torch.LongTensor(self.y_train))
            loss.backward()
            optimizer.step()
        
        # 评估
        model.eval()
        with torch.no_grad():
            y_pred = model.predict(torch.FloatTensor(X_test_3d))
            y_pred_proba = model(torch.FloatTensor(X_test_3d))[:, 1].numpy()
        
        return self._evaluate_model(self.y_test, y_pred, y_pred_proba, '3D-CNN')
    
    def run_gcn(self):
        """
        运行图卷积网络(GCN)实验
        
        Returns:
            dict: GCN模型性能指标
        """
        from torch_geometric.data import Data
        
        def create_graph_data(X, y):
            edge_index = torch.tensor([[i, j] for i in range(X.shape[0]) for j in range(X.shape[0]) if i != j]).t().contiguous()
            x = torch.FloatTensor(X)
            y = torch.LongTensor(y)
            return Data(x=x, edge_index=edge_index, y=y)
        
        train_data = create_graph_data(self.X_train, self.y_train)
        test_data = create_graph_data(self.X_test, self.y_test)
        
        model = GCNModel(num_features=self.X_train.shape[1], num_classes=2)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # 训练
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            outputs = model(train_data)
            loss = criterion(outputs, train_data.y)
            loss.backward()
            optimizer.step()
        
        # 评估
        model.eval()
        with torch.no_grad():
            y_pred = model.predict(test_data)
            y_pred_proba = model(test_data)[:, 1].numpy()
        
        return self._evaluate_model(self.y_test, y_pred, y_pred_proba, 'GCN')
    
    def run_attention_model(self):
        """
        运行注意力模型实验
        
        Returns:
            dict: 注意力模型性能指标
        """
        model = BiDirectionalCrossAttention(
            input_dim=self.X_train.shape[1], 
            output_dim=64
        )
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # 训练
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            protein_features = torch.FloatTensor(self.X_train[:, :self.X_train.shape[1]//2])
            ligand_features = torch.FloatTensor(self.X_train[:, self.X_train.shape[1]//2:])
            outputs, _ = model(protein_features, ligand_features)
            loss = criterion(outputs, torch.LongTensor(self.y_train))
            loss.backward()
            optimizer.step()
        
        # 评估
        model.eval()
        with torch.no_grad():
            protein_features_test = torch.FloatTensor(self.X_test[:, :self.X_test.shape[1]//2])
            ligand_features_test = torch.FloatTensor(self.X_test[:, self.X_test.shape[1]//2:])
            y_pred, _ = model(protein_features_test, ligand_features_test)
            _, y_pred = torch.max(y_pred, 1)
            y_pred_proba = torch.softmax(y_pred, dim=1)[:, 1].numpy()
        
        return self._evaluate_model(self.y_test, y_pred, y_pred_proba, 'Attention Model')
    
    def run_ablation_experiments(self):
        """
        运行消融实验
        
        Returns:
            dict: 各模型性能指标
        """
        results = {
            'Random Forest': self.run_random_forest(),
            '3D-CNN': self.run_3d_cnn(),
            'GCN': self.run_gcn(),
            'Attention Model': self.run_attention_model()
        }
        
        # 可视化性能对比
        self._plot_performance_comparison(results)
        
        # 保存结果到CSV
        self._save_results_to_csv(results)
        
        return results
    
    def _plot_performance_comparison(self, results):
        """
        绘制模型性能对比图
        
        Args:
            results (dict): 模型性能指标
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.2
        
        for i, (model_name, model_results) in enumerate(results.items()):
            performance = [model_results.get(metric, 0) for metric in metrics]
            plt.bar(x + i*width, performance, width, label=model_name)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width * 1.5, metrics)
        plt.legend()
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png')
        plt.close()
    
    def _save_results_to_csv(self, results):
        """
        将实验结果保存到CSV文件
        
        Args:
            results (dict): 模型性能指标
        """
        # 准备数据
        data = []
        for model_name, model_results in results.items():
            row = {'Model': model_name}
            row.update(model_results)
            data.append(row)
        
        # 转换为DataFrame并保存
        df = pd.DataFrame(data)
        df.to_csv('experiment_results.csv', index=False)
