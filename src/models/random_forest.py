import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

class RandomForestFeatureExtractor:
    @staticmethod
    def extract_molecule_features(smiles):
        """
        从SMILES提取分子特征
        
        Args:
            smiles (str): SMILES分子表示
        
        Returns:
            list: 分子特征
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        features = [
            Descriptors.ExactMolWt(mol),           # 精确分子量
            Descriptors.MolLogP(mol),              # LogP
            Descriptors.NumHDonors(mol),           # 氢键供体数
            Descriptors.NumHAcceptors(mol),        # 氢键受体数
            Descriptors.NumRotatableBonds(mol),    # 可旋转键数
            Descriptors.TPSA(mol),                 # 拓扑极性表面积
            mol.GetNumAtoms(),                     # 原子数
            mol.GetNumBonds(),                     # 键数
        ]
        return features

    @staticmethod
    def extract_protein_features(sequence):
        """
        从蛋白质序列提取特征
        
        Args:
            sequence (str): 氨基酸序列
        
        Returns:
            list: 蛋白质序列特征
        """
        from ..utils.constants import aa_properties
        
        hydrophobicity = sum(aa_properties.get(aa, [0,0,0])[0] for aa in sequence) / len(sequence)
        charge = sum(aa_properties.get(aa, [0,0,0])[1] for aa in sequence)
        volume = sum(aa_properties.get(aa, [0,0,0])[2] for aa in sequence) / len(sequence)
        
        return [
            len(sequence),               # 序列长度
            hydrophobicity,              # 平均疏水性
            charge,                      # 总电荷
            volume,                      # 平均体积
        ]

class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42):
        """
        初始化随机森林分类器
        
        Args:
            n_estimators (int): 树的数量
            random_state (int): 随机种子
        """
        self.scaler = StandardScaler()
        self.feature_extractor = RandomForestFeatureExtractor()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state,
            n_jobs=-1  # 使用所有CPU核心
        )
        self.label_encoder = LabelEncoder()
    
    def preprocess_data(self, df):
        """
        数据预处理
        
        Args:
            df (pd.DataFrame): 原始数据框
        
        Returns:
            tuple: 处理后的特征和标签
        """
        # 如果是通用特征数据，直接使用
        if 'flag' in df.columns:
            X = df.drop('flag', axis=1)
            y = df['flag']
            return X, y
        
        # 如果是分子和蛋白质特征数据
        try:
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
            
            return X, y
        
        except KeyError:
            raise ValueError("DataFrame must contain either general features with 'flag' column, or 'smile', 'enzyme', and 'flag' columns")
    
    def fit(self, df):
        """
        训练模型
        
        Args:
            df (pd.DataFrame): 训练数据
        """
        X, y = self.preprocess_data(df)
        
        # 特征缩放
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练模型
        self.model.fit(X_scaled, y)
    
    def predict(self, df):
        """
        预测标签
        
        Args:
            df (pd.DataFrame): 测试数据
        
        Returns:
            np.ndarray: 预测标签
        """
        # 提取特征
        X = df.drop('flag', axis=1)
        
        # 特征缩放
        X_scaled = self.scaler.transform(X)
        
        # 预测
        return self.model.predict(X_scaled)
    
    def predict_proba(self, df):
        """
        预测概率
        
        Args:
            df (pd.DataFrame): 测试数据
        
        Returns:
            np.ndarray: 预测概率
        """
        # 提取特征
        X = df.drop('flag', axis=1)
        
        # 特征缩放
        X_scaled = self.scaler.transform(X)
        
        # 预测概率
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, df):
        """
        评估模型性能
        
        Args:
            df (pd.DataFrame): 测试数据
        
        Returns:
            dict: 性能指标
        """
        X, y_true = self.preprocess_data(df)
        X_scaled = self.scaler.transform(X)
        
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    
    def grid_search_cv(self, df, param_grid=None):
        """
        使用网格搜索进行超参数调优
        
        Args:
            df (pd.DataFrame): 训练数据
            param_grid (dict): 超参数网格
        
        Returns:
            dict: 最佳参数和交叉验证得分
        """
        X, y = self.preprocess_data(df)
        X_scaled = self.scaler.fit_transform(X)
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring='roc_auc'
        )
        
        grid_search.fit(X_scaled, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
