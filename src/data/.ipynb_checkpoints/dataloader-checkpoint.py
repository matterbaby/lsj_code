import torch
from torch_geometric.loader import DataLoader
from .data_processing import smiles_to_graph, sequence_to_graph, sequence_to_graph_1

def create_dataloader(df, batch_size=32, shuffle=True):
    """创建数据加载器
    
    Args:
        df (pd.DataFrame): 包含smile和enzyme列的数据框
        batch_size (int, optional): 批次大小. Defaults to 32.
        shuffle (bool, optional): 是否打乱数据顺序. Defaults to True.
        
    Returns:
        DataLoader: PyTorch Geometric数据加载器
    """
    smiles_list = df['smile'].tolist()
    sequences = df['enzyme'].tolist()
    labels = torch.tensor(df['flag'].tolist(), dtype=torch.long)

    ligand_graphs = [smiles_to_graph(smiles) for smiles in smiles_list]
    protein_graphs = [sequence_to_graph_1(seq) for seq in sequences]

    data_pairs = [(protein_graph, ligand_graph, label)
                  for protein_graph, ligand_graph, label
                  in zip(protein_graphs, ligand_graphs, labels)]

    return DataLoader(data_pairs, batch_size=batch_size, shuffle=shuffle)
