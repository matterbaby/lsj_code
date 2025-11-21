import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score
from src.data import create_dataloader
from scipy import stats
from src.models import ClassificationModel

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        
        if alpha is not None:
            self.alpha = torch.tensor(alpha)
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train(model, train_loader, optimizer, criterion, device):
    """训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
    
    Returns:
        tuple: (平均损失, 准确率)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for protein_data, ligand_data, labels in train_loader:
        protein_data = protein_data.to(device)
        ligand_data = ligand_data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(protein_data, ligand_data)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 添加梯度裁剪，最大范数设为1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    
    return avg_loss, accuracy

def evaluate(model, val_loader, criterion, device):
    """评估模型
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
    
    Returns:
        tuple: (准确率, 平均损失, 所有预测结果, 所有真实标签)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for protein_data, ligand_data, labels in val_loader:
            protein_data = protein_data.to(device)
            ligand_data = ligand_data.to(device)
            labels = labels.to(device)
            
            outputs = model(protein_data, ligand_data)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_predictions.append(predicted)
            all_labels.append(labels)
    
    # 将列表转换为张量
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(val_loader)
    
    return accuracy, avg_loss, all_predictions, all_labels

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载并平衡数据
    from src.data.negative_sampling import generate_balanced_dataset
    print("\n准备数据集...")
    df = pd.read_csv('combined_data.csv')
    balanced_df = generate_balanced_dataset(df)
    
    # 划分数据集 (8:1:1)
    train_val_df, test_df = train_test_split(balanced_df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.111, random_state=42)
    
    print(f"\n数据集大小:")
    print(f"训练集: {len(train_df)}")
    print(f"验证集: {len(val_df)}")
    print(f"测试集: {len(test_df)}")
    
    # 计算类别权重
    train_labels = train_df['flag'].values
    num_samples = len(train_labels)
    num_classes = 2
    class_counts = np.bincount(train_labels)
    class_weights = num_samples / (num_classes * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print("\n类别权重:")
    for i, weight in enumerate(class_weights):
        print(f"Class {i}: {weight:.4f}")
    
    # 创建数据加载器
    train_loader = create_dataloader(train_df, batch_size=32)
    val_loader = create_dataloader(val_df, batch_size=32)
    test_loader = create_dataloader(test_df, batch_size=32)
    
    # 初始化模型
    model = ClassificationModel(hidden_dim=64, output_dim=2).to(device)
    
    # 使用Focal Loss，添加类别权重
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    # 创建预热调度器和主调度器
    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    
    # 预热调度器：从0.1倍学习率开始，在10个epoch内线性增加到原始学习率
    warmup_epochs = 10
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1,  # 初始学习率为原始学习率的0.1倍
        end_factor=1.0,    # 最终达到原始学习率
        total_iters=warmup_epochs
    )
    
    # 主调度器：使用ReduceLROnPlateau
    main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=5
    )
    
    # 组合两个调度器
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )
    
    # 训练设置
    num_epochs = 100
    best_accuracy = 0
    
    # 用于记录训练过程的列表
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    print("\n" + "="*70)
    print(f"{'Epoch':^10}{'Train Loss':^15}{'Train Acc':^15}{'Val Loss':^15}{'Val Acc':^15}")
    print("="*70)
    
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        
        # 验证
        val_acc, val_loss, predictions, labels = evaluate(model, val_loader, criterion, device)
        
        # 记录数据
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 更新学习率
        if epoch < warmup_epochs:
            # 在预热阶段，直接步进
            warmup_scheduler.step()
        else:
            # 预热后，使用验证集准确率来调整学习率
            main_scheduler.step(val_acc)
        
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印训练信息
        print(f"{epoch+1:^10d}{train_loss:^15.4f}{train_acc/100:^15.4f}{val_loss:^15.4f}{val_acc/100:^15.4f}")

        # 如果验证准确率提高，保存模型
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"{'★ Model saved! Best accuracy: {:.4f}'.format(best_accuracy/100):^70}")
        
        if (epoch + 1) % 10 == 0:
            print("-"*70)
    
    print("="*70)
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    test_acc, test_loss, predictions, labels = evaluate(model, test_loader, criterion, device)
    print(f"\n测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_acc/100:.4f}")
    
    # 打印评估报告
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    print("\n分类评估报告:")
    print(classification_report(labels, predictions))

    # 计算MCC
    mcc = matthews_corrcoef(labels, predictions)
    print(f"\nMatthews Correlation Coefficient (MCC): {mcc:.4f}")
    # 获取预测概率
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for protein_data, ligand_data, labels in test_loader:
            protein_data = protein_data.to(device)
            ligand_data = ligand_data.to(device)
            outputs = model(protein_data, ligand_data)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 计算AUROC
    auroc = roc_auc_score(all_labels, all_probs)
    print(f"Area Under the ROC Curve (AUROC): {auroc:.4f}")
    
    # 计算AUPRC
    auprc = average_precision_score(all_labels, all_probs)
    print(f"Area Under the Precision-Recall Curve (AUPRC): {auprc:.4f}")
    # 绘制并保存训练过程图
    import matplotlib.pyplot as plt
    
    # 1. 损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss During Training')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.close()
    
    # 2. 准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot([acc/100 for acc in train_accs], label='Training Accuracy')
    plt.plot([acc/100 for acc in val_accs], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy During Training')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_curve.png')
    plt.close()
    
    # 3. ROC曲线
    from sklearn.metrics import roc_curve, auc
    # 获取预测概率
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for protein_data, ligand_data, labels in test_loader:
            protein_data = protein_data.to(device)
            ligand_data = ligand_data.to(device)
            outputs = model(protein_data, ligand_data)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 计算ROC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curve.png')
    plt.close()

if __name__ == "__main__":
    main()
