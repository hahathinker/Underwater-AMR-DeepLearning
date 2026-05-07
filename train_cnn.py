"""
训练脚本 - CNN + Gauss 数据集
高斯信道下的 OFDM 调制信号分类训练

用法:
    python train_cnn.py          # 训练模型
    python train_cnn.py --test   # 测试模型
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
from model import CNN
from utils import Averager, count_acc, ensure_path
from dataset import GaussDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class TrainCNN:
    """
    CNN 训练器
    支持训练、验证、按 SNR 测试、混淆矩阵绘制
    """
    def __init__(self, batch_size=256, num_epochs=40, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 数据集
        self.dataset = GaussDataset()
        self.split_train = 0.6
        self.split_val = 0.3
        self.train_set, self.val_set, self.test_set = self._load_dataset()
        
        # 数据加载器
        self.batch_size = batch_size
        self.train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False)
        
        # 模型
        self.model = CNN(num_class=6).to(self.device)
        
        # 优化器
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, step_size=25, gamma=0.1)
        
        # 训练配置
        self.num_epochs = num_epochs
        self.model_path = "./model/cnn_underwater.pth"
        ensure_path("./model")
        
        # 训练历史记录
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        
        # SNR 测试配置
        self.snrs = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        self.class_names = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "256QAM"]

    def _load_dataset(self):
        """划分训练/验证/测试集"""
        total = len(self.dataset)
        length = [
            int(total * self.split_train),
            int(total * self.split_val)
        ]
        length.append(total - length[0] - length[1])
        print(f"数据集划分: 训练={length[0]}, 验证={length[1]}, 测试={length[2]}")
        return random_split(self.dataset, length)

    def _snr_load(self, snr):
        """按 SNR 筛选测试集样本"""
        snr_data, snr_label = [], []
        for i in range(len(self.test_set)):
            a, b, c = self.test_set[i]
            if c == snr:
                snr_data.append(a)
                snr_label.append(b)
        
        if len(snr_data) == 0:
            return None
        
        snr_data = torch.stack(snr_data)
        snr_label = torch.tensor(snr_label, dtype=torch.long)
        return TensorDataset(snr_data, snr_label)

    def evaluate_loader(self, loader):
        """评估一个 DataLoader 的 loss 和 accuracy"""
        loss_averager = Averager()
        acc_averager = Averager()

        self.model.eval()
        with torch.no_grad():
            for data, label, *_ in loader:
                data = data.to(self.device)
                label = label.to(self.device)
                data = data.unsqueeze(1)  # (B,2,1024) → (B,1,2,1024)
                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                loss_averager.add(loss.item())
                acc_averager.add(acc)

        return loss_averager.item(), acc_averager.item()

    def plot_confusion(self):
        """绘制混淆矩阵"""
        print("正在绘制混淆矩阵...")
        all_preds, all_labels = [], []
        self.model.eval()

        with torch.no_grad():
            for data, label, _ in self.test_loader:
                data = data.to(self.device)
                label = label.to(self.device)
                data = data.unsqueeze(1)
                logits = self.model(data)
                pred = torch.argmax(logits, dim=1)

                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - CNN (Gauss)')
        plt.savefig('confusion_matrix_cnn.png', dpi=300)
        plt.close()
        print("混淆矩阵已保存为 confusion_matrix_cnn.png")

    def plot_training_history(self):
        """绘制训练历史曲线（Loss 和 Accuracy）"""
        print("正在绘制训练历史曲线...")
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss 曲线
        ax1.plot(epochs, self.train_losses, 'b-', linewidth=2, label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', linewidth=2, label='Val Loss')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss - CNN', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy 曲线
        ax2.plot(epochs, self.train_accs, 'b-', linewidth=2, label='Train Acc')
        ax2.plot(epochs, self.val_accs, 'r-', linewidth=2, label='Val Acc')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy - CNN', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig('training_history_cnn.png', dpi=300)
        plt.close()
        print("训练历史曲线已保存为 training_history_cnn.png")

    def plot_snr_curve(self, snr_acc_results):
        """绘制 SNR 性能曲线"""
        print("正在绘制 SNR 性能曲线...")
        snrs, accs = zip(*snr_acc_results)
        
        plt.figure(figsize=(10, 6))
        plt.plot(snrs, accs, 'b-o', linewidth=2, markersize=6)
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='100%')
        plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90%')
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80%')
        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('SNR vs Accuracy - CNN (Gauss Dataset)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0.5, 1.05)
        plt.tight_layout()
        plt.savefig('snr_curve_cnn.png', dpi=300)
        plt.close()
        print("SNR 性能曲线已保存为 snr_curve_cnn.png")

    def train(self):
        """训练模型"""
        for epoch in range(1, self.num_epochs + 1):
            # 训练阶段
            train_loss_avg = Averager()
            train_acc_avg = Averager()
            self.model.train()

            for data, label, _ in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}"):
                data = data.to(self.device)
                label = label.to(self.device)
                data = data.unsqueeze(1)

                self.optim.zero_grad()
                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                train_loss_avg.add(loss.item())
                train_acc_avg.add(acc)
                loss.backward()
                self.optim.step()

            self.lr_scheduler.step()

            # 记录训练历史
            self.train_losses.append(train_loss_avg.item())
            self.train_accs.append(train_acc_avg.item())

            print(f'Epoch {epoch}, Train, Loss={train_loss_avg.item():.4f} Acc={train_acc_avg.item():.4f}')

            # 验证阶段
            val_loss, val_acc = self.evaluate_loader(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            print(f'Epoch {epoch}, Val, Loss={val_loss:.4f} Acc={val_acc:.4f}')

        # 绘制训练历史曲线
        self.plot_training_history()

        # 保存模型
        torch.save(self.model.state_dict(), self.model_path)
        print(f"模型已保存至 {self.model_path}")

        # 加载最佳模型进行测试
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device))
        self.model.eval()

        # 按 SNR 测试
        print("\n========== 按信噪比测试 ==========")
        snr_acc_results = []
        for snr in self.snrs:
            test_dataset = self._snr_load(snr)
            if test_dataset is None:
                print(f"snr={snr}, 无测试样本")
                continue
            snr_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            _, test_acc = self.evaluate_loader(snr_loader)
            snr_acc_results.append((snr, test_acc))
            print(f"snr={snr:3d}, Acc={test_acc:.4f}")

        # 绘制 SNR 性能曲线
        self.plot_snr_curve(snr_acc_results)

        # 绘制混淆矩阵
        self.plot_confusion()

    def test(self):
        """独立测试函数"""
        model = CNN(num_class=6)
        model.load_state_dict(
            torch.load(self.model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()

        test_loss_avg = Averager()
        test_acc_avg = Averager()

        with torch.no_grad():
            for data, label, _ in tqdm(self.test_loader):
                data = data.to(self.device)
                label = label.to(self.device)
                data = data.unsqueeze(1)
                logits = model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                test_loss_avg.add(loss.item())
                test_acc_avg.add(acc)

        print(f'Test, Loss={test_loss_avg.item():.4f} Acc={test_acc_avg.item():.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CNN Training on Gauss Dataset")
    parser.add_argument('--test', action='store_true', help='测试模式')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--epochs', type=int, default=40, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    args = parser.parse_args()

    trainer = TrainCNN(batch_size=args.batch_size, num_epochs=args.epochs, lr=args.lr)
    if args.test:
        trainer.test()
    else:
        trainer.train()
