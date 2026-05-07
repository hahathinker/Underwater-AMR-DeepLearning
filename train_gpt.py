"""
训练脚本 - GPT + Gauss 数据集
使用轻量级 Transformer (GPT) 进行 OFDM 信号调制识别

用法:
    python train_gpt.py          # 训练模型
    python train_gpt.py --test   # 测试模型
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
from model import GPT
from utils import Averager, count_acc, ensure_path
from dataset import GaussDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class DataAugmentation:
    """
    Gauss 数据集数据增强
    针对 I/Q 信号特点设计的轻量级增强
    """
    @staticmethod
    def add_noise(signal, snr_db_range=(25, 35)):
        """添加轻微高斯噪声"""
        snr_db = np.random.uniform(*snr_db_range)
        snr_linear = 10 ** (snr_db / 10)
        power_signal = np.mean(signal ** 2, keepdims=True)
        power_noise = power_signal / snr_linear
        noise = np.random.randn(*signal.shape) * np.sqrt(power_noise)
        return signal + noise.astype(np.float32)

    @staticmethod
    def time_shift(signal, max_shift=8):
        """时间偏移（模拟同步误差）"""
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return signal
        return np.roll(signal, shift, axis=-1)

    @staticmethod
    def amplitude_scale(signal, scale_range=(0.85, 1.15)):
        """幅度缩放（模拟增益变化）"""
        scale = np.random.uniform(*scale_range)
        return signal * scale

    @classmethod
    def apply(cls, signal, aug_prob=0.3):
        """以一定概率应用随机增强组合"""
        if np.random.random() > aug_prob:
            return signal
        
        aug_funcs = [
            cls.add_noise, cls.time_shift, cls.amplitude_scale,
        ]
        n_augs = np.random.randint(1, 3)
        chosen = np.random.choice(aug_funcs, n_augs, replace=False)
        
        for func in chosen:
            signal = func(signal)
        return signal


class TrainGPT:
    """
    GPT 训练器 - 使用 Transformer 进行信号分类 (优化版)
    
    GPT 模型将 I/Q 信号视为序列数据,
    通过双向自注意力机制捕获信号中的全局依赖关系
    
    优化:
    1. 更大的 batch_size (128) 提升 GPU 利用率
    2. OneCycleLR 学习率调度
    3. AdamW 优化器 + 权重衰减
    4. 梯度裁剪
    5. 数据增强
    6. 早停机制
    """
    def __init__(self, batch_size=128, num_epochs=60, lr=0.001):
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
            self.train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        self.test_loader = DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        
        # 模型 (2层 Transformer, 4头注意力)
        self.model = GPT(num_classes=6, n_embd=128, n_layer=2, n_head=4).to(self.device)
        
        # 优化器
        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        # 学习率调度 - OneCycleLR
        self.num_epochs = num_epochs
        self.total_steps = len(self.train_loader) * self.num_epochs
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optim, max_lr=lr,
            total_steps=self.total_steps,
            pct_start=0.1, anneal_strategy='cos',
            div_factor=25.0, final_div_factor=1000.0)
        
        # 早停
        self.patience = 15
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.early_stop_counter = 0
        
        # 梯度裁剪
        self.grad_clip_norm = 1.0
        
        # 数据增强
        self.use_augmentation = True
        self.augmentor = DataAugmentation()
        self.aug_prob = 0.3
        
        # 训练配置
        self.model_path = "./model/gpt_underwater.pth"
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
        loss_avg = Averager()
        acc_avg = Averager()

        self.model.eval()
        with torch.no_grad():
            for data, label, *_ in loader:
                data = data.to(self.device)
                label = label.to(self.device)
                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                loss_avg.add(loss.item())
                acc_avg.add(acc)

        return loss_avg.item(), acc_avg.item()

    def plot_confusion(self):
        """绘制混淆矩阵"""
        print("正在绘制混淆矩阵...")
        all_preds, all_labels = [], []
        self.model.eval()

        with torch.no_grad():
            for data, label, _ in self.test_loader:
                data = data.to(self.device)
                label = label.to(self.device)
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
        plt.title('Confusion Matrix - GPT (Gauss)')
        plt.savefig('confusion_matrix_gpt.png', dpi=300)
        plt.close()
        print("混淆矩阵已保存为 confusion_matrix_gpt.png")

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
        ax1.set_title('Training and Validation Loss - GPT', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy 曲线
        ax2.plot(epochs, self.train_accs, 'b-', linewidth=2, label='Train Acc')
        ax2.plot(epochs, self.val_accs, 'r-', linewidth=2, label='Val Acc')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy - GPT', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig('training_history_gpt.png', dpi=300)
        plt.close()
        print("训练历史曲线已保存为 training_history_gpt.png")

    def plot_snr_curve(self, snr_acc_results):
        """绘制 SNR 性能曲线"""
        print("正在绘制 SNR 性能曲线...")
        snrs, accs = zip(*snr_acc_results)
        
        plt.figure(figsize=(10, 6))
        plt.plot(snrs, accs, 'g-^', linewidth=2, markersize=6)
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='100%')
        plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90%')
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80%')
        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('SNR vs Accuracy - GPT (Gauss Dataset)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0.5, 1.05)
        plt.tight_layout()
        plt.savefig('snr_curve_gpt.png', dpi=300)
        plt.close()
        print("SNR 性能曲线已保存为 snr_curve_gpt.png")

    def train(self):
        """训练模型"""
        for epoch in range(1, self.num_epochs + 1):
            # 训练阶段
            train_loss_avg = Averager()
            train_acc_avg = Averager()
            self.model.train()

            for data, label, _ in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}"):
                # 数据增强
                if self.use_augmentation:
                    data_np = data.cpu().numpy()
                    data_np = self.augmentor.apply(data_np, aug_prob=self.aug_prob)
                    data = torch.FloatTensor(data_np)

                data = data.to(self.device)
                label = label.to(self.device)

                self.optim.zero_grad()
                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                train_loss_avg.add(loss.item())
                train_acc_avg.add(acc)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optim.step()
                
                # 每个 batch 更新学习率
                self.lr_scheduler.step()

            current_lr = self.optim.param_groups[0]['lr']

            # 记录训练历史
            self.train_losses.append(train_loss_avg.item())
            self.train_accs.append(train_acc_avg.item())

            print(f'Epoch {epoch}, Train, Loss={train_loss_avg.item():.4f} '
                  f'Acc={train_acc_avg.item():.4f}, LR={current_lr:.6f}')

            # 验证阶段
            val_loss, val_acc = self.evaluate_loader(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            print(f'Epoch {epoch}, Val, Loss={val_loss:.4f} Acc={val_acc:.4f}')

            # 早停检查
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), self.model_path)
                print(f"✅ 保存最佳模型, Val Acc={val_acc:.4f}")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    print(f"🛑 早停触发! {self.patience} 个 epoch 验证准确率未提升")
                    break

        # 绘制训练历史曲线
        self.plot_training_history()

        # 加载最佳模型
        print(f"\n加载最佳模型 (Val Acc={self.best_val_acc:.4f}) 进行测试...")
        self.model.load_state_dict(self.best_model_state)
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
        model = GPT(num_classes=6, n_embd=128, n_layer=2, n_head=4)
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
                logits = model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                test_loss_avg.add(loss.item())
                test_acc_avg.add(acc)

        print(f'Test, Loss={test_loss_avg.item():.4f} Acc={test_acc_avg.item():.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GPT Training on Gauss Dataset")
    parser.add_argument('--test', action='store_true', help='测试模式')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--epochs', type=int, default=60, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    args = parser.parse_args()

    trainer = TrainGPT(batch_size=args.batch_size, num_epochs=args.epochs, lr=args.lr)
    if args.test:
        trainer.test()
    else:
        trainer.train()
