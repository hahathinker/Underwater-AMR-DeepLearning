"""
训练脚本 - RMLCNN + RML2016.10a 数据集
经典调制识别基准数据集训练

用法:
    python train1.py          # 训练模型
    python train1.py --test   # 测试模型
"""

import os
import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
from model import RMLCNN
from utils import Averager, count_acc, ensure_path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle
import gc


class DatasetRML2016:
    """
    RML2016.10a 数据集加载器
    
    数据格式:
        pkl 文件: dict, key=(mod_type, snr), value=signals (1000, 2, 128)
        11 种调制类型, 20 个 SNR 级别 (-20 ~ 18dB)
    """
    def __init__(self, pkl_path="./dataset/RML2016.10a_dict.pkl/RML2016.10a_dict.pkl"):
        self.modulations = {
            'BPSK': 0, 'QPSK': 1, '8PSK': 2,
            'QAM16': 3, 'QAM64': 4, 'PAM4': 5
        }
        self.X, self.label_mod, self.label_snr = self._load_data(pkl_path)
        gc.collect()

    def _load_data(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            rml_data = pickle.load(f, encoding='latin1')
        
        X, label_mod, label_snr = [], [], []
        for (mod_type, snr), signals in rml_data.items():
            if mod_type not in self.modulations:
                continue
            for sig in signals:
                X.append(sig)
                label_mod.append(mod_type)
                label_snr.append(snr)
        
        X = np.array(X)
        return X, label_mod, label_snr

    def __getitem__(self, idx):
        x = self.X[idx]
        mod = self.label_mod[idx]
        snr = self.label_snr[idx]
        label = self.modulations[mod]
        return x, label, snr

    def __len__(self):
        return self.X.shape[0]


class DataAugmentation:
    """
    RML2016 数据增强策略
    针对 I/Q 信号特点设计的轻量级增强
    """
    @staticmethod
    def add_noise(signal, snr_db_range=(15, 25)):
        """添加轻微高斯噪声（兼容 (2,128) 单样本和 (B,2,128) batch）"""
        snr_db = np.random.uniform(*snr_db_range)
        snr_linear = 10 ** (snr_db / 10)
        # 兼容不同维度的输入
        if signal.ndim == 2:  # (2, 128) 单样本
            power_signal = np.mean(signal ** 2, keepdims=True)
        else:  # (B, 2, 128) batch
            power_signal = np.mean(signal ** 2, axis=(1, 2), keepdims=True)
        power_noise = power_signal / snr_linear
        noise = np.random.randn(*signal.shape) * np.sqrt(power_noise)
        return signal + noise.astype(np.float32)

    @staticmethod
    def time_shift(signal, max_shift=4):
        """时间偏移（模拟同步误差，兼容 (2,128) 和 (B,2,128)）"""
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return signal
        return np.roll(signal, shift, axis=-1)

    @staticmethod
    def amplitude_scale(signal, scale_range=(0.8, 1.2)):
        """幅度缩放（模拟增益变化，兼容 (2,128) 和 (B,2,128)）"""
        scale = np.random.uniform(*scale_range)
        return signal * scale

    @staticmethod
    def phase_rotation(signal, max_phase=np.pi / 8):
        """相位旋转（模拟载波相位偏移，兼容 (2,128) 和 (B,2,128)）"""
        phase = np.random.uniform(-max_phase, max_phase)
        cos_p, sin_p = np.cos(phase), np.sin(phase)
        if signal.ndim == 2:  # (2, 128) 单样本
            I = signal[0:1, :] * cos_p - signal[1:2, :] * sin_p
            Q = signal[0:1, :] * sin_p + signal[1:2, :] * cos_p
            return np.concatenate([I, Q], axis=0)
        else:  # (B, 2, 128) batch
            I = signal[:, 0:1, :] * cos_p - signal[:, 1:2, :] * sin_p
            Q = signal[:, 0:1, :] * sin_p + signal[:, 1:2, :] * cos_p
            return np.concatenate([I, Q], axis=1)

    @classmethod
    def apply(cls, signal, aug_prob=0.5):
        """以一定概率应用随机增强组合"""
        if np.random.random() > aug_prob:
            return signal
        
        aug_funcs = [
            cls.add_noise, cls.time_shift,
            cls.amplitude_scale, cls.phase_rotation,
        ]
        n_augs = np.random.randint(1, 3)
        chosen = np.random.choice(aug_funcs, n_augs, replace=False)
        
        for func in chosen:
            signal = func(signal)
        return signal


class TrainRML:
    """
    RMLCNN 训练器 - 针对 RML2016 数据集优化 (v3)
    
    改进点:
    1. 数据增强（噪声、时移、幅度、相位）
    2. OneCycleLR 学习率调度 (按总 batch 步数)
    3. 早停机制 (Early Stopping)
    4. 梯度裁剪 (Gradient Clipping)
    5. 标签平滑 (Label Smoothing)
    6. 数据归一化 (z-score)
    7. 更长的训练 + 更激进的增强
    """
    def __init__(self, batch_size=256, num_epochs=100, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 数据集
        self.dataset = DatasetRML2016()
        self.split_train = 0.6
        self.split_val = 0.3
        self.train_set, self.val_set, self.test_set = self._load_dataset()
        
        # 数据加载器
        self.batch_size = batch_size
        self.train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size,
            shuffle=True, num_workers=0, pin_memory=True)
        self.val_loader = DataLoader(
            self.val_set, batch_size=self.batch_size,
            shuffle=False, num_workers=0, pin_memory=True)
        self.test_loader = DataLoader(
            self.test_set, batch_size=self.batch_size,
            shuffle=False, num_workers=0, pin_memory=True)
        
        # 模型
        self.model = RMLCNN(num_class=6).to(self.device)
        
        # 优化器
        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=lr,
            weight_decay=5e-4, betas=(0.9, 0.999))
        
        # 学习率调度 - 按总 batch 步数
        self.num_epochs = num_epochs
        self.total_steps = len(self.train_loader) * self.num_epochs
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optim, max_lr=lr,
            total_steps=self.total_steps,
            pct_start=0.1, anneal_strategy='cos',
            div_factor=25.0, final_div_factor=1000.0)
        
        # 早停
        self.patience = 20
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.early_stop_counter = 0
        
        # 梯度裁剪
        self.grad_clip_norm = 1.0
        
        # 标签平滑
        self.label_smoothing = 0.1
        
        # 数据增强
        self.use_augmentation = True
        self.augmentor = DataAugmentation()
        self.aug_prob = 0.5
        
        # 数据归一化参数 (z-score)
        self._compute_normalization_stats()
        
        # 训练历史记录
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        
        # 模型保存
        self.model_path = "./model/rmlcnn.pth"
        ensure_path("./model")
        
        # SNR 测试配置
        self.snrs = list(range(-20, 20, 2))  # -20 ~ 18
        self.class_names = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4"]

    def _compute_normalization_stats(self):
        """计算训练集的均值和标准差用于归一化"""
        all_data = []
        for i in range(len(self.train_set)):
            data, _, _ = self.train_set[i]
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            all_data.append(data)
        all_data = torch.stack(all_data)
        self.data_mean = all_data.mean(dim=(0, 2), keepdim=True)  # (2, 1)
        self.data_std = all_data.std(dim=(0, 2), keepdim=True) + 1e-8
        print(f"数据归一化: mean={self.data_mean.squeeze().tolist()}, std={self.data_std.squeeze().tolist()}")

    def _load_dataset(self):
        total = len(self.dataset)
        length = [int(total * self.split_train)]
        length.append(int(total * self.split_val))
        length.append(total - length[0] - length[1])
        print(f"数据集划分: 训练={length[0]}, 验证={length[1]}, 测试={length[2]}")
        return random_split(self.dataset, length)

    def _snr_load(self, snr):
        """按 SNR 筛选测试集样本"""
        snr_data, snr_label = [], []
        for i in range(len(self.test_set)):
            a, b, c = self.test_set[i]
            if c == snr:
                # 确保是 Tensor
                if isinstance(a, np.ndarray):
                    a = torch.from_numpy(a).float()
                snr_data.append(a)
                snr_label.append(b)
        
        if len(snr_data) == 0:
            return None
        
        snr_data = torch.stack(snr_data)
        snr_label = torch.tensor(snr_label, dtype=torch.long)
        return TensorDataset(snr_data, snr_label)

    def _label_smooth_loss(self, logits, labels, smoothing=None):
        """标签平滑损失函数"""
        if smoothing is None:
            smoothing = self.label_smoothing
        n_classes = logits.size(-1)
        smooth_labels = torch.full_like(logits, smoothing / (n_classes - 1))
        smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
        return loss

    def evaluate_loader(self, loader):
        """评估一个 DataLoader 的 loss 和 accuracy"""
        loss_avg = Averager()
        acc_avg = Averager()

        self.model.eval()
        with torch.no_grad():
            for data, label, *_ in loader:
                # 确保是 Tensor 并归一化
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data).float()
                data = self._normalize(data)
                # _normalize 已将 data 移到 device
                label = label.to(self.device)
                data = data.unsqueeze(1)  # (B,2,128) → (B,1,2,128)
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
                # 归一化 (与训练一致)
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data).float()
                data = self._normalize(data)
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
        plt.title('Confusion Matrix - RMLCNN (RML2016)')
        plt.savefig('confusion_matrix_rml.png', dpi=300)
        plt.close()
        print("混淆矩阵已保存为 confusion_matrix_rml.png")

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
        ax1.set_title('Training and Validation Loss - RMLCNN', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy 曲线
        ax2.plot(epochs, self.train_accs, 'b-', linewidth=2, label='Train Acc')
        ax2.plot(epochs, self.val_accs, 'r-', linewidth=2, label='Val Acc')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy - RMLCNN', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig('training_history_rml.png', dpi=300)
        plt.close()
        print("训练历史曲线已保存为 training_history_rml.png")

    def plot_snr_curve(self, snr_acc_results):
        """绘制 SNR 性能曲线"""
        print("正在绘制 SNR 性能曲线...")
        snrs, accs = zip(*snr_acc_results)
        
        plt.figure(figsize=(10, 6))
        plt.plot(snrs, accs, 'r-s', linewidth=2, markersize=6)
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='100%')
        plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90%')
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80%')
        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('SNR vs Accuracy - RMLCNN (RML2016 Dataset)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig('snr_curve_rml.png', dpi=300)
        plt.close()
        print("SNR 性能曲线已保存为 snr_curve_rml.png")

    def _normalize(self, data):
        """z-score 归一化"""
        data = data.to(self.device)
        mean = self.data_mean.to(self.device)
        std = self.data_std.to(self.device)
        return (data - mean) / std

    def train(self):
        """训练模型"""
        for epoch in range(1, self.num_epochs + 1):
            # 训练阶段
            train_loss_avg = Averager()
            train_acc_avg = Averager()
            self.model.train()

            for data, label, _ in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}"):
                # 确保是 Tensor
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data).float()
                
                # 数据增强 (在 numpy 域进行)
                if self.use_augmentation:
                    data_np = data.cpu().numpy()
                    data_np = self.augmentor.apply(data_np, aug_prob=self.aug_prob)
                    data = torch.FloatTensor(data_np)

                # 归一化
                data = self._normalize(data)
                label = label.to(self.device)
                data = data.unsqueeze(1)

                self.optim.zero_grad()
                logits = self.model(data)
                
                # 标签平滑损失
                loss = self._label_smooth_loss(logits, label)
                acc = count_acc(logits, label)
                train_loss_avg.add(loss.item())
                train_acc_avg.add(acc)

                loss.backward()
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
                self.best_model_state = copy.deepcopy(self.model.state_dict())
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
            
            test_acc_avg = Averager()
            with torch.no_grad():
                for data, label in snr_loader:
                    # 归一化 (与训练一致)
                    data = self._normalize(data)
                    label = label.to(self.device)
                    data = data.unsqueeze(1)
                    logits = self.model(data)
                    acc = count_acc(logits, label)
                    test_acc_avg.add(acc)
            
            acc = test_acc_avg.item()
            snr_acc_results.append((snr, acc))
            print(f"snr={snr:3d}, Acc={acc:.4f}")

        # 汇总
        print("\n========== SNR - Accuracy 汇总 ==========")
        for snr, acc in snr_acc_results:
            print(f"{snr}    {acc:.4f}")
        
        avg_acc = np.mean([acc for _, acc in snr_acc_results])
        print(f"\n平均准确率: {avg_acc:.4f}")

        # 绘制 SNR 性能曲线
        self.plot_snr_curve(snr_acc_results)

        # 绘制混淆矩阵
        self.plot_confusion()

    def test(self):
        """独立测试函数"""
        model = RMLCNN(num_class=6)
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
    parser = argparse.ArgumentParser(description="RMLCNN Training on RML2016")
    parser.add_argument('--test', action='store_true', help='测试模式')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0005, help='学习率')
    args = parser.parse_args()

    trainer = TrainRML(batch_size=args.batch_size, num_epochs=args.epochs, lr=args.lr)
    if args.test:
        trainer.test()
    else:
        trainer.train()
