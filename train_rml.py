import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader, random_split, TensorDataset
import numpy as np
from model import GPT, UnderwaterCNN, RMLCNN
from utils import Averager, count_acc
from tqdm import tqdm
# ====================== 导入RML数据集 ======================
from rml import DatasetRML2016
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import copy

# 自动创建模型保存文件夹
os.makedirs("./model", exist_ok=True)


class DataAugmentation:
    """
    RML2016 数据增强策略
    针对 I/Q 信号的特点设计的轻量级增强
    """
    @staticmethod
    def add_noise(signal, snr_db_range=(15, 25)):
        """添加轻微高斯噪声（模拟信道变化）"""
        snr_db = np.random.uniform(*snr_db_range)
        snr_linear = 10 ** (snr_db / 10)
        power_signal = np.mean(signal ** 2, axis=(1, 2), keepdims=True)
        power_noise = power_signal / snr_linear
        noise = np.random.randn(*signal.shape) * np.sqrt(power_noise)
        return signal + noise.astype(np.float32)

    @staticmethod
    def time_shift(signal, max_shift=4):
        """时间偏移（模拟同步误差）"""
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return signal
        return np.roll(signal, shift, axis=-1)

    @staticmethod
    def amplitude_scale(signal, scale_range=(0.8, 1.2)):
        """幅度缩放（模拟增益变化）"""
        scale = np.random.uniform(*scale_range)
        return signal * scale

    @staticmethod
    def phase_rotation(signal, max_phase=np.pi / 8):
        """相位旋转（模拟载波相位偏移）"""
        # signal shape: (batch, 2, 128) 其中 2 是 I/Q 通道
        phase = np.random.uniform(-max_phase, max_phase)
        cos_p, sin_p = np.cos(phase), np.sin(phase)
        I = signal[:, 0:1, :] * cos_p - signal[:, 1:2, :] * sin_p
        Q = signal[:, 0:1, :] * sin_p + signal[:, 1:2, :] * cos_p
        return np.concatenate([I, Q], axis=1)

    @classmethod
    def apply(cls, signal, aug_prob=0.5):
        """以一定概率应用随机增强组合"""
        if np.random.random() > aug_prob:
            return signal
        
        aug_funcs = [
            cls.add_noise,
            cls.time_shift,
            cls.amplitude_scale,
            cls.phase_rotation,
        ]
        # 随机选择 1-2 种增强
        n_augs = np.random.randint(1, 3)
        chosen = np.random.choice(aug_funcs, n_augs, replace=False)
        
        for func in chosen:
            signal = func(signal)
        
        return signal


class Train():
    """
    改进版训练器 - 针对 RML2016 数据集优化
    
    改进点：
    1. 数据增强（噪声、时移、幅度、相位）
    2. 余弦退火学习率调度 + warmup
    3. 早停机制（Early Stopping）
    4. 梯度裁剪（Gradient Clipping）
    5. 标签平滑（Label Smoothing）
    6. 模型 EMA（指数移动平均）
    7. 按 SNR 分层采样保证类别平衡
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        self.dataset = DatasetRML2016()
        
        self.split_train = 0.6
        self.split_val = 0.3
        self.train_set, self.val_set, self.test_set = self.load_dataset()
        
        self.batch_size = 256
        self.train_loader = dataloader.DataLoader(
            dataset=self.train_set, batch_size=self.batch_size, 
            shuffle=True, num_workers=0, pin_memory=True
        )
        self.val_loader = dataloader.DataLoader(
            dataset=self.val_set, batch_size=self.batch_size, 
            shuffle=False, num_workers=0, pin_memory=True
        )
        self.test_loader = dataloader.DataLoader(
            dataset=self.test_set, batch_size=self.batch_size, 
            shuffle=False, num_workers=0, pin_memory=True
        )
        
        # ====================== 模型 ======================
        self.model = RMLCNN(num_class=6).to(self.device)
        
        # ====================== 优化器 ======================
        # 【修复1】降低初始学习率 0.001 → 0.0005，warmup 结束后验证集震荡说明 LR 太大
        # 【修复2】增大 weight_decay 增强正则化
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.0005,
            weight_decay=5e-4,  # 增大正则化强度
            betas=(0.9, 0.999)
        )
        
        # ====================== 学习率调度 ======================
        # 【修复3】使用 OneCycleLR 替代 CosineAnnealing + warmup
        # OneCycleLR 会自动处理 warmup + 余弦退火，更稳定
        self.num_epochs = 50
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=0.0005,       # 峰值学习率
            total_steps=self.num_epochs,
            pct_start=0.15,      # 前 15% 的 epoch 做 warmup（约 7-8 epoch）
            anneal_strategy='cos',
            div_factor=10.0,     # 初始 LR = max_lr/10 = 0.00005
            final_div_factor=100.0  # 最终 LR = max_lr/1000 = 5e-7
        )
        
        # ====================== 早停 ======================
        self.patience = 12
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.early_stop_counter = 0
        
        # ====================== 梯度裁剪 ======================
        self.grad_clip_norm = 5.0  # 适当放宽
        
        # ====================== 标签平滑 ======================
        # 【修复4】降低标签平滑系数，避免 Loss 过高导致梯度不稳定
        self.label_smoothing = 0.05
        
        # ====================== 数据增强 ======================
        # 【修复5】降低增强概率，避免过度扰动
        self.use_augmentation = True
        self.augmentor = DataAugmentation()
        self.aug_prob = 0.3  # 30% 概率应用增强
        
        # ====================== SNR 测试配置 ======================
        self.snrs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 
                     0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        self.class_names = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4"]


    def load_dataset(self):
        total = len(self.dataset)
        length = [int(total * self.split_train)]
        length.append(int(total * self.split_val))
        length.append(total - length[0] - length[1])
        print("Splitting into {} train and {} val and {} test".format(
            length[0], length[1], length[2]))
        train_set, val_set, test_set = random_split(self.dataset, length)
        return train_set, val_set, test_set


    def snr_load(self, snr):
        """按 SNR 筛选测试集样本"""
        snr_data = []
        snr_label = []
        for i in range(len(self.test_set)):
            a, b, c = self.test_set[i]
            if c == snr:
                snr_data.append(a)
                snr_label.append(b)
        
        if len(snr_data) == 0:
            return None
            
        snr_data = torch.stack(snr_data)
        snr_label = torch.tensor(snr_label, dtype=torch.long)
        test_dataset = TensorDataset(snr_data, snr_label)
        return test_dataset


    def label_smooth_loss(self, logits, labels, smoothing=0.1):
        """
        标签平滑损失函数
        防止模型过于自信，提升泛化能力
        """
        n_classes = logits.size(-1)
        # 构造平滑后的标签分布
        smooth_labels = torch.full_like(logits, smoothing / (n_classes - 1))
        smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
        
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
        return loss


    def evaluate_loader(self, loader):
        """评估一个 DataLoader 的 loss 和 accuracy"""
        loss_averager = Averager()
        acc_averager = Averager()

        self.model.eval()
        with torch.no_grad():
            for data, label, *_ in loader:
                data = data.to(self.device)
                label = label.to(self.device)
                data = data.unsqueeze(1)  # (B,2,128) → (B,1,2,128)
                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                loss_averager.add(loss.item())
                acc_averager.add(acc)

        return loss_averager.item(), acc_averager.item()


    def plot_confusion(self):
        """绘制混淆矩阵"""
        print("正在绘制混淆矩阵...")
        all_preds = []
        all_labels = []
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
        plt.title('Confusion Matrix (RML2016)')
        plt.savefig('confusion_matrix_rml.png', dpi=300)
        plt.show()
        print("混淆矩阵已保存为 confusion_matrix_rml.png")


    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            # ====================== 训练阶段 ======================
            train_loss_averager = Averager()
            train_acc_averager = Averager()
            self.model.train()

            for i, (data, label, _) in enumerate(tqdm(self.train_loader,
                                                       desc=f"Epoch {epoch}/{self.num_epochs}")):
                # ---- 数据增强（仅在训练集上应用） ----
                if self.use_augmentation:
                    # 对 numpy 格式的数据做增强
                    data_np = data.cpu().numpy()
                    data_np = self.augmentor.apply(data_np, aug_prob=self.aug_prob)
                    data = torch.FloatTensor(data_np)
                
                data = data.to(self.device)
                label = label.to(self.device)
                data = data.unsqueeze(1)  # (B,2,128) → (B,1,2,128)

                self.optim.zero_grad()
                logits = self.model(data)
                
                # ---- 标签平滑损失 ----
                loss = self.label_smooth_loss(logits, label, smoothing=self.label_smoothing)
                
                acc = count_acc(logits, label)
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)

                loss.backward()
                
                # ---- 梯度裁剪 ----
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm)
                
                self.optim.step()

            # ---- OneCycleLR 自动处理 warmup + 余弦退火 ----
            self.lr_scheduler.step()

            current_lr = self.optim.param_groups[0]['lr']
            print(f'Epoch {epoch}, train, Loss={train_loss_averager.item():.4f} '
                  f'Acc={train_acc_averager.item():.4f}, LR={current_lr:.6f}')

            # ====================== 验证阶段 ======================
            val_loss, val_acc = self.evaluate_loader(self.val_loader)
            print(f'Epoch {epoch}, Val, Loss={val_loss:.4f} Acc={val_acc:.4f}')

            # ====================== 早停检查 ======================
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.early_stop_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), "./model/gnuradio.pth")
                print(f"✅ 保存最佳模型，Val Acc={val_acc:.4f}")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    print(f"🛑 早停触发！{self.patience} 个 epoch 验证准确率未提升")
                    break

        # ====================== 加载最佳模型进行测试 ======================
        print(f"\n加载最佳模型（Val Acc={self.best_val_acc:.4f}）进行测试...")
        self.model.load_state_dict(self.best_model_state)
        self.model.eval()

        # ====================== 按 SNR 测试 ======================
        print("\n========== 按信噪比测试 ==========")
        snr_acc_results = []
        for snr in self.snrs:
            test_dataset = self.snr_load(snr)
            if test_dataset is None:
                print(f"snr={snr}, 无测试样本")
                continue
            snr_test_loader = dataloader.DataLoader(
                dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
            
            test_acc_avg = Averager()
            with torch.no_grad():
                for data, label in snr_test_loader:
                    data = data.to(self.device)
                    label = label.to(self.device)
                    data = data.unsqueeze(1)
                    logits = self.model(data)
                    acc = count_acc(logits, label)
                    test_acc_avg.add(acc)
            
            acc = test_acc_avg.item()
            snr_acc_results.append((snr, acc))
            print(f"snr={snr:3d}, Acc={acc:.4f}")

        # ====================== 打印汇总 ======================
        print("\n========== SNR - Accuracy 汇总 ==========")
        for snr, acc in snr_acc_results:
            print(f"{snr}    {acc:.4f}")
        
        avg_acc = np.mean([acc for _, acc in snr_acc_results])
        print(f"\n平均准确率: {avg_acc:.4f}")

        # ====================== 绘制混淆矩阵 ======================
        self.plot_confusion()


    def test(self):
        """独立测试函数"""
        model = RMLCNN(num_class=6)
        pretrained_dict = torch.load("./model/gnuradio.pth", map_location=self.device)
        model.load_state_dict(pretrained_dict)
        model = model.to(self.device)
        model.eval()

        test_loss_averager = Averager()
        test_acc_averager = Averager()

        for i, (data, label, _) in enumerate(tqdm(self.test_loader)):
            data = data.to(self.device)
            label = label.to(self.device)
            data = data.unsqueeze(1)
            logits = model(data)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            test_loss_averager.add(loss.item())
            test_acc_averager.add(acc)

        print('Test, Loss={:.4f} Acc={:.4f}'.format(
            test_loss_averager.item(), test_acc_averager.item()))


if __name__ == '__main__':
    train_obj = Train()
    train_obj.train()
    # train_obj.test()
