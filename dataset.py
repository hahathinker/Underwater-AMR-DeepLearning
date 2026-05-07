"""
Gauss 数据集加载器
高斯信道下的 OFDM 调制信号数据集
包含 6 种调制类型: BPSK, QPSK, 8PSK, 16QAM, 64QAM, 256QAM
每个样本: I/Q 双通道, 形状 (2, 1024)
"""

import torch
import os
import os.path as osp
import numpy as np
from scipy.io import loadmat
import gc
from typing import Tuple


class GaussDataset():
    """
    读取 Gauss 信道下的 OFDM 调制信号数据集 (.mat 文件)
    
    数据格式:
        .mat 文件中的 'dataset' 变量: shape (1, 32000), dtype=object
        每个样本: [[IQ数据(2,1024)], [标签信息]]
        标签信息: [[调制类型(str)], [SNR值(int)]]
    """
    
    def __init__(self, data_dir: str = "./dataset/Gauss",
                 modulations: list = None,
                 num_samples_per_class: int = 2000 * 16):
        """
        Args:
            data_dir: 数据集目录路径
            modulations: 要加载的调制类型列表, 默认全部6种
            num_samples_per_class: 每类加载的样本数
        """
        if modulations is None:
            modulations = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "256QAM"]
        
        self.modulations = modulations
        self.num_samples_per_class = num_samples_per_class
        
        # 调制类型到标签的映射
        self.mod_to_label = {mod: idx for idx, mod in enumerate(self.modulations)}
        
        # 加载数据
        self.X, self.label_mod, self.label_snr = self._load_data(data_dir)
        gc.collect()
    
    def _load_data(self, data_dir: str):
        """加载所有调制类型的 .mat 文件"""
        X, label_mod, label_snr = [], [], []
        
        for mod_name in self.modulations:
            file_path = osp.join(data_dir, f"{mod_name}.mat")
            if not osp.exists(file_path):
                print(f"警告: 文件不存在 {file_path}, 跳过")
                continue
            
            data = loadmat(file_path)
            dataset = data['dataset']  # shape: (1, 32000)
            
            for i in range(self.num_samples_per_class):
                sample = dataset[0, i]  # shape: (2, 1), dtype=object
                # sample[0]: IQ数据, shape (1,) -> 内部是 (2, 1024) float64
                # sample[1]: 标签信息, shape (1,) -> [[调制类型, SNR]]
                iq_data = sample[0][0]   # (2, 1024), float64
                label_info = sample[1]   # (1,) containing [[mod_type, snr]]
                
                # 解析标签
                label_arr = label_info[0]  # [[array(['BPSK']), array([[-10]])]]
                lab_mod = str(label_arr[0][0])  # "['BPSK']"
                lab_snr = int(label_arr[0][1])
                
                # 清理调制类型字符串: "['BPSK']" -> "BPSK"
                lab_mod = lab_mod.strip("[]'\" ")
                
                X.append(iq_data)
                label_mod.append(lab_mod)
                label_snr.append(lab_snr)
        
        X = np.array(X)   # (N, 2, 1024)
        return X, label_mod, label_snr
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        x = self.X[idx]
        mod = self.label_mod[idx]
        snr = self.label_snr[idx]
        label = self.mod_to_label[mod]
        
        x = torch.FloatTensor(x)           # (2, 1024)
        label = torch.tensor(label, dtype=torch.long)
        return x, label, snr
    
    def __len__(self) -> int:
        return self.X.shape[0]


# ========== 测试 ==========
if __name__ == "__main__":
    dataset = GaussDataset()
    print(f"总样本数: {len(dataset)}")
    x, label, snr = dataset[0]
    print(f"样本形状: {x.shape}, 标签: {label}, SNR: {snr}")
    print(f"调制类型映射: {dataset.mod_to_label}")
