# 解决OMP报错（必须放最顶部）
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pickle
import gc
from typing import Tuple

class DatasetRML2016():
    def __init__(self):
        # 【关键】修正为你文件的实际路径
        # 相对路径写法（推荐，可移植）
        #self.pkl_path = os.path.join("dataset", "RML2016.10a_dict.pkl")
        
        # 或者用绝对路径写法（二选一即可）
        self.pkl_path = r"C:\Users\Lenovo\OFDM-OTFS-modulation-recognition\dataset\RML2016.10a_dict.pkl\RML2016.10a_dict.pkl"
        
        # 匹配你原来的6种调制方式
        self.modulations = {
            'BPSK': 0,
            'QPSK': 1,
            '8PSK': 2,
            'QAM16': 3,
            'QAM64': 4,
            'PAM4': 5
        }

        self.X, self.label_mod, self.label_snr = self.load_data()
        gc.collect()

    def load_data(self):
        # 加载数据集
        with open(self.pkl_path, 'rb') as f:
            rml_data = pickle.load(f, encoding='latin1')
        
        X, label_mod, label_snr = [], [], []
        
        # 读取数据
        for (mod_type, snr), signals in rml_data.items():
            if mod_type not in self.modulations:
                continue
            for sig in signals:
                X.append(sig)
                label_mod.append(mod_type)
                label_snr.append(snr)
        
        X = np.array(X)
        return X, label_mod, label_snr

    def __getitem__(self, idx: int):
        x = self.X[idx]
        mod = self.label_mod[idx]
        snr = self.label_snr[idx]
        label = self.modulations[mod]
        return x, label, snr

    def __len__(self) -> int:
        return self.X.shape[0]

# 测试运行
if __name__ == "__main__":
    dataset = DatasetRML2016()
    print("🎉 数据集加载成功！")
    print(f"总样本数量：{len(dataset)}")
    print(f"单个样本形状：{dataset[0][0].shape} (I/Q两路 + 128个采样点)")
    print(f"样本标签：{dataset[0][1]}")
    print(f"样本SNR：{dataset[0][2]}")