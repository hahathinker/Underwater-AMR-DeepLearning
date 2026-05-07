# OFDM-OTFS 调制识别项目

**基于深度学习的水声 OFDM/OTFS 信号自动调制识别系统**

本项目利用深度学习技术对 OFDM 和 OTFS 调制信号进行自动调制识别 (Automatic Modulation Recognition, AMR)。支持 CNN、残差网络 (RMLCNN) 和 Transformer (GPT) 三种深度学习架构。

---

## 数据集

### 1. Gauss 数据集（高斯信道 OFDM）
- 6 种调制类型: BPSK, QPSK, 8PSK, 16QAM, 64QAM, 256QAM
- 每种调制类型 32000 个样本, 总计 192,000 条
- 每个样本: I/Q 双通道, 形状 (2, 1024)
- 信噪比: -10dB ~ 20dB（间隔 2dB, 共 16 个 SNR 级别）
- 每个 SNR 级别每类 2000 个样本

### 2. RML2016.10a 数据集
- 经典调制识别基准数据集
- 11 种调制类型, 20 个 SNR 级别 (-20 ~ 18dB)
- 每个样本: I/Q 双通道, 形状 (2, 128)
- 本项目使用其中 6 类: BPSK, QPSK, 8PSK, QAM16, QAM64, PAM4

### 3. OTFS 数据集（待补充）
- OTFS (Orthogonal Time Frequency Space) 调制信号
- 面向高速移动场景

---

## 模型架构

### 1. CNN
多层卷积神经网络, 适用于水声 OFDM/OTFS 信号分类。

| 层 | 参数 |
| --- | --- |
| Conv1 | 1×3 conv, 64 通道, stride (1,4) |
| Conv2 | 1×3 conv, 96 通道, stride (1,4) |
| Conv3 | 1×3 conv, 192 通道, stride (1,4) |
| Conv4 | 1×3 conv, 384 通道, stride (2,4) |
| 全连接 | 384 → num_class |
| 激活函数 | ReLU |
| 正则化 | BatchNorm + Dropout(0.4) |

### 2. RMLCNN（残差网络）
针对 RML2016.10a 数据集优化的残差卷积网络。

- **残差块**: 3 个残差块, 每块包含 2 层 3×3 卷积 + BatchNorm + SiLU + Dropout2d
- **初始卷积**: 大核卷积 (1×7), 快速下采样
- **全连接层**: 256 → 128 → num_class, 含 Dropout(0.5)
- **优化器**: AdamW (lr=0.0005, weight_decay=5e-4)
- **学习率调度**: OneCycleLR（余弦退火）
- **数据增强**: 噪声注入、时间偏移、幅度缩放、相位旋转
- **正则化**: 标签平滑 (ε=0.05)、梯度裁剪 (max_norm=5.0)、早停 (patience=12)

### 3. GPT（轻量级 Transformer）
基于 GPT 架构的轻量级 Transformer 模型。

| 组件 | 参数 |
| --- | --- |
| Embedding | 线性投影 2 → 128 |
| 注意力头 | 2 头因果自注意力 |
| Block 数 | 1 |
| Dropout | 0.1 |
| 位置编码 | 可学习位置编码 |

---

## 项目结构

```
OFDM-OTFS-modulation-recognition/
├── dataset.py           # Gauss 数据集加载器
├── model.py             # 模型定义 (CNN, RMLCNN, GPT)
├── utils.py             # 工具函数 (Averager, count_acc)
├── train_cnn.py         # CNN 训练脚本 (Gauss 数据集)
├── train_rml.py         # RMLCNN 训练脚本 (RML2016 数据集)
├── train_gpt.py         # GPT 训练脚本 (Gauss 数据集)
├── demodulate.py        # 推理/解调脚本
├── requirements.txt     # Python 依赖
├── README.md            # 项目说明文档
└── dataset/
    ├── Gauss/           # Gauss 数据集 (.mat 文件)
    │   ├── BPSK.mat
    │   ├── QPSK.mat
    │   ├── 8PSK.mat
    │   ├── 16QAM.mat
    │   ├── 64QAM.mat
    │   └── 256QAM.mat
    ├── OTFS/            # OTFS 数据集 (待补充)
    └── RML2016.10a_dict.pkl/
        └── RML2016.10a_dict.pkl  # RML2016 数据集
```

---

## 环境配置

```bash
pip install -r requirements.txt
```

依赖: `torch`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `tqdm`

---

## 快速开始

### 训练 CNN（Gauss 数据集）

```bash
python train_cnn.py
```

### 训练 RMLCNN（RML2016.10a 数据集）

```bash
python train_rml.py
```

### 训练 GPT（Gauss 数据集）

```bash
python train_gpt.py
```

### 推理/解调

```bash
# 使用 CNN 模型对单个 .mat 文件进行推理
python demodulate.py --model cnn --input sample.mat

# 使用 GPT 模型批量推理
python demodulate.py --model gpt --input_dir ./test_data

# 使用 RMLCNN 模型推理
python demodulate.py --model rml --input sample.npy
```

---

## 实验结果

### CNN (Gauss 数据集, SNR=-10dB)
- 6 类调制识别准确率: 待训练后更新

### RMLCNN (RML2016.10a 数据集)
- 6 类调制识别, SNR 范围 -20 ~ 18dB
- 平均准确率: 待训练后更新

### GPT (Gauss 数据集)
- 6 类调制识别准确率: 待训练后更新

---

## 引用

如果本项目对你的研究有帮助, 请引用以下论文:

```bibtex
@ARTICLE{10102600,
  author={Lin, Wensheng and Hou, Dongbin and Huang, Junsheng and Li, Lixin and Han, Zhu},
  journal={IEEE Transactions on Vehicular Technology}, 
  title={Transfer Learning for Automatic Modulation Recognition Using a Few Modulated Signal Samples}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/TVT.2023.3267270}}
```

---

## 许可证

本项目仅供学术研究使用。
