# Underwater-AMR-DeepLearning

**基于深度学习的水声信号调制方式识别系统**

本项目面向水声通信场景，利用深度学习技术对 OFDM 和 OTFS 调制信号进行自动调制识别（Automatic Modulation Recognition, AMR）。项目包含完整的数据集处理、模型构建、训练评估与测试流程，支持 CNN、Transformer（GPT）和残差网络等多种深度学习架构。

---

## 目录

- [项目背景](#项目背景)
- [数据集](#数据集)
- [模型架构](#模型架构)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [实验结果](#实验结果)
- [引用](#引用)
- [参考](#参考)

---

## 项目背景

水声信道具有多径效应严重、带宽有限、时变性强等特点，对信号调制方式的自动识别提出了更高要求。传统的调制识别方法依赖人工特征提取，难以适应复杂的水声信道环境。本项目基于深度学习，实现了端到端的自动调制识别系统，主要针对以下调制方式：

- **OFDM 信号**: BPSK, QPSK, 8PSK, 16QAM, 64QAM, 256QAM
- **OTFS 信号**: BPSK, QPSK, 8PSK, 16QAM, 64QAM, 256QAM
- **RML2016.10a 数据集**: BPSK, QPSK, 8PSK, QAM16, QAM64, PAM4
- **实测信道信号**: 12 种调制组合（HackRF 硬件采集）

---

## 数据集

### 1. Gauss 数据集（高斯信道 OFDM）

高斯信道下的 OFDM 调制信号数据集。包含 6 种调制类型（BPSK, QPSK, 8PSK, 16QAM, 64QAM, 256QAM），信噪比范围 -10dB ~ 20dB（间隔 2dB）。每种信噪比下每类信号生成 2000 条数据，总计 **192,000 条**。信号经过信道编码、调制、串并转换、扩频、脉冲成型等完整处理流程。

### 2. Rayleigh 数据集（瑞利信道 OFDM）

瑞利信道下的 OFDM 信号，基本信息与 Gauss 数据集相同，信噪比范围 -20dB ~ 10dB（间隔 2dB）。

### 3. OTFS 数据集（高速移动信道）

OTFS（Orthogonal Time Frequency Space）调制是面向高速移动场景（如无人机通信、高铁通信）的关键 6G 技术，能有效对抗多普勒效应。本数据集包含 6 种 OTFS 信号，信噪比范围 -20dB ~ 10dB（间隔 2dB）。

### 4. RML2016.10a 数据集

经典的调制识别基准数据集，包含 6 种调制类型，信号格式为 I/Q 双通道（2×128），信噪比范围 -20dB ~ 30dB（间隔 2dB）。本项目中用于 RMLCNN 模型的训练与评估。

### 5. 实测信道 OFDM 数据集（HackRF）

使用两个 HackRF 硬件在室内复杂电磁环境下采集，通信距离 5~10m。包含 12 种调制组合（如 BPSK+BPSK, BPSK+QPSK, QPSK+16QAM 等），每种 30,000 条数据。

> 数据集下载链接：[OneDrive](https://mailnwpueducn-my.sharepoint.com/:u:/g/personal/houdongbin_mail_nwpu_edu_cn/Ef8WIcCdVwFGhXZyQhUk-w0BZOb0MluwHo-rDzm8jFTR3A?e=qlxUet)

---

## 模型架构

### 1. UnderwaterCNN（水下信道 CNN）

多层卷积神经网络，适用于水声 OFDM/OTFS 信号分类：

| 层       | 参数                  |
| -------- | --------------------- |
| Conv1    | 1×3 conv, 32 通道     |
| Conv2    | 1×3 conv, 64 通道     |
| Conv3    | 1×3 conv, 128 通道    |
| Conv4    | 1×3 conv, 256 通道    |
| 全连接   | 256 → 128 → num_class |
| 激活函数 | ReLU                  |
| 池化     | MaxPool2d(1×2)        |

### 2. RMLCNN（RML 数据集残差网络）

针对 RML2016.10a 数据集优化的残差卷积网络：

- **残差块（ResidualBlock）**: 3 个残差块，每块包含 2 层 3×3 卷积 + BatchNorm + SiLU 激活 + Dropout2d
- **初始卷积**: 大核卷积 (1×7)，快速下采样
- **全连接层**: 256 → 128 → num_class，含 Dropout(0.5)
- **优化器**: AdamW (lr=0.0005, weight_decay=5e-4)
- **学习率调度**: OneCycleLR（余弦退火）
- **数据增强**: 噪声注入、时间偏移、幅度缩放、相位旋转
- **正则化**: 标签平滑 (ε=0.05)、梯度裁剪 (max_norm=5.0)、早停 (patience=12)

### 3. GPT（轻量级 Transformer）

基于 GPT-2/GPT-3 架构的轻量级 Transformer 模型：

| 组件      | 参数                |
| --------- | ------------------- |
| Embedding | 线性投影 1024 → 128 |
| 注意力头  | 2 头因果自注意力    |
| Block 数  | 1                   |
| Dropout   | 0.1                 |
| 位置编码  | 可学习位置编码      |

---

## 项目结构

```
Underwater-AMR-DeepLearning/
├── model.py                 # 模型定义（UnderwaterCNN, RMLCNN, ResidualBlock, GPT）
├── rml.py                   # RML2016.10a 数据集加载
├── dataset.py               # HackRF 实测数据集加载
├── Underwater_dataset.py    # 水声 OFDM/OTFS 数据集加载（.mat 文件）
├── utils.py                 # 工具函数（Averager, count_acc, 置信区间）
├── train_cnn.py             # UnderwaterCNN 训练脚本
├── train_gpt.py             # GPT 训练脚本
├── train_rml.py             # RMLCNN 训练脚本（含数据增强、早停等）
├── snr_test.py              # 独立 SNR 测试脚本
├── fushi.py                 # 日志解析与可视化
├── requirements.txt         # Python 依赖
├── scripts/
│   └── setup_server.sh      # 远程服务器一键部署脚本
├── dataset/
│   └── Gauss/               # 数据集存放目录
└── model/                   # 模型权重保存目录
```

---

## 环境配置

### 本地环境

```bash
pip install -r requirements.txt
```

依赖清单：`torch`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `tqdm`

### 远程服务器（SeataCloud）

SSH 登录后一键部署：

```bash
git clone https://github.com/hahathinker/Underwater-AMR-DeepLearning.git
cd Underwater-AMR-DeepLearning
bash scripts/setup_server.sh
conda activate rml
```

脚本会自动安装 Miniconda、创建 `rml` 环境、安装 PyTorch（CUDA 版）及所有依赖。

---

## 快速开始

### 训练 RMLCNN（RML2016.10a 数据集）

```bash
python train_rml.py
```

- 自动下载/加载 RML2016.10a_dict.pkl
- 支持多 SNR 训练与独立测试
- 训练过程自动保存最佳模型、绘制混淆矩阵

### 训练 UnderwaterCNN（水声 OFDM 数据集）

```bash
python train_cnn.py
```

### 训练 GPT（水声 OFDM 数据集）

```bash
python train_gpt.py
```

### SNR 独立测试

```bash
python snr_test.py
```

---

## 实验结果

### RMLCNN 在 RML2016.10a 上的性能

| SNR (dB) | 准确率 |
| -------- | ------ |
| 0        | ~45%   |
| 10       | ~60%   |
| 18       | ~65%+  |

> 详细结果请运行 `python train_rml.py` 查看训练日志。

---

## 引用

如果本项目对你的研究有帮助，请引用以下论文：

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

## 参考

[1] R. Hadani et al., "Orthogonal Time Frequency Space Modulation," 2017 IEEE Wireless Communications and Networking Conference (WCNC), San Francisco, CA, USA, 2017, pp. 1-6.

[2] Brown T, Mann B, Ryder N, et al, "Language models are few-shot learners," in Advances in neural information processing systems, vol. 33, no. 1, pp. 1877-1901, Jun. 2020.

---

## 许可证

本项目仅供学术研究使用。
