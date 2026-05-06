#!/bin/bash
# ============================================================
# 远程服务器环境初始化脚本
# 在 SeataCloud 服务器上运行：bash scripts/setup_server.sh
# ============================================================

set -e

echo "=========================================="
echo "  开始初始化深度学习训练环境"
echo "=========================================="

# 1. 更新系统
echo "[1/5] 更新系统包..."
apt-get update -y && apt-get upgrade -y

# 2. 安装基础工具
echo "[2/5] 安装基础工具..."
apt-get install -y build-essential git curl wget unzip

# 3. 安装 Miniconda（如果未安装）
if ! command -v conda &> /dev/null; then
    echo "[3/5] 安装 Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init
else
    echo "[3/5] Miniconda 已安装，跳过"
fi

# 4. 创建 conda 环境
echo "[4/5] 创建 conda 环境 'rml'..."
conda create -n rml python=3.9 -y
source activate rml

# 5. 安装 PyTorch（CUDA 版本）
echo "[5/5] 安装 PyTorch 和依赖..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy matplotlib seaborn scikit-learn tqdm

echo ""
echo "=========================================="
echo "  ✅ 环境初始化完成！"
echo "  使用 conda activate rml 激活环境"
echo "=========================================="
