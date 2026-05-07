"""
模型定义 - CNN, RMLCNN, GPT

包含三种调制识别模型架构:
1. CNN - 多层卷积网络 (Gauss 数据集)
2. RMLCNN - 残差网络 (RML2016 数据集)
3. GPT - 轻量级 Transformer (Gauss 数据集)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    多层卷积神经网络, 适用于 OFDM 信号调制识别
    
    输入: (batch, 1, 2, 1024) - I/Q 双通道时域信号
    输出: (batch, num_class) - 分类 logits
    
    结构:
    Conv2D(1→64) → Conv2D(64→96) → MaxPool2d → Conv2D(96→192) → MaxPool2d → Conv2D(192→384) → AdaptiveMaxPool2d → Linear(384→num_class)
    """
    def __init__(self, num_class=6):
        super(CNN, self).__init__()
        
        # 第一层: 大卷积核捕获 I/Q 相关性
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 第二层
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=(2, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        
        # 第三层
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        
        # 第四层
        self.conv4 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(output_size=(1, 1))
        )
        
        # 分类器
        self.classifier = nn.Linear(384, num_class)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """
    残差块 (Residual Block)
    
    结构: Conv → BN → SiLU → Dropout → Conv → BN → (残差连接) → SiLU
    当 stride≠1 或 in_ch≠out_ch 时, shortcut 使用 1×1 卷积调整维度
    """
    def __init__(self, in_ch, out_ch, stride=(1, 1), dropout=0.3):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.silu1 = nn.SiLU(inplace=True)
        self.drop1 = nn.Dropout2d(dropout)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.silu2 = nn.SiLU(inplace=True)
        
        # 残差连接: 如果维度不匹配, 使用 1x1 卷积调整
        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu1(out)
        out = self.drop1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity  # 残差连接
        out = self.silu2(out)
        
        return out


class RMLCNN(nn.Module):
    """
    改进版 RMLCNN - 适配 RML2016 信号数据
    
    输入: (batch, 1, 2, 128) - I/Q 双通道信号
    输出: (batch, num_class) - 分类 logits
    
    改进点:
    1. 使用更浅的网络 + 更大初始卷积核, 更适合小尺寸输入 (2,128)
    2. 引入残差连接 (Residual Block), 缓解梯度消失
    3. 使用 Swish/SiLU 激活函数替代 ReLU, 更平滑的梯度
    4. 全局平均池化替代全连接层前的展平
    """
    def __init__(self, num_class=6, dropout_rate=0.3):
        super(RMLCNN, self).__init__()
        
        # 初始层: 大卷积核捕获 I/Q 相关性
        # 输入 (B, 1, 2, 128) → (B, 64, 2, 64)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3)),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )
        
        # 残差块1: (B, 64, 2, 64) → (B, 96, 2, 32)
        self.res_block1 = ResidualBlock(64, 96, stride=(1, 2), dropout=dropout_rate)
        
        # 残差块2: (B, 96, 2, 32) → (B, 128, 2, 16)
        self.res_block2 = ResidualBlock(96, 128, stride=(1, 2), dropout=dropout_rate)
        
        # 残差块3: (B, 128, 2, 16) → (B, 256, 2, 8)
        self.res_block3 = ResidualBlock(128, 256, stride=(1, 2), dropout=dropout_rate)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_class)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SelfAttention(nn.Module):
    """双向自注意力机制（信号分类用，无需因果掩码）"""
    def __init__(self, n_embd=128, n_head=4, attn_drop=0.1, resid_drop=0.1):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        assert n_embd % n_head == 0, f"n_embd({n_embd}) must be divisible by n_head({n_head})"
        
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.resid_dropout = nn.Dropout(resid_drop)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """Transformer Block (LayerNorm + Attention + MLP)"""
    def __init__(self, n_embd=128, n_head=4, resid_drop=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd=n_embd, n_head=n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.GELU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(resid_drop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    轻量级 Transformer 模型, 适用于信号序列分类
    
    输入: (batch, 2, 1024) - I/Q 双通道信号
    输出: (batch, num_class) - 分类 logits
    
    改进:
    1. 双向注意力 (非因果) — 更适合信号分类
    2. 更多注意力头 (4头)
    3. 更宽的 MLP (4x)
    4. 更好的权重初始化
    """
    def __init__(self, num_classes=6, n_embd=128, n_layer=2, n_head=4):
        super().__init__()
        self.input_proj = nn.Linear(2, n_embd)
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, n_embd) * 0.02)
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.Sequential(*[
            Block(n_embd=n_embd, n_head=n_head) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, num_classes)
        
        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        x = x.permute(0, 2, 1)    # [B, 1024, 2]
        x = self.input_proj(x)    # [B, 1024, 128]
        x = x + self.pos_emb
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        x = x.mean(dim=1)         # 全局平均池化
        return self.head(x)
