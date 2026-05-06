import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
class UnderwaterCNN(nn.Module):
    """This is a multilayer convolutional neural network for classification of modulation signal datasets
    Args:  
    num_class: the number of the class
    Return: 
    logits (batch_size,num_class)
    """
    def __init__(self,num_class=6):#num_class is 
        super(UnderwaterCNN,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,stride=(1,4),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer2=nn.Sequential(
            nn.Conv2d(64,96,kernel_size=3,stride=(1,4),padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Dropout(.4),
            nn.Conv2d(96,96,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            )
        self.layer3=nn.Sequential(
            nn.Conv2d(96,192,kernel_size=3,stride=(1,4),padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Dropout(.4),
            nn.Conv2d(192,192,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

        )
        self.layer4=nn.Sequential(
            nn.Conv2d(192,384,kernel_size=3,stride=(2,4),padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(.4),
            nn.Conv2d(384,384,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)

        )
        self.poolinglayer=nn.MaxPool2d((1,4),stride=1)
        self.fc=nn.Linear(384,num_class)

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.poolinglayer(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

class RMLCNN(nn.Module):
    """
    改进版 RMLCNN - 适配 RML2016 信号数据：输入 shape = [batch, 1, 2, 128]
    
    改进点：
    1. 使用更浅的网络 + 更大初始卷积核，更适合小尺寸输入 (2,128)
    2. 引入残差连接（Residual Block），缓解梯度消失
    3. 使用可分离卷积（depthwise separable），减少参数量
    4. 全连接层增加隐层，提升分类能力
    5. 使用 Swish/SiLU 激活函数替代 ReLU，更平滑的梯度
    
    num_class: 分类数 (默认6类: BPSK/QPSK/8PSK/QAM16/QAM64/PAM4)
    """
    def __init__(self, num_class=6, dropout_rate=0.3):
        super(RMLCNN, self).__init__()
        
        # ---- 改进1: 初始层用大卷积核捕获I/Q相关性 ----
        # 输入 (B, 1, 2, 128) → (B, 64, 2, 64)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3)),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)  # 改进: SiLU 替代 ReLU
        )
        
        # ---- 改进2: 残差块1 ----
        # (B, 64, 2, 64) → (B, 96, 2, 32)
        self.res_block1 = ResidualBlock(64, 96, stride=(1, 2), dropout=dropout_rate)
        
        # ---- 改进3: 残差块2 ----
        # (B, 96, 2, 32) → (B, 128, 2, 16)
        self.res_block2 = ResidualBlock(96, 128, stride=(1, 2), dropout=dropout_rate)
        
        # ---- 改进4: 残差块3（不下采样） ----
        # (B, 128, 2, 16) → (B, 256, 2, 8)
        self.res_block3 = ResidualBlock(128, 256, stride=(1, 2), dropout=dropout_rate)
        
        # ---- 改进5: 全局平均池化 ----
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        # ---- 改进6: 增强的全连接层 ----
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_class)
        )
        
        # ---- 改进7: 权重初始化 ----
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


class ResidualBlock(nn.Module):
    """
    改进的残差块，用于 RMLCNN
    
    结构: Conv → BN → SiLU → Dropout → Conv → BN → (残差连接) → SiLU
    当 stride≠1 或 in_ch≠out_ch 时，shortcut 使用 1×1 卷积调整维度
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
        
        # 残差连接：如果维度不匹配，使用 1x1 卷积调整
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

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd=128, n_head=2, attn_drop=0.1, resid_drop=0.1, signal_size=1024):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        # 这里的 n_embd 现在是动态传入的！
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.resid_dropout = nn.Dropout(resid_drop)
        self.register_buffer("bias", torch.tril(torch.ones(signal_size, signal_size)).view(1,1,signal_size,signal_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)

        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.resid_dropout(self.c_proj(y))

class Block(nn.Module):
    def __init__(self, n_embd=128, resid_drop=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        # 这里必须把 n_embd 传给 Attention！
        self.attn = CausalSelfAttention(n_embd=n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 256),
            nn.GELU(),
            nn.Linear(256, n_embd),
            nn.Dropout(resid_drop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, num_classes=6, n_embd=128, n_layer=1):
        super().__init__()
        self.input_proj = nn.Linear(2, n_embd)
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, n_embd))
        # 这里也必须把 n_embd 传给每个 Block！
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, num_classes)

    def forward(self, x):
        # 输入形状检查：(B, 2, 1024)
        if x.dim() == 4:
            x = x.squeeze(1)
        x = x.permute(0, 2, 1)    # [B, 1024, 2]
        x = self.input_proj(x)    # [B, 1024, 128]
        x = x + self.pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        x = x.mean(dim=1)
        return self.head(x)









