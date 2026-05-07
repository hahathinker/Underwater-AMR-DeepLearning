"""
工具函数模块
包含 Averager（平均值计算）、count_acc（准确率计算）等通用工具
"""

import os
import torch
import torch.nn.functional as F
import numpy as np


def ensure_path(path: str):
    """确保路径存在, 不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


class Averager():
    """滑动平均值计算器"""
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    """计算分类准确率
    
    Args:
        logits: 模型输出 logits, shape (batch, num_class)
        label: 真实标签, shape (batch,)
    
    Returns:
        准确率 (float)
    """
    pred = F.softmax(logits, dim=1).argmax(dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    return (pred == label).type(torch.FloatTensor).mean().item()


def compute_confidence_interval(data):
    """计算均值和置信区间 (95% confidence)
    
    Args:
        data: 输入数据列表
    
    Returns:
        m: 均值
        pm: 置信区间
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm
