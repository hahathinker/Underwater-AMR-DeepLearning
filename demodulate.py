"""
解调/推理脚本
使用训练好的模型对新的 I/Q 信号进行调制方式识别

用法:
    # 使用预训练模型对单个样本进行推理
    python demodulate.py --model cnn --input sample.mat
    
    # 对目录下所有 .mat 文件进行批量推理
    python demodulate.py --model gpt --input_dir ./test_data
    
    # 使用 RMLCNN 模型推理
    python demodulate.py --model rml --input sample.npy
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from scipy.io import loadmat, savemat
from model import CNN, RMLCNN, GPT
from utils import ensure_path
from tqdm import tqdm


# 调制类型映射
GAUSS_CLASS_NAMES = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "256QAM"]
RML_CLASS_NAMES = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4"]


def load_model(model_type: str, num_class: int, device: torch.device):
    """
    加载训练好的模型
    
    Args:
        model_type: 模型类型 ('cnn', 'rml', 'gpt')
        num_class: 分类数
        device: 设备
    
    Returns:
        加载好权重的模型
    """
    model_paths = {
        'cnn': './model/cnn_underwater.pth',
        'rml': './model/rmlcnn.pth',
        'gpt': './model/gpt_underwater.pth',
    }
    
    model_map = {
        'cnn': CNN(num_class=num_class),
        'rml': RMLCNN(num_class=num_class),
        'gpt': GPT(num_classes=num_class),
    }
    
    model_path = model_paths.get(model_type)
    if model_path is None:
        raise ValueError(f"未知的模型类型: {model_type}, 可选: cnn, rml, gpt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}, 请先运行训练脚本")
    
    model = model_map[model_type]
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"模型已加载: {model_path}")
    return model


def preprocess_signal(signal: np.ndarray, model_type: str) -> torch.Tensor:
    """
    预处理输入信号
    
    Args:
        signal: 输入信号, shape (2, 1024) 或 (2, 128)
        model_type: 模型类型
    
    Returns:
        预处理后的张量
    """
    # 确保信号是 float32
    signal = signal.astype(np.float32)
    
    # 转换为张量
    x = torch.FloatTensor(signal)
    
    # CNN/RMLCNN 需要添加通道维度: (2, 1024) → (1, 2, 1024)
    if model_type in ['cnn', 'rml']:
        x = x.unsqueeze(0)  # (1, 2, 1024)
    
    # 添加 batch 维度: (C, H, W) → (1, C, H, W)
    x = x.unsqueeze(0)
    
    return x


def predict_single(model, signal: np.ndarray, model_type: str,
                   class_names: list, device: torch.device) -> dict:
    """
    对单个信号进行预测
    
    Args:
        model: 加载的模型
        signal: 输入信号, shape (2, 1024) 或 (2, 128)
        model_type: 模型类型
        class_names: 类别名称列表
        device: 设备
    
    Returns:
        预测结果字典
    """
    x = preprocess_signal(signal, model_type)
    x = x.to(device)
    
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
    
    # 获取所有类别的概率
    all_probs = probs.cpu().numpy()[0]
    
    result = {
        'predicted_class': class_names[pred_class],
        'predicted_index': pred_class,
        'confidence': confidence,
        'all_probabilities': {
            name: float(prob) for name, prob in zip(class_names, all_probs)
        }
    }
    
    return result


def predict_batch(model, signals: np.ndarray, model_type: str,
                  class_names: list, device: torch.device) -> list:
    """
    对批量信号进行预测
    
    Args:
        model: 加载的模型
        signals: 输入信号数组, shape (N, 2, 1024) 或 (N, 2, 128)
        model_type: 模型类型
        class_names: 类别名称列表
        device: 设备
    
    Returns:
        预测结果列表
    """
    results = []
    for i in tqdm(range(len(signals)), desc="推理进度"):
        result = predict_single(model, signals[i], model_type, class_names, device)
        results.append(result)
    
    return results


def load_input_file(file_path: str) -> np.ndarray:
    """
    加载输入文件 (.mat 或 .npy)
    
    Args:
        file_path: 文件路径
    
    Returns:
        信号数据, shape (2, 1024) 或 (2, 128)
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.mat':
        data = loadmat(file_path)
        # 尝试常见的变量名
        for key in ['data', 'dataset', 'signal', 'iq', 'X']:
            if key in data:
                return data[key]
        # 如果找不到, 返回第一个非元数据变量
        for key in data:
            if not key.startswith('__'):
                return data[key]
        raise ValueError(f"无法从 {file_path} 中解析信号数据")
    
    elif ext == '.npy':
        return np.load(file_path)
    
    else:
        raise ValueError(f"不支持的文件格式: {ext}, 仅支持 .mat 和 .npy")


def main():
    parser = argparse.ArgumentParser(description="调制识别推理工具")
    parser.add_argument('--model', type=str, required=True,
                        choices=['cnn', 'rml', 'gpt'],
                        help='模型类型: cnn (UnderwaterCNN), rml (RMLCNN), gpt (GPT)')
    parser.add_argument('--input', type=str, default=None,
                        help='输入文件路径 (.mat 或 .npy)')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='输入目录路径 (批量推理)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出结果保存路径 (.mat 或 .json)')
    parser.add_argument('--num_class', type=int, default=6,
                        help='分类数')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 确定类别名称
    if args.model == 'rml':
        class_names = RML_CLASS_NAMES
    else:
        class_names = GAUSS_CLASS_NAMES
    
    # 加载模型
    model = load_model(args.model, args.num_class, device)
    
    # 单文件推理
    if args.input:
        print(f"加载输入文件: {args.input}")
        signal = load_input_file(args.input)
        print(f"信号形状: {signal.shape}")
        
        result = predict_single(model, signal, args.model, class_names, device)
        
        print("\n========== 推理结果 ==========")
        print(f"预测调制类型: {result['predicted_class']}")
        print(f"置信度: {result['confidence']:.4f}")
        print("\n各类别概率:")
        for name, prob in sorted(result['all_probabilities'].items(),
                                  key=lambda x: x[1], reverse=True):
            print(f"  {name}: {prob:.4f}")
        
        # 保存结果
        if args.output:
            ensure_path(os.path.dirname(args.output) if os.path.dirname(args.output) else '.')
            ext = os.path.splitext(args.output)[1].lower()
            if ext == '.mat':
                savemat(args.output, {
                    'predicted_class': result['predicted_class'],
                    'predicted_index': result['predicted_index'],
                    'confidence': result['confidence'],
                    'probabilities': list(result['all_probabilities'].values()),
                    'class_names': list(result['all_probabilities'].keys()),
                })
            else:
                import json
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
            print(f"\n结果已保存至: {args.output}")
    
    # 批量推理
    elif args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"错误: 目录不存在 {args.input_dir}")
            return
        
        # 收集所有 .mat 和 .npy 文件
        files = []
        for f in os.listdir(args.input_dir):
            if f.endswith('.mat') or f.endswith('.npy'):
                files.append(os.path.join(args.input_dir, f))
        
        if not files:
            print(f"目录中没有找到 .mat 或 .npy 文件: {args.input_dir}")
            return
        
        print(f"找到 {len(files)} 个文件, 开始批量推理...")
        
        all_results = {}
        for file_path in tqdm(files, desc="批量推理"):
            try:
                signal = load_input_file(file_path)
                result = predict_single(model, signal, args.model, class_names, device)
                all_results[os.path.basename(file_path)] = result
            except Exception as e:
                print(f"处理 {file_path} 时出错: {e}")
        
        # 输出汇总
        print("\n========== 批量推理结果汇总 ==========")
        for fname, result in all_results.items():
            print(f"{fname}: {result['predicted_class']} (置信度: {result['confidence']:.4f})")
        
        # 保存结果
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n结果已保存至: {args.output}")
    
    else:
        print("请指定 --input 或 --input_dir 参数")
        parser.print_help()


if __name__ == '__main__':
    main()
