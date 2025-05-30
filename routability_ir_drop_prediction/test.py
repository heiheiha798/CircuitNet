# Copyright 2022 CircuitNet. All rights reserved.

from __future__ import print_function

import os
import os.path as osp
import json
import numpy as np
import torch

from tqdm import tqdm

from datasets.build_dataset import build_dataset
from utils.metrics import build_metric, build_roc_prc_metric
from models.build_model import build_model
from utils.configs import Parser

# --- 将量化辅助函数放在全局，方便复用 ---d
def quantize_dequantize_int(tensor, bits):
    q_min = -2**(bits - 1)
    q_max = 2**(bits - 1) - 1
    scale = tensor.abs().max() / q_max
    if scale == 0: return tensor
    quantized = torch.round(tensor / scale).clamp(q_min, q_max)
    dequantized = quantized * scale
    return dequantized

def quantize_dequantize_fp4(tensor):
    fp4_levels = torch.tensor([
        0.0, 0.0625, 0.125, 0.25, 0.5, 1.0, 1.5, 2.0,
        -0.0625, -0.125, -0.25, -0.5, -1.0, -1.5, -2.0
    ], device=tensor.device, dtype=tensor.dtype)
    max_val = tensor.abs().max()
    if max_val == 0: return tensor
    scaled_levels = fp4_levels * max_val
    diff = tensor.unsqueeze(-1) - scaled_levels
    indices = torch.abs(diff).argmin(dim=-1)
    dequantized = scaled_levels[indices]
    return dequantized
    
def quantize_tensor(tensor, precision):
    """根据精度字符串对单个tensor进行伪量化"""
    if precision == 'INT8':
        return quantize_dequantize_int(tensor, bits=8)
    elif precision == 'INT4':
        return quantize_dequantize_int(tensor, bits=4)
    elif precision == 'FP8':
        if hasattr(torch, 'float8_e4m3fn'):
            return tensor.to(torch.float8_e4m3fn).to(torch.float32)
        else:
            print("Warning: FP8 native dtype not found. Skipping FP8 quantization.")
            return tensor
    elif precision == 'FP4':
        return quantize_dequantize_fp4(tensor)
    return tensor

# --- 新的量化处理函数 ---

def quantize_weights(model, precision, module_name='all'):
    """
    对指定模块的权重进行伪量化。
    """
    print(f"===> Applying fake {precision} quantization to WEIGHTS of module: [{module_name}]")
    
    target_module = model
    if module_name != 'all':
        if not hasattr(model, module_name):
            print(f"Error: Model has no module named '{module_name}'. Applying to 'all' instead.")
            print(f"Available top-level modules: {[name for name, _ in model.named_children()]}")
        else:
            target_module = getattr(model, module_name)

    target_module.eval()
    with torch.no_grad():
        for param in target_module.parameters():
            param.data = quantize_tensor(param.data, precision)

def add_activation_hooks(model, precision, module_name='all'):
    """
    通过hook对指定模块的激活值进行伪量化。
    """
    print(f"===> Applying fake {precision} quantization to ACTIVATIONS of module: [{module_name}]")

    target_module = model
    if module_name != 'all':
        if not hasattr(model, module_name):
            print(f"Error: Model has no module named '{module_name}'. Applying to 'all' instead.")
            print(f"Available top-level modules: {[name for name, _ in model.named_children()]}")
        else:
            target_module = getattr(model, module_name)

    # --- 修改后的Hook函数 ---
    def quantization_hook(module, input, output):
        # 检查输出是否为元组
        if isinstance(output, tuple):
            # 如果是元组，遍历其中每个元素
            quantized_items = []
            for item in output:
                # 只对Tensor类型的元素进行量化
                if isinstance(item, torch.Tensor):
                    quantized_items.append(quantize_tensor(item, precision))
                else:
                    # 其他类型的元素（如None）保持原样
                    quantized_items.append(item)
            return tuple(quantized_items)
        # 如果输出是单个Tensor，直接量化
        elif isinstance(output, torch.Tensor):
            return quantize_tensor(output, precision)
        # 其他情况，直接返回
        else:
            return output

    # 对目标模块下的所有子模块注册hook
    for submodule in target_module.modules():
        submodule.register_forward_hook(quantization_hook)


def test():
    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)
    if arg.arg_file is not None:
        with open(arg.arg_file, 'rt') as f:
            arg_dict.update(json.load(f))

    arg_dict['ann_file'] = arg_dict['ann_file_test'] 
    arg_dict['test_mode'] = True

    print('===> Loading datasets')
    # build_dataset 现在会返回一个配置好批处理和并行加载的 DataLoader
    dataset = build_dataset(arg_dict)

    print('===> Building model')
    model = build_model(arg_dict)
    
    if not arg_dict['cpu']:
        model = model.cuda()
    
    # 根据参数应用不同的量化策略
    if arg_dict['precision'] != 'FP32':
        if arg_dict['quant_target'] == 'weights':
            quantize_weights(model, arg_dict['precision'], arg_dict['quant_module'])
        elif arg_dict['quant_target'] == 'activations':
            add_activation_hooks(model, arg_dict['precision'], arg_dict['quant_module'])
        elif arg_dict['quant_target'] == 'both':
            quantize_weights(model, arg_dict['precision'], arg_dict['quant_module'])
            add_activation_hooks(model, arg_dict['precision'], arg_dict['quant_module'])

    metrics = {k:build_metric(k) for k in arg_dict['eval_metric']}
    # avg_metrics 现在将累加每个样本的指标值
    avg_metrics = {k:0 for k in arg_dict['eval_metric']}

    count = 0
    # dataset 是 DataLoader，其长度是批次数
    with tqdm(total=len(dataset)) as bar:
        # 每次循环拿出一个批次的数据
        for features, labels, label_paths in dataset:
            if arg_dict['cpu']:
                input, target = features, labels
            else:
                input, target = features.cuda(), labels.cuda()

            # 模型对整个批次进行推理
            prediction = model(input)

            # --- 核心修改：遍历批次中的每个样本 ---
            # input.shape[0] 是当前批次的大小
            for i in range(input.shape[0]):
                # 从批次中逐个取出样本
                single_target = target[i]
                single_prediction = prediction[i]

                # 对单个样本调用评测函数
                for metric, metric_func in metrics.items():
                    # squeeze(0) 用于移除单个样本的通道维度 (如果为1)
                    metric_val = metric_func(single_target.cpu(), single_prediction.squeeze(0).cpu())
                    if not metric_val == 1:
                        # 累加每个样本的指标值
                        avg_metrics[metric] += metric_val
            
            # plot_roc 部分如果需要按样本保存，也应放在上面的循环内
            if arg_dict['plot_roc']:
                # ... 此处逻辑暂不处理，因为重点是评测 ...
                pass 
            
            bar.update(1)
    
    # --- 核心修改：计算最终平均指标 ---
    # dataset.dataset 指向原始的Dataset对象，其长度是样本总数
    num_samples = len(dataset.dataset)
    for metric, total_metric in avg_metrics.items():
        # 用指标总和除以样本总数，得到最终平均值
        print("===> Avg. {}: {:.4f}".format(metric, total_metric / num_samples))

    if arg_dict['plot_roc']:
        roc_metric, _ = build_roc_prc_metric(**arg_dict)
        print("\n===> AUC of ROC. {:.4f}".format(roc_metric))

if __name__ == "__main__":
    test()
