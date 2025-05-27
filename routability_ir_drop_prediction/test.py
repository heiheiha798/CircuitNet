# Copyright 2022 CircuitNet. All rights reserved.

from __future__ import print_function

import os
import os.path as osp
import json
import numpy as np
import torch # 新增导入

from tqdm import tqdm

from datasets.build_dataset import build_dataset
from utils.metrics import build_metric, build_roc_prc_metric
from models.build_model import build_model
from utils.configs import Parser


def apply_fake_quantization(model, precision='FP32'):
    """
    Applies fake quantization to the model's weights for various precisions.
    """
    if precision == 'FP32':
        print("===> Testing with original FP32 precision.")
        return model

    # --- Helper function for Integer Quantization ---
    def quantize_dequantize_int(tensor, bits):
        q_min = -2**(bits - 1)
        q_max = 2**(bits - 1) - 1
        scale = tensor.abs().max() / q_max
        if scale == 0: return tensor # Avoid division by zero for all-zero tensors
        quantized = torch.round(tensor / scale).clamp(q_min, q_max)
        dequantized = quantized * scale
        return dequantized

    # --- Helper function for FP4 Fake Quantization ---
    def quantize_dequantize_fp4(tensor):
        # Define a non-uniform grid for FP4 (1 sign, 2 exponent, 1 mantissa)
        # These 8 positive values + their negatives + zero make up the 16 possible values
        fp4_levels = torch.tensor([
            0.0, 0.0625, 0.125, 0.25, 0.5, 1.0, 1.5, 2.0, # Positive values
            -0.0625, -0.125, -0.25, -0.5, -1.0, -1.5, -2.0 # Negative values
        ], device=tensor.device, dtype=tensor.dtype)
        
        # Scale levels to the tensor's range
        max_val = tensor.abs().max()
        if max_val == 0: return tensor
        scaled_levels = fp4_levels * max_val

        # Snap each value in the tensor to the nearest level in the grid
        # Add dimensions for broadcasting
        diff = tensor.unsqueeze(-1) - scaled_levels
        indices = torch.abs(diff).argmin(dim=-1)
        dequantized = scaled_levels[indices]
        return dequantized

    print(f"===> Applying fake {precision} quantization to model weights.")
    model.eval()
    with torch.no_grad():
        for param in model.parameters():
            if precision == 'INT8':
                param.data = quantize_dequantize_int(param.data, bits=8)
            elif precision == 'INT4':
                param.data = quantize_dequantize_int(param.data, bits=4)
            elif precision == 'FP8':
                # Use native torch support if available (PyTorch 2.1+)
                if hasattr(torch, 'float8_e4m3fn'):
                    param.data = param.data.to(torch.float8_e4m3fn).to(torch.float32)
                else:
                    print("Warning: FP8 native dtype not found in your PyTorch version. Skipping FP8.")
                    return model # Exit early if not supported
            elif precision == 'FP4':
                param.data = quantize_dequantize_fp4(param.data)
    
    return model


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
    # Initialize dataset
    dataset = build_dataset(arg_dict)

    print('===> Building model')
    # Initialize model parameters
    model = build_model(arg_dict)
    if not arg_dict['cpu']:
        model = model.cuda()

    # 应用伪量化
    model = apply_fake_quantization(model, arg_dict['precision'])

    # Build metrics
    metrics = {k:build_metric(k) for k in arg_dict['eval_metric']}
    avg_metrics = {k:0 for k in arg_dict['eval_metric']}

    count =0
    with tqdm(total=len(dataset)) as bar:
        for feature, label, label_path in dataset:
            if arg_dict['cpu']:
                input, target = feature, label
            else:
                input, target = feature.cuda(), label.cuda()

            # 注意：低精度量化后，模型权重仍在FP32中模拟，因此输入不需要转换
            # （这与FP16的真半精度推理不同）
            prediction = model(input)
            for metric, metric_func in metrics.items():
                if not metric_func(target.cpu(), prediction.squeeze(1).cpu()) == 1:
                    avg_metrics[metric] += metric_func(target.cpu(), prediction.squeeze(1).cpu())

            if arg_dict['plot_roc']:
                save_path = osp.join(arg_dict['save_path'], 'test_result')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                file_name = osp.splitext(osp.basename(label_path[0]))[0]
                save_path = osp.join(save_path, f'{file_name}.npy')
                output_final = prediction.float().detach().cpu().numpy()
                np.save(save_path, output_final)
                count +=1

            bar.update(1)
    
    for metric, avg_metric in avg_metrics.items():
        print("===> Avg. {}: {:.4f}".format(metric, avg_metric / len(dataset))) 

    # eval roc&prc
    if arg_dict['plot_roc']:
        roc_metric, _ = build_roc_prc_metric(**arg_dict)
        print("\n===> AUC of ROC. {:.4f}".format(roc_metric))


if __name__ == "__main__":
    test()
