# plot_loss.py

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_loss_curve(log_path, save_dir):
    """
    读取保存的 loss 记录文件并绘制曲线图。
    
    Args:
        log_path (str): loss_history.npy 文件的路径。
        save_dir (str): 曲线图图片的保存目录。
    """
    # 步骤 1: 加载 .npy 文件
    # np.load() 会直接读取文件内容并将其转换为一个 Numpy 数组
    try:
        loss_data = np.load(log_path)
    except FileNotFoundError:
        print(f"错误: 找不到日志文件 {log_path}")
        print("请先确认训练已运行并生成了 loss_history.npy 文件。")
        return

    # 步骤 2: 提取数据列
    # loss_data 是一个 N x 2 的数组
    # 所有行的第 0 列是迭代次数 (x轴)
    # 所有行的第 1 列是 loss 值 (y轴)
    iterations = loss_data[:, 0]
    losses = loss_data[:, 1]

    # 步骤 3: 使用 Matplotlib 绘图
    plt.figure(figsize=(12, 7))  # 创建一个图形窗口
    plt.plot(iterations, losses, label='Training Loss', color='deepskyblue') # 绘制曲线
    plt.yscale('log')

    # 添加图表元素，使其更美观、易读
    plt.title('Training Loss Curve (Log Scale)', fontsize=16) 
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Loss (Log Scale)', fontsize=12)
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.6) # which="both" 对主次刻度都显示网格
    
    # 步骤 4: 保存图像
    output_filename = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(output_filename)
    print(f"成功！Loss 曲线图已保存至: {output_filename}")
    
    # 如果你想在运行时直接看到图片，可以取消下面这行的注释
    # plt.show()

if __name__ == '__main__':
    
    log_file_path = 'loss_history.npy'
    save_directory = '.'
    
    plot_loss_curve(log_file_path, save_directory)