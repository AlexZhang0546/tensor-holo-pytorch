# src/losses/ddpm_loss.py
"""
DDPM 训练阶段的相位正则损失（复数版本）。
直接从复数光场提取相位，计算空间标准差和均值偏移量，
与原 TensorFlow 中 stage2 的 mean_loss / std_loss 对应。
"""

import torch
import numpy as np


def compute_ddpm_phase_loss(
    complex_field: torch.Tensor,
    pad: int = 0,
    res_h: int = None,
    res_w: int = None,
):
    """
    计算 DDPM 网络的相位分布正则项。

    Args:
        complex_field: DDPM 校正后的复数场 (B, 3, H_pad, W_pad)。
        pad:           边缘填充像素数，用于裁剪有效区域。
        res_h:         原始（无填充）高度，若为 None 则自动从输入尺寸扣除 2*pad。
        res_w:         原始宽度，同上。

    Returns:
        std_loss:  各通道空间标准差的均值（标量）。
        mean_loss: 各通道空间均值偏离 0.5 的绝对值的平均（标量）。
    """
    # 从复数场提取相位，并归一化到 [0, 1]
    phs = torch.angle(complex_field) / (2.0 * np.pi) + 0.5  # (B, C, H, W)

    # 如果存在填充，裁剪中心区域
    if pad > 0:
        if res_h is None:
            res_h = phs.shape[2] - 2 * pad
        if res_w is None:
            res_w = phs.shape[3] - 2 * pad
        phs = phs[:, :, pad:pad + res_h, pad:pad + res_w]

    # 空间维度的标准差（对每个样本、每个通道独立计算，然后取均值）
    std_per_channel = phs.std(dim=(2, 3))   # (B, C)
    std_loss = std_per_channel.mean()

    # 空间均值偏离 0.5 的程度
    mean_per_channel = phs.mean(dim=(2, 3))
    mean_loss = (mean_per_channel - 0.5).abs().mean()

    return std_loss, mean_loss