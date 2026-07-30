"""
复数域损失函数
- complex_holo_loss : 直接比较复数预测与复数目标
- complex_ddpm_phase_loss : 从复数场提取相位，计算相位分布正则项
"""

import torch
import torch.nn.functional as F
import numpy as np


def complex_holo_loss(
    pred_complex: torch.Tensor,
    target_complex: torch.Tensor,
    loss_type: str = 'l1',
    method: str = 'magnitude_phase'   # 'magnitude_phase' or 'complex_diff'
) -> torch.Tensor:
    """
    复数全息图损失，替代原 compute_holo_loss。

    Args:
        pred_complex: 预测复数场 (B, 3, H, W)，复数类型。
        target_complex: 目标复数场 (B, 3, H, W)，复数类型。
        loss_type: 底层像素损失类型，'l1' 或 'l2'。
        method:
            - 'magnitude_phase': 分别计算实部和虚部的损失，等价于振幅-相位损失。
            - 'complex_diff': 基于复数乘积的全局相位不变损失（与原 cos/sin 分量等价）。

    Returns:
        loss: 标量损失值。
    """
    if loss_type == 'l1':
        loss_fn = F.l1_loss
    elif loss_type == 'l2':
        loss_fn = F.mse_loss
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Use 'l1' or 'l2'.")

    if method == 'magnitude_phase':
        # 直接对实部、虚部计算 L1/L2，等价于独立振幅-相位损失
        loss_real = loss_fn(pred_complex.real, target_complex.real)
        loss_imag = loss_fn(pred_complex.imag, target_complex.imag)
        return loss_real + loss_imag

    elif method == 'complex_diff':
        # 计算 target * conj(pred)，其实部与 |target| 比较，虚部与 0 比较
        # 该方式自动去除了全局相位偏移
        diff = target_complex * torch.conj(pred_complex)
        target_abs = torch.abs(target_complex)
        loss_cos = loss_fn(diff.real, target_abs)          # 对齐振幅
        loss_sin = loss_fn(diff.imag, torch.zeros_like(diff.imag))  # 相位差应为0
        return loss_cos + loss_sin

    else:
        raise ValueError(f"Unsupported method: {method}. Use 'magnitude_phase' or 'complex_diff'.")


def complex_ddpm_phase_loss(
    complex_field: torch.Tensor,
    pad: int = 0,
    res_h: int = None,
    res_w: int = None
):
    """
    DDPM 相位正则损失（复数版本）。
    直接从复数场提取相位，计算空间标准差和均值偏移量。

    Args:
        complex_field: DDPM 校正后的复数场 (B, 3, H_pad, W_pad)。
        pad: 边缘填充量，裁剪后计算。
        res_h, res_w: 原始（无填充）的高度和宽度。

    Returns:
        std_loss:  各通道空间标准差的均值。
        mean_loss: 各通道空间均值偏离 0.5 的绝对值的平均（归一化到 [0,1] 的相位）。
    """
    # 从复数场提取相位，并归一化到 [0, 1]
    phs = torch.angle(complex_field) / (2.0 * np.pi) + 0.5  # (B, C, H, W)

    if pad > 0:
        if res_h is None:
            res_h = phs.shape[2] - 2 * pad
        if res_w is None:
            res_w = phs.shape[3] - 2 * pad
        phs = phs[:, :, pad:pad + res_h, pad:pad + res_w]

    # 空间维度的标准差
    std_per_channel = phs.std(dim=(2, 3))   # (B, C)
    std_loss = std_per_channel.mean()

    # 空间均值偏离 0.5 的程度
    mean_per_channel = phs.mean(dim=(2, 3))
    mean_loss = (mean_per_channel - 0.5).abs().mean()

    return std_loss, mean_loss