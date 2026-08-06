"""
复数域损失函数
- complex_holo_loss : 直接比较复数预测与复数目标
"""

import torch
import torch.nn.functional as F


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
