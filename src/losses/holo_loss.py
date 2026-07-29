"""
全息图振幅‑相位损失（holo_loss）。
对应原 TF 代码 _get_loss 中计算 holo_loss 的部分，
包括相位差计算、全局相位偏移去除，以及 L1/L2 损失。
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_holo_loss(
    amp_out: torch.Tensor,
    phs_out: torch.Tensor,
    amp_gt: torch.Tensor,
    phs_gt: torch.Tensor,
    pad: int = 0,
    loss_type: str = 'l1'
) -> torch.Tensor:
    """
    计算全息图振幅‑相位损失。

    Args:
        amp_out: 预测振幅，形状 (B, 3, H, W)，取值范围 [0, √2]。
        phs_out: 预测相位，形状 (B, 3, H, W)，取值范围 [0, 1]。
        amp_gt:  目标振幅，形状同上。
        phs_gt:  目标相位，形状同上。
        pad:     边缘裁剪像素数（0 表示不裁剪）。
        loss_type: 损失类型，'l1' 或 'l2'。

    Returns:
        holo_loss: 标量损失值。
    """
    # 选择损失函数
    if loss_type == 'l1':
        loss_fn = F.l1_loss
    elif loss_type == 'l2':
        loss_fn = F.mse_loss
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Use 'l1' or 'l2'.")

    # 1. 裁剪 pad 区域（保留中心有效区）
    if pad > 0:
        amp_out = amp_out[:, :, pad:-pad, pad:-pad]
        phs_out = phs_out[:, :, pad:-pad, pad:-pad]
        amp_gt  = amp_gt[:, :, pad:-pad, pad:-pad]
        phs_gt  = phs_gt[:, :, pad:-pad, pad:-pad]

    # 2. 相位从 [0, 1] 还原到 [-π, π]
    phs_out_scaled = (phs_out - 0.5) * 2.0 * np.pi
    phs_gt_scaled  = (phs_gt  - 0.5) * 2.0 * np.pi

    # 3. 计算包裹的相位差（保证在 [-π, π] 内）
    diff = phs_gt_scaled - phs_out_scaled
    phs_diff = torch.atan2(torch.sin(diff), torch.cos(diff))

    # 4. 去除全局相位偏移（每个颜色通道独立，在空间维度求均值并减去）
    global_phase = phs_diff.mean(dim=(2, 3), keepdim=True)   # (B, C, 1, 1)
    phs_diff = phs_diff - global_phase

    # 5. 计算 cos 和 sin 分量损失
    loss_cos = loss_fn(amp_gt * torch.cos(phs_diff), amp_out)
    loss_sin = loss_fn(amp_gt * torch.sin(phs_diff),
                       torch.zeros_like(amp_gt))

    holo_loss = loss_cos + loss_sin
    return holo_loss