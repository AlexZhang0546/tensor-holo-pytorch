"""
通用评估指标：SSIM 和 PSNR。
替代原 TensorFlow 中的 tf.image.ssim / tf.image.psnr。
基于 PyTorch 实现，支持与原始代码相同的计算逻辑。
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = math.sqrt(2),
    kernel_size: int = 11,
    sigma: float = 1.5,
    C1: Optional[float] = None,
    C2: Optional[float] = None,
) -> torch.Tensor:
    if C1 is None:
        C1 = (0.01 * data_range) ** 2
    if C2 is None:
        C2 = (0.03 * data_range) ** 2

    # 统一为4D张量 (B,C,H,W)
    no_batch = img1.dim() == 3
    if no_batch:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    B, C, H, W = img1.shape
    device = img1.device

    # 生成高斯窗口
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
    coords -= (kernel_size - 1) / 2.0
    gauss = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    gauss /= gauss.sum()
    window_2d = gauss[:, None] * gauss[None, :]
    window = window_2d.expand(C, 1, kernel_size, kernel_size).to(img1.dtype)

    ssim_vals = []
    for i in range(B):
        a = img1[i:i+1]  # (1, C, H, W)
        b = img2[i:i+1]
        mu1 = F.conv2d(a, window, padding=kernel_size//2, groups=C)
        mu2 = F.conv2d(b, window, padding=kernel_size//2, groups=C)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(a * a, window, padding=kernel_size//2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(b * b, window, padding=kernel_size//2, groups=C) - mu2_sq
        sigma12 = F.conv2d(a * b, window, padding=kernel_size//2, groups=C) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_vals.append(ssim_map.mean())
    return torch.stack(ssim_vals).mean()


def compute_psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = math.sqrt(2),
) -> torch.Tensor:
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    B = img1.shape[0]
    psnr_vals = []
    for i in range(B):
        mse = F.mse_loss(img1[i], img2[i], reduction='mean')
        if mse == 0:
            psnr = torch.tensor(100.0, device=img1.device)
        else:
            psnr = 20.0 * torch.log10(torch.tensor(data_range, device=img1.device) / torch.sqrt(mse))
        psnr_vals.append(psnr)
    return torch.stack(psnr_vals).mean()
