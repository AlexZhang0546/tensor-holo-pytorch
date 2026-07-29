"""
复数构造、FFT/IFT、频移（fftshift/ifftshift）等基础工具函数。
完全替代原 TensorFlow optics.py 中的 tf_compl_*, tf_fft2d, tf_ifft2d,
tf_fftshift2d, tf_ifftshift2d，并保持与 NCHW 数据格式一致。
"""

import torch
import numpy as np


# ----------------------------------------------------------------------
# 复数构造
# ----------------------------------------------------------------------
def compl_exp(phase: torch.Tensor, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    """
    根据相位（-π~π）构造单位复数：exp(j * phase)。
    替代 tf_compl_exp。
    
    Args:
        phase: 实数张量，任意形状，数值范围通常为 [-π, π]。
        dtype: 输出复数类型，默认 torch.complex64。
    
    Returns:
        复数张量，形状与 phase 相同。
    """
    # torch.polar 可一步生成复数，但要求振幅为 1
    ones = torch.ones_like(phase, dtype=phase.dtype if phase.is_floating_point() else torch.float32)
    return torch.polar(ones, phase).to(dtype)


def compl_val(amplitude: torch.Tensor, phase: torch.Tensor, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    """
    根据振幅和相位构造复数：amplitude * exp(j * phase)。
    替代 tf_compl_val。
    
    Args:
        amplitude: 振幅张量，非负实数。
        phase:     相位张量，-π~π。
        dtype:     输出复数类型。
    
    Returns:
        复数张量。
    """
    # 确保振幅和相位同设备、同数据类型（但可以是不同浮点型，polar 会自动转换）
    # torch.polar 需要两个相同 dtype 的张量，我们在内部转换
    amp = amplitude.to(dtype=torch.float32 if amplitude.dtype != torch.float64 else torch.float64)
    phs = phase.to(dtype=amp.dtype)
    return torch.polar(amp, phs).to(dtype)


# ----------------------------------------------------------------------
# 2D FFT / IFFT（NCHW 布局）
# ----------------------------------------------------------------------
def fft2d(x: torch.Tensor, dim: tuple = (-2, -1)) -> torch.Tensor:
    """
    对输入张量的最后两个维度进行 2D 快速傅里叶变换。
    替代 tf_fft2d，保持 NCHW 不变。
    
    Args:
        x:   输入张量，形状 (B, C, H, W)，可为实数或复数。
        dim: 进行 FFT 的维度，默认最后两维。
    
    Returns:
        复数张量，形状不变。
    """
    if not x.is_complex():
        x = torch.complex(x, torch.zeros_like(x))
    return torch.fft.fft2(x, dim=dim)


def ifft2d(x: torch.Tensor, dim: tuple = (-2, -1)) -> torch.Tensor:
    """
    对输入张量的最后两个维度进行 2D 逆快速傅里叶变换。
    替代 tf_ifft2d。
    
    Args:
        x: 复数张量。
        dim: 进行 IFFT 的维度。
    
    Returns:
        复数张量。
    """
    if not x.is_complex():
        x = torch.complex(x, torch.zeros_like(x))
    return torch.fft.ifft2(x, dim=dim)


# ----------------------------------------------------------------------
# fftshift / ifftshift for NCHW tensors (applied to last two dims)
# ----------------------------------------------------------------------
def fftshift2d(x: torch.Tensor, dim: tuple = (-2, -1)) -> torch.Tensor:
    """
    对最后两个空间维度进行 fftshift（将零频率移到图像中心）。
    替代 tf_fftshift2d。
    
    Args:
        x: 输入张量，形状 (..., H, W)。
        dim: 要进行移位的两个维度。
    
    Returns:
        移位后的张量。
    """
    # 对每个维度进行 roll，移动量为 (size + 1) // 2
    shifts = [(x.shape[d] + 1) // 2 for d in dim]
    # torch.roll 可以同时指定 shifts 和 dims
    return torch.roll(x, shifts=shifts, dims=dim)


def ifftshift2d(x: torch.Tensor, dim: tuple = (-2, -1)) -> torch.Tensor:
    """
    对最后两个空间维度进行 ifftshift（将零频率从图像中心移回角落）。
    替代 tf_ifftshift2d。
    
    Args:
        x: 输入张量。
        dim: 要进行移位的两个维度。
    
    Returns:
        移位后的张量。
    """
    shifts = [x.shape[d] // 2 for d in dim]   # ifftshift 的移动量是 size // 2
    return torch.roll(x, shifts=shifts, dims=dim)