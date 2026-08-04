# src/optics/complex_utils.py
"""
复数运算与二维傅里叶变换工具函数。
替代原 TensorFlow 版本中的：
    - tf_compl_exp
    - tf_compl_val
    - tf_fft2d
    - tf_ifft2d
    - tf_fftshift2d
    - tf_ifftshift2d

所有函数均接受形状为 (B, C, H, W) 的张量（NCHW），
并返回形状相同的张量。
"""

import torch
import numpy as np


def compl_exp(phase: torch.Tensor) -> torch.Tensor:
    """
    复数指数 e^{i * phase}。
    输入：实数张量 phase (任意形状)，相位以弧度为单位。
    输出：复数张量 (cos(phase) + i * sin(phase))。
    """
    return torch.complex(torch.cos(phase), torch.sin(phase))


def compl_val(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """
    由振幅和相位构造复数：amplitude * e^{i * phase}。
    输入：
        amplitude: 非负实数张量
        phase:     实数张量（弧度）
    输出：复数张量。
    """
    return amplitude * compl_exp(phase)


def fft2d(x: torch.Tensor) -> torch.Tensor:
    """
    对最后两维进行二维快速傅里叶变换。
    输入：
        x: (B, C, H, W) 实数或复数张量
    输出：
        复数频谱，形状不变，fft 在 (H, W) 维度执行。
    """
    if not x.is_complex():
        x = torch.complex(x, torch.zeros_like(x))
    return torch.fft.fft2(x, dim=(-2, -1))


def ifft2d(x: torch.Tensor) -> torch.Tensor:
    """
    对最后两维进行二维快速傅里叶逆变换（无缩放，即 "backward" 模式）。
    输入：
        x: (B, C, H, W) 复数张量
    输出：
        复数场，形状不变。
    """
    if not x.is_complex():
        x = torch.complex(x, torch.zeros_like(x))
    return torch.fft.ifft2(x, dim=(-2, -1))


def fftshift2d(x: torch.Tensor, input_shape=None) -> torch.Tensor:
    """
    将零频率分量移到频谱中心（fftshift），沿最后两维操作。
    等效于 tf_fftshift2d，input_shape 参数保留但不再使用（PyTorch 自动处理奇偶）。
    """
    return torch.fft.fftshift(x, dim=(-2, -1))


def ifftshift2d(x: torch.Tensor, input_shape=None) -> torch.Tensor:
    """
    fftshift 的逆操作，将中心零频移回角落。
    等效于 tf_ifftshift2d。
    """
    return torch.fft.ifftshift(x, dim=(-2, -1))