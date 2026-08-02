"""
物理光圈滤波（替代 tf_filter_phs_only）
模拟 SLM 或光学系统的低通特性：将相位全息图转为复数场，在频域乘以圆形孔径掩模，
再传播回原深度。

修复说明：
  适配 dpm.py 返回的逐样本/逐通道振幅最大值 amp_max，其形状为 (B, C, 1, 1) 的张量，
  避免 amp_max.item() 引发的尺寸错误；同时兼容旧版的标量/None 输入。
"""

import torch
import torch.nn.functional as F
import numpy as np
from .complex_utils import compl_val, compl_exp, fft2d, ifft2d, fftshift2d, ifftshift2d


def circ_filter(batch: int, num_channels: int, res_h: int, res_w: int,
                filter_radius: float, device=None, dtype=torch.complex64) -> torch.Tensor:
    """
    生成圆形低通滤波器掩模，值为 1 的区域内允许频率通过。
    
    Args:
        batch, num_channels, res_h, res_w: 输出掩模的形状参数。
        filter_radius: 频域半径（像素单位），超出该半径的值为 0。
    
    Returns:
        掩模张量，形状 (batch, num_channels, H, W)，复数类型（实部为掩模值）。
    """
    y, x = torch.meshgrid(
        torch.linspace(-(res_h - 1) / 2, (res_h - 1) / 2, res_h, device=device),
        torch.linspace(-(res_w - 1) / 2, (res_w - 1) / 2, res_w, device=device),
        indexing='ij'
    )
    mask = (x**2 + y**2) <= filter_radius**2
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch, num_channels, -1, -1)
    return mask.to(dtype)


def filter_phs_only(phs_only: torch.Tensor,
                    unnormalize_input: bool = False,
                    normalize_output: bool = True,
                    propagator=None,
                    depth_shift: float = 0.0,
                    batch: int = 2,
                    num_channels: int = 3,
                    res_h: int = 384,
                    res_w: int = 384,
                    radius: int = None,
                    phs_max: list = None,
                    amp_max = None,
                    wavelength: list = None) -> tuple:
    """
    模拟物理孔径的低通滤波。
    
    Args:
        phs_only: 相位图，形状 (B, C, H, W)。默认已归一化到 [0, 1]（若 unnormalize_input=True 则乘以 phs_max）。
        unnormalize_input: 是否先将输入相位反归一化到弧度。
        normalize_output:  是否将输出相位重新归一化到 [0, 1]。
        propagator: 传播算子（用于将滤波后的场传回 depth_shift）。
        depth_shift: 反向传播距离。
        radius: 频域孔径半径；若为 None，则取 min(H, W)/2。
        phs_max: 每个通道的最大相位弧度（用于反归一化/归一化）。
        amp_max: 用于构造复数场的振幅。
                 可以是标量、形状 (B, C, 1, 1) 的张量（来自 dpm 模块），
                 或 None（默认使用全 1）。
    
    Returns:
        amp_filtered: 滤波后的振幅 (B, C, H, W)
        phs_filtered: 滤波后的相位，若 normalize_output 则为 [0, 1]，否则为弧度。
    """
    if radius is None:
        radius = min(res_h, res_w) // 2

    # ---- 反归一化输入 ----
    if unnormalize_input and phs_max is not None:
        phs_max_tensor = torch.tensor(phs_max, dtype=phs_only.dtype, device=phs_only.device).view(1, -1, 1, 1)
        # 注意：原代码中 unnormalize_input 时执行 (phs_only - 0.5) * phs_max，这里假设输入已居中在 0.5 附近
        phs_only = (phs_only - 0.5) * phs_max_tensor

    # ---- 构造复数场：振幅由 amp_max 决定 ----
    if amp_max is None:
        amp_tensor = torch.ones_like(phs_only)
    elif torch.is_tensor(amp_max):
        # amp_max 形状 (B, C, 1, 1) 或可广播的形状，扩展至 (B, C, H, W)
        amp_tensor = amp_max.expand_as(phs_only)
    else:
        # 标量
        amp_tensor = torch.full_like(phs_only, amp_max)

    cpx = compl_val(amp_tensor, phs_only)

    # ---- 频域滤波 ----
    cpx_fft = fftshift2d(fft2d(cpx))
    mask = circ_filter(batch, num_channels, res_h, res_w, radius,
                       device=cpx.device, dtype=cpx.dtype)
    cpx_fft_filtered = cpx_fft * mask
    cpx_filtered = ifft2d(ifftshift2d(cpx_fft_filtered))

    # ---- 反向传播 ----
    if depth_shift != 0.0:
        if propagator is None:
            raise ValueError("propagator required for depth shift")
        tf_wavelength = torch.tensor(wavelength, dtype=cpx_filtered.real.dtype, device=cpx_filtered.device).view(1, -1, 1, 1)
        cpx_filtered = propagator(cpx_filtered, depth_shift) * compl_exp(2 * np.pi * depth_shift / tf_wavelength)

    # ---- 提取振幅和相位 ----
    amp_filtered = torch.abs(cpx_filtered)
    phs_filtered = torch.angle(cpx_filtered)

    if normalize_output:
        # 归一化到 [0, 1]
        phs_filtered = phs_filtered / (2.0 * np.pi) + 0.5

    return amp_filtered, phs_filtered