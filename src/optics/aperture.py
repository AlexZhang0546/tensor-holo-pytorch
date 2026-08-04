# src/optics/aperture.py
"""
物理孔径模拟与相位全息图滤波（PyTorch 版本）。
替代原 TensorFlow 中的 tf_filter_phs_only 和 np_circ_filter。
所有运算均以复数张量 (B, C, H, W) 为核心，支持与传播算子组合。
"""

import torch
import numpy as np
from .complex_utils import compl_val, compl_exp, fft2d, ifft2d, fftshift2d, ifftshift2d


def circ_filter(
    batch: int,
    num_channels: int,
    res_h: int,
    res_w: int,
    radius: int = None
) -> torch.Tensor:
    """
    生成圆形低通滤波器（频域掩模）。

    Args:
        batch: 批量大小。
        num_channels: 通道数（通常为 3，对应 RGB 三色光场）。
        res_h: 空间高度。
        res_w: 空间宽度。
        radius: 滤波器半径（像素），若为 None 则使用 min(res_h, res_w) // 2。

    Returns:
        滤波器张量，形状 (batch, num_channels, res_h, res_w)，元素值为 0.0 或 1.0。
    """
    if radius is None:
        radius = min(res_h, res_w) // 2

    # 构建归一化坐标
    y, x = torch.meshgrid(
        torch.linspace(-(res_h - 1) / 2, (res_h - 1) / 2, res_h),
        torch.linspace(-(res_w - 1) / 2, (res_w - 1) / 2, res_w),
        indexing='ij'
    )
    dist = torch.sqrt(x**2 + y**2)
    mask = (dist <= radius).float()  # (H, W)

    # 扩展到 (batch, num_channels, H, W)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch, num_channels, -1, -1)
    return mask


def filter_phs_only(
    phs_only: torch.Tensor,
    unnormalize_input: bool = False,
    normalize_output: bool = True,
    propagator: object = None,
    depth_shift: float = 0.0,
    batch: int = 2,
    num_channels: int = 3,
    res_h: int = 384,
    res_w: int = 384,
    radius: int = None,
    phs_max: list = None,
    amp_max: float = 1.0,
    wavelength: list = None
):
    """
    模拟物理孔径对双相位编码全息图的低通滤波效应。

    该函数将归一化或未归一化的相位图转换为纯相位复数场，在频域施加圆形滤波器，
    并可选择性地将滤波后的场传播回参考平面。

    Args:
        phs_only: 相位图，形状 (B, C, H, W)。
                  若 unnormalize_input=True，则值应在 [0, 1] 之间；
                  否则预期为弧度值。
        unnormalize_input: 是否对输入相位进行反归一化。
        normalize_output: 是否将输出相位归一化到 [0, 1]。
        propagator: 传播算子对象，需支持 __call__(cpx, distance)。
        depth_shift: 传播距离（mm）。若不为 0，将滤波后的场正向传播 depth_shift，
                     再乘以补偿因子 exp(j·2π·depth_shift/λ)。
        batch, num_channels, res_h, res_w: 保留参数，用于构建滤波器及反归一化。
        radius: 滤波器半径（像素），None 则自动取 min(H,W)//2。
        phs_max: 各通道最大相位（弧度），仅当 unnormalize_input=True 时使用。
        amp_max: 纯相位全息图的振幅（通常为 SLM 最大振幅）。
        wavelength: 波长列表（mm），用于传播补偿及反归一化。

    Returns:
        amp_filtered: 滤波后振幅图 (B, C, H, W)。
        phs_filtered: 滤波后相位图 (B, C, H, W)，根据 normalize_output 决定是否归一化。
    """
    if radius is None:
        radius = min(res_h, res_w) // 2

    # ---------- 1. 输入相位反归一化 ----------
    if unnormalize_input:
        if phs_max is None:
            raise ValueError("phs_max must be provided when unnormalize_input=True")
        # 将归一化相位 [0,1] 映射到 [-0.5*phs_max, 0.5*phs_max] 然后平移到 [0, phs_max]
        # 注意：原 TF 中 phs_only 在 unnormalize 之前是 [0,1]，而 unnormalize 对应减法 0.5 再乘 phs_max
        # 但原代码实际是 (phs_only - 0.5) * phs_max，使得范围变回 [-phs_max/2, phs_max/2]？不对，
        # 原 TF 中 phs_max 是各通道最大相位（如 2π），phs_only 是 [0,1]，
        # unnormalize 后变为 (phs_only-0.5)*phs_max，范围 [-phs_max/2, phs_max/2]。
        # 之后构造复数场时直接用该值作为相位，即复数场的相位范围是 [-phs_max/2, phs_max/2]。
        phs_max_tensor = torch.tensor(phs_max, device=phs_only.device, dtype=phs_only.dtype)
        phs = (phs_only - 0.5) * phs_max_tensor.view(1, -1, 1, 1)
    else:
        phs = phs_only  # 假设已经是弧度值

    # ---------- 2. 构造纯相位复数场 ----------
    # 振幅均匀为 amp_max
    amplitude = torch.full_like(phs, amp_max.item() if isinstance(amp_max, torch.Tensor) else amp_max)
    field = compl_val(amplitude, phs)

    # ---------- 3. 频域低通滤波 ----------
    field_fft = fftshift2d(fft2d(field))
    mask = circ_filter(batch, num_channels, res_h, res_w, radius).to(field_fft.device).to(field_fft.dtype)
    field_fft_filtered = field_fft * mask
    field_filtered = ifft2d(ifftshift2d(field_fft_filtered))

    # ---------- 4. 可选反向传播 ----------
    if depth_shift != 0.0:
        if propagator is None:
            raise ValueError("propagator must be provided when depth_shift != 0")
        if wavelength is None:
            raise ValueError("wavelength must be provided when depth_shift != 0")
        # 正向传播 depth_shift 距离
        field_filtered = propagator(field_filtered, depth_shift)
        # 额外相位补偿 exp(j * 2π * depth_shift / λ)
        wlen = torch.tensor(wavelength, device=field.device).view(1, -1, 1, 1)
        compensation = compl_exp(2 * np.pi * depth_shift / wlen)
        field_filtered = field_filtered * compensation

    # ---------- 5. 提取振幅与相位 ----------
    amp_filtered = torch.abs(field_filtered)
    phs_filtered = torch.angle(field_filtered)

    # ---------- 6. 输出相位归一化 ----------
    if normalize_output:
        phs_filtered = phs_filtered / (2.0 * np.pi) + 0.5

    return amp_filtered, phs_filtered