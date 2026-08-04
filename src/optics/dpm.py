# src/optics/dpm.py
"""
双相位编码（Double Phase Method）相关函数。
包括：
    - 相位包裹 (_wrap_phase, _wrap_greater_than_max)
    - 高斯模糊 (_gaussian_blur_2d)
    - Maimone 双相位法 (dpm_maimone)
    - 带限双相位法 (bldpm)
    - 抗混叠双相位法 (aadpm)

所有函数均接受 (B, C, H, W) 形状的 NCHW 张量，返回形状一致的张量。
相位值除特殊说明外均为弧度制，范围通常在 [-π, π] 或经过包裹后的 [0, phs_max]。
"""

import torch
import torch.nn.functional as F
import numpy as np
from .complex_utils import (
    compl_exp,
    fft2d,
    ifft2d,
    fftshift2d,
    ifftshift2d,
)


# ----------------------------------------------------------------------
# 内部辅助函数
# ----------------------------------------------------------------------
def _wrap_greater_than_max(phs: torch.Tensor, phs_max: torch.Tensor) -> torch.Tensor:
    """
    将相位包裹到 [0, phs_max] 范围内。
    输入 phs 未归一化（弧度），phs_max 为各通道的最大相位（形状与 phs 广播兼容）。
    """
    # 将中心移至 phs_max/2
    phs = phs + phs_max / 2.0
    # 包裹：小于 0 则加 2π，大于 phs_max 则减 2π
    phs = torch.where(phs < 0, phs + 2.0 * np.pi, phs)
    phs = torch.where(phs > phs_max, phs - 2.0 * np.pi, phs)
    return phs


def _wrap_phase(
    phs_only: torch.Tensor,
    phs_max: list = None,
    adaptive_phs_shift: bool = False,
) -> torch.Tensor:
    """
    对相位图进行包裹，使其不超过指定的最大相位。
    
    参数:
        phs_only: 形状 (B, C, H, W)，相位值（弧度，未归一化）。
        phs_max:  长度等于 C 的列表，每通道最大相位（弧度），若为 None 则不包裹。
        adaptive_phs_shift: 是否自适应地中心化相位（当相位范围较小时直接平移到中间）。
    
    返回:
        包裹后的相位，形状不变。
    """
    if phs_max is None:
        return phs_only

    phs_max_tensor = torch.tensor(phs_max, device=phs_only.device, dtype=phs_only.dtype)
    # 重塑为 (1, C, 1, 1) 便于广播
    phs_max_4d = phs_max_tensor.view(1, -1, 1, 1)

    if not adaptive_phs_shift:
        return _wrap_greater_than_max(phs_only, phs_max_4d)

    # 逐通道自适应处理
    wrapped_channels = []
    for c in range(phs_only.shape[1]):
        chan = phs_only[:, c : c + 1, :, :]  # (B, 1, H, W)
        pmax = phs_max[c]
        chan_max = chan.max()
        chan_min = chan.min()
        if (chan_max - chan_min) <= pmax:
            # 范围小于最大相位，直接平移到 [0, pmax] 中央
            chan = chan + (pmax - chan_min - chan_max) / 2.0
        else:
            chan = _wrap_greater_than_max(chan, phs_max_4d[:, c : c + 1, :, :])
        wrapped_channels.append(chan)
    return torch.cat(wrapped_channels, dim=1)


def _gaussian_blur_2d(
    x: torch.Tensor,
    sigma: float,
    kernel_width: int,
) -> torch.Tensor:
    """
    对复数场进行二维高斯模糊（分别对实部与虚部应用相同的深度可分离卷积）。
    
    参数:
        x: 复数张量 (B, C, H, W)
        sigma: 高斯核的标准差
        kernel_width: 卷积核宽度（奇数）
    
    返回:
        模糊后的复数张量
    """
    if sigma <= 0.0:
        return x

    # 生成 1D 高斯
    coords = torch.arange(kernel_width, dtype=x.real.dtype, device=x.device)
    coords -= (kernel_width - 1) / 2.0
    gauss = torch.exp(-(coords**2) / (2.0 * sigma**2))
    gauss /= gauss.sum()
    # 2D 高斯核
    kernel_2d = gauss[:, None] * gauss[None, :]  # (kw, kw)
    # 扩展为 depthwise 卷积核形状 (C, 1, kw, kw)
    C = x.shape[1]
    kernel = kernel_2d.expand(C, 1, kernel_width, kernel_width)

    # 分组卷积（depthwise）
    real_blurred = F.conv2d(x.real, kernel, padding=kernel_width // 2, groups=C)
    imag_blurred = F.conv2d(x.imag, kernel, padding=kernel_width // 2, groups=C)
    return torch.complex(real_blurred, imag_blurred)


def _checkerboard_double_phase(
    amp: torch.Tensor,
    phs_zero_mean: torch.Tensor,
) -> torch.Tensor:
    """
    将振幅与零均值相位转换为双相位编码的棋盘格图案，并通过 pixel shuffle 恢复全分辨率。
    此函数假设 amp 已经归一化到 [0,1]。
    
    参数:
        amp:           (B, C, H, W) 归一化振幅
        phs_zero_mean: (B, C, H, W) 零均值相位（弧度）
    
    返回:
        phs_only: (B, C, H, W) 双相位编码后的相位图（弧度）
    """
    phs_offset = torch.acos(amp)                     # 反余弦
    phs_low = phs_zero_mean - phs_offset
    phs_high = phs_zero_mean + phs_offset

    # 棋盘格排列（取偶/奇行和列）
    phs_1_1 = phs_low[:, :, 0::2, 0::2]
    phs_1_2 = phs_high[:, :, 0::2, 1::2]
    phs_2_1 = phs_high[:, :, 1::2, 0::2]
    phs_2_2 = phs_low[:, :, 1::2, 1::2]

    # 在通道维拼接：C*4 通道
    phs_cat = torch.cat([phs_1_1, phs_1_2, phs_2_1, phs_2_2], dim=1)  # (B, 4C, H/2, W/2)

    # pixel shuffle 恢复全分辨率 (B, C, H, W)
    phs_only = F.pixel_shuffle(phs_cat, upscale_factor=2)
    return phs_only


# ----------------------------------------------------------------------
# 公开 DPM 函数
# ----------------------------------------------------------------------
def dpm_maimone(
    cpx: torch.Tensor,
    propagator=None,
    depth_shift: float = 0.0,
    adaptive_phs_shift: bool = False,
    batch: int = 1,
    num_channels: int = 3,
    res_h: int = 384,
    res_w: int = 384,
    axis: int = 2,
    phs_max: list = None,
    amp_max: float = None,
    clamp: bool = False,
    normalize: bool = True,
    wavelength: list = None,
) -> (torch.Tensor, torch.Tensor):
    """
    Maimone 等人 2017 的双相位方法（先对行/列降采样，再棋盘格排列）。
    
    参数:
        cpx:                 复数场 (B, C, H, W)
        propagator:          传播算子 (接受 cpx, distance 返回 cpx)
        depth_shift:         传播距离（mm）
        adaptive_phs_shift:  是否自适应相位中心化
        batch, num_channels, res_h, res_w: 保留参数，用于兼容
        axis:                降采样维度，2 表示行（高度），3 表示列（宽度）
        phs_max:             最大相位列表（弧度），如 [2π, 2π, 2π]
        amp_max:             用于归一化的最大振幅，若为 None 则自动计算
        clamp:               是否钳制振幅 ≤ 1
        normalize:           是否将输出相位归一化到 [0,1]
        wavelength:          波长列表（mm），用于传播补偿（若 depth_shift != 0）
    
    返回:
        phs_only:  归一化或未归一化的相位图 (B, C, H, W)
        amp_max:   使用的最大振幅值
    """
    # ---- 传播与相位补偿 ----
    if propagator is not None and depth_shift != 0:
        wlen = torch.tensor(wavelength, device=cpx.device).view(1, -1, 1, 1)
        cpx = propagator(cpx, depth_shift) * compl_exp(-2 * np.pi * depth_shift / wlen)

    amp = torch.abs(cpx)
    phs = torch.angle(cpx)

    # ---- 振幅归一化 ----
    if amp_max is None:
        amp_max = amp.max() + 1e-6
    amp = amp / amp_max
    if clamp:
        amp = torch.clamp(amp, max=1.0 - 1e-6)

    # ---- 零均值相位 ----
    phs_zero_mean = phs - phs.mean(dim=(2, 3), keepdim=True)

    # ---- 根据 axis 降采样 ----
    if axis == 3:   # 降低列（宽度）
        amp = amp[:, :, :, 0::2]
        phs_zero_mean = phs_zero_mean[:, :, :, 0::2]
    elif axis == 2: # 降低行（高度）
        amp = amp[:, :, 0::2, :]
        phs_zero_mean = phs_zero_mean[:, :, 0::2, :]
    else:
        raise ValueError("axis must be 2 (rows) or 3 (columns)")

    # ---- 双相位计算 ----
    phs_offset = torch.acos(amp)
    phs_low = phs_zero_mean - phs_offset
    phs_high = phs_zero_mean + phs_offset

    # ---- 棋盘格排列（依据 axis） ----
    if axis == 3:
        phs_1_1 = phs_low[:, :, 0::2, :]
        phs_1_2 = phs_high[:, :, 0::2, :]
        phs_2_1 = phs_high[:, :, 1::2, :]
        phs_2_2 = phs_low[:, :, 1::2, :]
    else:  # axis == 2
        phs_1_1 = phs_low[:, :, :, 0::2]
        phs_1_2 = phs_high[:, :, :, 0::2]
        phs_2_1 = phs_high[:, :, :, 1::2]
        phs_2_2 = phs_low[:, :, :, 1::2]

    phs_cat = torch.cat([phs_1_1, phs_1_2, phs_2_1, phs_2_2], dim=1)  # (B, 4C, H', W')
    phs_only = F.pixel_shuffle(phs_cat, upscale_factor=2)            # 恢复至 (B, C, H, W)

    # ---- 相位包裹与归一化 ----
    if phs_max is not None:
        phs_only = _wrap_phase(phs_only, phs_max, adaptive_phs_shift)

    if normalize and phs_max is not None:
        phs_max_tensor = torch.tensor(phs_max, device=phs_only.device, dtype=phs_only.dtype)
        phs_only = phs_only / phs_max_tensor.view(1, -1, 1, 1)

    return phs_only, amp_max


def bldpm(
    cpx: torch.Tensor,
    propagator=None,
    depth_shift: float = 0.0,
    adaptive_phs_shift: bool = False,
    batch: int = 1,
    num_channels: int = 3,
    res_h: int = 384,
    res_w: int = 384,
    k: float = 0.5,
    phs_max: list = None,
    amp_max: float = None,
    clamp: bool = False,
    normalize: bool = True,
    wavelength: list = None,
) -> (torch.Tensor, torch.Tensor):
    """
    带限双相位方法（Band‑limited Double Phase Method, Sui et al. 2021）。
    在频域中施加锥形滤波器以抑制混叠。
    """
    # ---- 构建空间频率掩模 ----
    square_filter = True
    device = cpx.device
    dtype = cpx.real.dtype
    y = torch.arange(-(res_h // 2), res_h // 2, device=device, dtype=dtype)
    x = torch.arange(-(res_w // 2), res_w // 2, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # (H, W)

    # 堆叠三个通道
    xx = xx.unsqueeze(0).expand(num_channels, -1, -1)  # (C, H, W)
    yy = yy.unsqueeze(0).expand(num_channels, -1, -1)

    if square_filter:
        side_min = min(res_h, res_w)
        xx = xx / side_min
        yy = yy / side_min
    else:
        xx = xx / res_w
        yy = yy / res_h

    tan_pi_alpha_u = torch.tan(yy * np.pi)
    tan_pi_alpha_mu = torch.tan(xx * np.pi)
    mask = (torch.abs(tan_pi_alpha_u * tan_pi_alpha_mu) <= k)

    if square_filter:
        mask_two = (torch.abs(xx) <= 0.5) & (torch.abs(yy) <= 0.5)
        mask = mask & mask_two

    mask = mask.unsqueeze(0).to(dtype)  # (1, C, H, W)

    # ---- 传播 ----
    if propagator is not None and depth_shift != 0:
        wlen = torch.tensor(wavelength, device=device).view(1, -1, 1, 1)
        cpx = propagator(cpx, depth_shift) * compl_exp(-2 * np.pi * depth_shift / wlen)

    # ---- 频域滤波 ----
    cpx_fft = fftshift2d(fft2d(cpx)) * mask
    cpx = ifft2d(ifftshift2d(cpx_fft))

    # ---- 振幅与相位提取 ----
    amp = torch.abs(cpx)
    phs = torch.angle(cpx)

    if amp_max is None:
        amp_max = amp.max() + 1e-6
    amp = amp / amp_max
    if clamp:
        amp = torch.clamp(amp, max=1.0 - 1e-6)

    phs_zero_mean = phs - phs.mean(dim=(2, 3), keepdim=True)

    # ---- 双相位棋盘格（全图，无预降采样） ----
    phs_only = _checkerboard_double_phase(amp, phs_zero_mean)

    # ---- 包裹与归一化 ----
    if phs_max is not None:
        phs_only = _wrap_phase(phs_only, phs_max, adaptive_phs_shift)

    if normalize and phs_max is not None:
        phs_max_tensor = torch.tensor(phs_max, device=device, dtype=dtype)
        phs_only = phs_only / phs_max_tensor.view(1, -1, 1, 1)

    return phs_only, amp_max


def aadpm(
    cpx: torch.Tensor,
    propagator=None,
    depth_shift: float = 0.0,
    adaptive_phs_shift: bool = False,
    batch: int = 1,
    num_channels: int = 3,
    res_h: int = 384,
    res_w: int = 384,
    sigma: float = 0.5,
    kernel_width: int = 5,
    phs_max: list = None,
    amp_max: float = None,
    clamp: bool = False,
    normalize: bool = True,
    wavelength: list = None,
) -> (torch.Tensor, torch.Tensor):
    """
    抗混叠双相位方法（Anti‑aliasing Double Phase Method）。
    在双相位编码前对复数场施加高斯模糊以抑制混叠。
    """
    # ---- 传播 ----
    if propagator is not None and depth_shift != 0:
        wlen = torch.tensor(wavelength, device=cpx.device).view(1, -1, 1, 1)
        cpx = propagator(cpx, depth_shift) * compl_exp(-2 * np.pi * depth_shift / wlen)

    # ---- 预模糊 ----
    cpx = _gaussian_blur_2d(cpx, sigma, kernel_width)

    # ---- 振幅与相位 ----
    amp = torch.abs(cpx)
    phs = torch.angle(cpx)

    if amp_max is None:
        amp_max = amp.max() + 1e-6
    amp = amp / amp_max
    if clamp:
        amp = torch.clamp(amp, max=1.0 - 1e-6)

    phs_zero_mean = phs - phs.mean(dim=(2, 3), keepdim=True)

    # ---- 双相位棋盘格 ----
    phs_only = _checkerboard_double_phase(amp, phs_zero_mean)

    # ---- 包裹与归一化 ----
    if phs_max is not None:
        phs_only = _wrap_phase(phs_only, phs_max, adaptive_phs_shift)

    if normalize and phs_max is not None:
        phs_max_tensor = torch.tensor(phs_max, device=cpx.device, dtype=cpx.real.dtype)
        phs_only = phs_only / phs_max_tensor.view(1, -1, 1, 1)

    return phs_only, amp_max
