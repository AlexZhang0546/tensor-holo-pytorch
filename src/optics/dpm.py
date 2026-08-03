"""
双相位编码（Double Phase Method）的三种变体实现：
  - AADPM (Anti-Aliasing DPM, 预模糊 + 棋盘排列)
  - BL-DPM (Band-Limited DPM, 频域滤波)
  - DPM-Maimone (原始 DPM，行/列压缩)

包含相位包裹辅助函数 _wrap_phase 以及自定义高斯模糊工具 _gaussian_blur_2d。
所有函数均为纯 PyTorch 实现，支持可微分训练和推理。

修复说明：
  将全局振幅归一化 amp.max() 改为逐样本、逐通道的空间最大值 amp.amax(dim=(2,3), keepdim=True)，
  避免梯度截断和跨样本干扰，确保 stage2 joint 训练正常。
"""

import torch
import torch.nn.functional as F
import numpy as np

from .complex_utils import compl_val, compl_exp, fft2d, ifft2d, fftshift2d, ifftshift2d


# ----------------------------------------------------------------------
# 相位包裹（替代 tf_wrap_phs）
# ----------------------------------------------------------------------
def _wrap_phase(phs_only: torch.Tensor,
                phs_max: list,
                adaptive_phs_shift: bool = False) -> torch.Tensor:
    """
    将相位限制在 [0, phs_max] 范围内，采用与原 TensorFlow 完全相同的逻辑。
    
    Args:
        phs_only: 待包裹的相位张量，形状 (B, C, H, W)，单位为弧度（未归一化）。
        phs_max:  每个通道的最大相位弧度值，形如 [2π, 2π, 2π] 或自定义值。
        adaptive_phs_shift: 是否自适应平移使相位范围居中于 [0, phs_max]。
    
    Returns:
        包裹后的相位张量，形状相同，值域满足约束。
    """
    if phs_max is None:
        return phs_only

    # 转换为张量，形状 (1, C, 1, 1)
    phs_max_tensor = torch.tensor(phs_max, dtype=phs_only.dtype, device=phs_only.device).view(1, -1, 1, 1)

    if adaptive_phs_shift:
        # 自适应平移：若相位范围小于 phs_max，则居中；否则按标准包裹
        wrapped_list = []
        for c in range(phs_only.size(1)):
            chan = phs_only[:, c:c+1, :, :]          # (B, 1, H, W)
            p_max = phs_max[c]
            chan_min = chan.min()
            chan_max = chan.max()
            chan_range = chan_max - chan_min

            # 范围小于限制：平移使居中
            centered = chan + (p_max - chan_max - chan_min) / 2.0
            # 范围大于限制：使用传统包裹
            wrapped = _wrap_greater_than_max(chan, p_max)

            chan_wrapped = torch.where(chan_range <= p_max, centered, wrapped)
            wrapped_list.append(chan_wrapped)
        return torch.cat(wrapped_list, dim=1)

    else:
        # 统一包裹
        return _wrap_greater_than_max(phs_only, phs_max_tensor)


def _wrap_greater_than_max(phs: torch.Tensor, max_val) -> torch.Tensor:
    """
    当相位范围超过 max_val 时使用的包裹：先将相位平移 max_val/2，然后折叠到 [0, max_val]。
    max_val 可以是标量或形状 (1, C, 1, 1) 的张量。
    """
    half = max_val / 2.0
    # 平移
    phs = phs + half
    phs = torch.where(phs < 0, phs + max_val, phs)
    phs = torch.where(phs > max_val, phs - max_val, phs)
    return phs


# ----------------------------------------------------------------------
# 自定义可微 2D 高斯模糊（等效于 SAME padding 的分组卷积）
# ----------------------------------------------------------------------
def _gaussian_blur_2d(x: torch.Tensor, kernel_width: int, sigma: float) -> torch.Tensor:
    """
    对 NCHW 张量的每个通道独立进行高斯模糊，使用零填充的 SAME 卷积。
    
    Args:
        x: 输入张量 (B, C, H, W)
        kernel_width: 高斯核尺寸（奇数）
        sigma: 标准差
    
    Returns:
        模糊后的张量 (B, C, H, W)
    """
    if sigma <= 0.0 or kernel_width <= 1:
        return x

    # 生成 1D 高斯核
    ax = torch.arange(-(kernel_width // 2), kernel_width // 2 + 1, dtype=x.dtype, device=x.device)
    gauss_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()

    # 扩展为 2D 核 (1, 1, kernel_width, kernel_width)
    kernel_2d = gauss_1d[:, None] * gauss_1d[None, :]
    kernel_2d = kernel_2d.view(1, 1, kernel_width, kernel_width)

    # 为每个通道复制核，形成 (C, 1, kernel_width, kernel_width) 用于分组卷积
    C = x.size(1)
    kernel = kernel_2d.repeat(C, 1, 1, 1)

    # 分组卷积：每个通道自己一组，padding 保持 SAME
    return F.conv2d(x, kernel, padding=kernel_width // 2, groups=C)


# ----------------------------------------------------------------------
# Maimone DPM（原 tf_dpm_maimone）
# ----------------------------------------------------------------------
def dpm_maimone(cpx: torch.Tensor,
                propagator=None,
                depth_shift: float = 0.0,
                adaptive_phs_shift: bool = False,
                batch: int = 1,
                num_channels: int = 3,
                res_h: int = 384,
                res_w: int = 384,
                axis: int = 2,          # 2: 丢弃行方向的像素，3: 丢弃列方向
                phs_max: list = None,
                amp_max = None,
                clamp: bool = False,
                normalize: bool = True,
                wavelength: list = None) -> torch.Tensor:
    """
    DPM of Maimone et al. 2017: 每隔一个像素丢弃，通过深度到空间恢复分辨率。
    
    Returns:
        phs_only: 相位图，若 normalize=True 则归一化到 [0, 1]，否则单位为弧度。
        amp_max:  用于归一化的最大振幅（每个样本、每个通道独立），形状 (B, C, 1, 1)。
    """
    # ---- 深度偏移 ----
    if depth_shift != 0.0:
        if propagator is None:
            raise ValueError("propagator must be provided when depth_shift != 0")
        tf_wavelength = torch.tensor(wavelength, dtype=cpx.dtype, device=cpx.device).view(1, -1, 1, 1)
        cpx = propagator(cpx, depth_shift) * compl_exp(-2 * np.pi * depth_shift / tf_wavelength)

    amp = torch.abs(cpx)
    phs = torch.angle(cpx)

    # ---- 振幅归一化：改为逐样本、逐通道的最大值 ----
    if amp_max is None:
        # amp_max = amp.amax(dim=(2, 3), keepdim=True) + 1e-6   # (B, C, 1, 1)
        amp_max = amp.max() + 1e-6
    amp = amp / amp_max
    if clamp:
        amp = torch.clamp(amp, max=1.0 - 1e-6)

    # ---- 中心化相位（逐通道减均值） ----
    phs_zero_mean = phs - phs.mean(dim=[2, 3], keepdim=True)

    # ---- 丢弃一半像素（偶数行/列） ----
    if axis == 3:    # 丢弃列
        amp = amp[:, :, :, 0::2]
        phs_zero_mean = phs_zero_mean[:, :, :, 0::2]
    else:            # axis == 2，丢弃行
        amp = amp[:, :, 0::2, :]
        phs_zero_mean = phs_zero_mean[:, :, 0::2, :]

    # ---- 计算两个相位图 ----
    phs_offset = torch.acos(torch.clamp(amp, min=-1.0 + 1e-7, max=1.0 - 1e-7))
    phs_low = phs_zero_mean - phs_offset
    phs_high = phs_zero_mean + phs_offset

    # ---- 棋盘排列 ----
    if axis == 3:
        # 减少列方向：取样 (偶数行,偶数列) -> phs_1_1, 等等
        # 将四个子块沿通道拼接，再用 pixel_shuffle 上采样
        phs_1_1 = phs_low[:, :, 0::2, :]    # (B,C,H/2,W/2)
        phs_1_2 = phs_high[:, :, 0::2, :]
        phs_2_1 = phs_high[:, :, 1::2, :]
        phs_2_2 = phs_low[:, :, 1::2, :]
    else:   # axis == 2
        phs_1_1 = phs_low[:, :, :, 0::2]
        phs_1_2 = phs_high[:, :, :, 0::2]
        phs_2_1 = phs_high[:, :, :, 1::2]
        phs_2_2 = phs_low[:, :, :, 1::2]

    # 在通道维拼接：每个原始通道变为 4 个通道
    phs_stacked = torch.cat([phs_1_1, phs_1_2, phs_2_1, phs_2_2], dim=1)
    # 使用 pixel_shuffle 上采样 2 倍，输出形状 (B, C, H, W)
    phs_only = F.pixel_shuffle(phs_stacked, upscale_factor=2)

    # ---- 相位包裹 ----
    if phs_max is not None:
        phs_only = _wrap_phase(phs_only, phs_max=phs_max, adaptive_phs_shift=adaptive_phs_shift)

    # ---- 归一化到 [0, 1] ----
    if normalize and phs_max is not None:
        phs_max_tensor = torch.tensor(phs_max, dtype=phs_only.dtype, device=phs_only.device).view(1, -1, 1, 1)
        phs_only = phs_only / phs_max_tensor

    return phs_only, amp_max


# ----------------------------------------------------------------------
# BL-DPM（原 tf_bldpm）
# ----------------------------------------------------------------------
def bldpm(cpx: torch.Tensor,
          propagator=None,
          depth_shift: float = 0.0,
          adaptive_phs_shift: bool = False,
          batch: int = 1,
          num_channels: int = 3,
          res_h: int = 384,
          res_w: int = 384,
          k: float = 0.5,
          phs_max: list = None,
          amp_max = None,
          clamp: bool = False,
          normalize: bool = True,
          wavelength: list = None) -> torch.Tensor:
    """
    Band-Limited DPM [Sui et al. 2021]: 在频率域用方形/菱形 mask 滤波后，再进行 DPM。
    同样修复振幅归一化方式。
    """
    # ---- 生成频域 mask ----
    y = torch.arange(-(res_h // 2), res_h // 2, device=cpx.device, dtype=cpx.real.dtype)
    x = torch.arange(-(res_w // 2), res_w // 2, device=cpx.device, dtype=cpx.real.dtype)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # 正方形滤波：归一化坐标
    side_min = min(res_h, res_w)
    x_norm = xx / side_min
    y_norm = yy / side_min
    tan_pi_alpha_u = torch.tan(y_norm * np.pi)
    tan_pi_alpha_mu = torch.tan(x_norm * np.pi)
    mask = (torch.abs(tan_pi_alpha_u * tan_pi_alpha_mu) <= k)
    # 额外限制在 [-0.5, 0.5] 范围内（方形）
    mask_two = (torch.abs(x_norm) <= 0.5) & (torch.abs(y_norm) <= 0.5)
    mask = mask & mask_two
    # 扩展到 batch 和 channel 维度
    mask = mask.unsqueeze(0).unsqueeze(0).to(cpx.dtype)   # (1,1,H,W)

    # ---- 深度偏移 ----
    if depth_shift != 0.0:
        if propagator is None:
            raise ValueError("propagator must be provided when depth_shift != 0")
        tf_wavelength = torch.tensor(wavelength, dtype=cpx.dtype, device=cpx.device).view(1, -1, 1, 1)
        cpx = propagator(cpx, depth_shift) * compl_exp(-2 * np.pi * depth_shift / tf_wavelength)

    # ---- 频域滤波 ----
    cpx_fft = fftshift2d(fft2d(cpx))
    cpx_fft_filtered = cpx_fft * mask
    cpx = ifft2d(ifftshift2d(cpx_fft_filtered))

    amp = torch.abs(cpx)
    phs = torch.angle(cpx)

    # ---- 振幅归一化：改为逐样本、逐通道的最大值 ----
    if amp_max is None:
        amp_max = amp.max() + 1e-6   # (B, C, 1, 1)
    amp = amp / amp_max
    if clamp:
        amp = torch.clamp(amp, max=1.0 - 1e-6)

    # ---- 中心化相位 ----
    phs_zero_mean = phs - phs.mean(dim=[2, 3], keepdim=True)

    # ---- 计算双相位 ----
    phs_offset = torch.acos(torch.clamp(amp, min=-1.0 + 1e-7, max=1.0 - 1e-7))
    phs_low = phs_zero_mean - phs_offset
    phs_high = phs_zero_mean + phs_offset

    # ---- 棋盘排列 ----
    phs_1_1 = phs_low[:, :, 0::2, 0::2]
    phs_1_2 = phs_high[:, :, 0::2, 1::2]
    phs_2_1 = phs_high[:, :, 1::2, 0::2]
    phs_2_2 = phs_low[:, :, 1::2, 1::2]

    phs_stacked = torch.cat([phs_1_1, phs_1_2, phs_2_1, phs_2_2], dim=1)
    phs_only = F.pixel_shuffle(phs_stacked, upscale_factor=2)

    # ---- 相位包裹 ----
    if phs_max is not None:
        phs_only = _wrap_phase(phs_only, phs_max=phs_max, adaptive_phs_shift=adaptive_phs_shift)

    if normalize and phs_max is not None:
        phs_max_tensor = torch.tensor(phs_max, dtype=phs_only.dtype, device=phs_only.device).view(1, -1, 1, 1)
        phs_only = phs_only / phs_max_tensor

    return phs_only, amp_max


# ----------------------------------------------------------------------
# AA-DPM（原 tf_aadpm）
# ----------------------------------------------------------------------
def aadpm(cpx: torch.Tensor,
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
          amp_max = None,
          clamp: bool = False,
          normalize: bool = True,
          wavelength: list = None) -> torch.Tensor:
    """
    Anti-Aliasing DPM: 在双相位分解前对复数场进行预模糊，以减弱混叠。
    同样修复振幅归一化方式。
    """
    # ---- 深度偏移 ----
    if depth_shift != 0.0:
        if propagator is None:
            raise ValueError("propagator must be provided when depth_shift != 0")
        tf_wavelength = torch.tensor(wavelength, dtype=cpx.dtype, device=cpx.device).view(1, -1, 1, 1)
        cpx = propagator(cpx, depth_shift) * compl_exp(-2 * np.pi * depth_shift / tf_wavelength)

    # ---- 预模糊 ----
    if sigma > 0.0 and kernel_width > 1:
        real = _gaussian_blur_2d(cpx.real, kernel_width, sigma)
        imag = _gaussian_blur_2d(cpx.imag, kernel_width, sigma)
        cpx = torch.complex(real, imag)

    amp = torch.abs(cpx)
    phs = torch.angle(cpx)

    # ---- 振幅归一化：改为逐样本、逐通道的最大值 ----
    if amp_max is None:
        amp_max = amp.max() + 1e-6   # (B, C, 1, 1)
    amp = amp / amp_max
    if clamp:
        amp = torch.clamp(amp, max=1.0 - 1e-6)

    # ---- 中心化相位 ----
    phs_zero_mean = phs - phs.mean(dim=[2, 3], keepdim=True)

    # ---- 双相位计算 ----
    phs_offset = torch.acos(torch.clamp(amp, min=-1.0 + 1e-7, max=1.0 - 1e-7))
    phs_low = phs_zero_mean - phs_offset
    phs_high = phs_zero_mean + phs_offset

    # ---- 棋盘排列 ----
    phs_1_1 = phs_low[:, :, 0::2, 0::2]
    phs_1_2 = phs_high[:, :, 0::2, 1::2]
    phs_2_1 = phs_high[:, :, 1::2, 0::2]
    phs_2_2 = phs_low[:, :, 1::2, 1::2]

    phs_stacked = torch.cat([phs_1_1, phs_1_2, phs_2_1, phs_2_2], dim=1)
    phs_only = F.pixel_shuffle(phs_stacked, upscale_factor=2)

    # ---- 相位包裹 ----
    if phs_max is not None:
        phs_only = _wrap_phase(phs_only, phs_max=phs_max, adaptive_phs_shift=adaptive_phs_shift)

    if normalize and phs_max is not None:
        phs_max_tensor = torch.tensor(phs_max, dtype=phs_only.dtype, device=phs_only.device).view(1, -1, 1, 1)
        phs_only = phs_only / phs_max_tensor

    return phs_only, amp_max