"""
角谱（AS）和菲涅尔（Fresnel）传播算子。
将原 TensorFlow 中的 Propagation 基类体系改写为 PyTorch nn.Module，
支持 double_pad 以抑制边缘效应，工厂函数 propagator_factory 替代 tf_propagator。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .complex_utils import compl_exp, fft2d, ifft2d, ifftshift2d


class Propagation(nn.Module):
    """
    衍射传播基类。
    输入形状：(B, C, H, W) 复数场。
    使用角谱法或菲涅尔近似，自动处理 double_pad 扩充。
    """

    def __init__(self,
                 input_shape: tuple,      # (H, W)
                 pitch: float,
                 wavelengths: np.ndarray, # (3,) 或 (num_channels,)
                 double_pad: bool = False):
        """
        Args:
            input_shape: 原始无 padding 的 (高, 宽)。
            pitch:       像素尺寸 (mm)。
            wavelengths: 各通道波长数组 (mm)，例如 [450e-6, 520e-6, 638e-6]。
            double_pad:  是否将输入尺寸加倍扩充（原代码的 double_pad）。
        """
        super().__init__()
        self.input_shape = input_shape
        if double_pad:
            self.m_pad = input_shape[0] // 2
            self.n_pad = input_shape[1] // 2
        else:
            self.m_pad = 0
            self.n_pad = 0

        # 波长张量，形状 (1, C, 1, 1)
        wlen = np.array(wavelengths).reshape(1, -1, 1, 1).astype(np.float32)
        self.register_buffer('wavelengths', torch.from_numpy(wlen))

        # 预计算频率坐标 fx, fy（已做 ifftshift，可直接与 FFT 结果相乘）
        self.pitch = pitch
        fx, fy = self._compute_xy_grid()
        self.register_buffer('fx', fx)
        self.register_buffer('fy', fy)

        # unit_phase_shift 在子类中缓存为 buffer，这里先置 None
        # self.unit_phase_shift = None

    def _compute_xy_grid(self):
        """
        生成空间频率网格 fx, fy，并应用 ifftshift2d。
        返回形状 (1, 1, H_pad, W_pad) 的张量。
        """
        M = self.input_shape[0] + 2 * self.m_pad
        N = self.input_shape[1] + 2 * self.n_pad

        # 坐标从 -N/2 到 N/2-1（等间距）
        x = torch.arange(-(N // 2), N // 2, dtype=torch.float32)
        y = torch.arange(-(M // 2), M // 2, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')  # 矩阵索引：yy 对应高度

        # 空间频率：除以 pitch * 尺寸
        fx = xx / (self.pitch * N)
        fy = yy / (self.pitch * M)

        # 添加 batch 和 channel 维度 (1,1,H,W)
        fx = fx.unsqueeze(0).unsqueeze(0)
        fy = fy.unsqueeze(0).unsqueeze(0)

        # 应用 ifftshift，将零频率从中心移到角落（与 torch.fft.fft2 无移位对应）
        fx = ifftshift2d(fx)
        fy = ifftshift2d(fy)
        return fx, fy

    def _compute_unit_phase_shift(self):
        """
        由子类实现，返回单位距离的相位延迟张量 (1, C, H_pad, W_pad)。
        """
        raise NotImplementedError

    def forward(self, input_field: torch.Tensor, z_dist: float) -> torch.Tensor:
        """
        执行传播。

        Args:
            input_field: 复数场，形状 (B, C, H, W)。
            z_dist:      传播距离 (mm)，正值表示远离源平面。

        Returns:
            复数场，形状 (B, C, H, W)。
        """
        return self._propagate(input_field, z_dist)

    def _propagate(self, input_field: torch.Tensor, z_dist: float) -> torch.Tensor:
        # 确保 unit_phase_shift 已计算
        if self.unit_phase_shift is None:
            self.unit_phase_shift = self._compute_unit_phase_shift()

        # 对输入进行常数零填充
        padded = F.pad(input_field,
                       pad=(self.n_pad, self.n_pad, self.m_pad, self.m_pad),
                       mode='constant', value=0.0)

        # 计算传递函数 H（形状 (1, C, H_pad, W_pad)）
        H = compl_exp(z_dist * self.unit_phase_shift)

        # 前向 FFT（无 fftshift）
        obj_fft = fft2d(padded)

        # 频域相乘
        out_fft = obj_fft * H

        # 逆 FFT，并裁剪回原始尺寸
        out_field_padded = ifft2d(out_fft)
        out_field = out_field_padded[:, :,
                                     self.m_pad:self.m_pad + self.input_shape[0],
                                     self.n_pad:self.n_pad + self.input_shape[1]]
        return out_field


class ASPropagation(Propagation):
    """
    角谱传播（Angular Spectrum）。
    unit_phase_shift = 2π/λ * sqrt(1 - (λ fx)² - (λ fy)²)
    """
    def __init__(self, input_shape, pitch, wavelengths, double_pad=False):
        super().__init__(input_shape, pitch, wavelengths, double_pad)
        # 立即计算并注册为 buffer（这样 .to(device) 时会自动迁移）
        self.register_buffer('unit_phase_shift', self._compute_unit_phase_shift())

    def _compute_unit_phase_shift(self):
        # 波长和频率需要广播：波长形状 (1, C, 1, 1), fx,fy 形状 (1, 1, H, W)
        # 结果形状 (1, C, H, W)
        term = 1.0 - (self.wavelengths * self.fx) ** 2 - (self.wavelengths * self.fy) ** 2
        # 数值稳定性：确保 sqrt 内非负（倏逝波可设为零相位或小正数，这里按原逻辑直接 sqrt）
        term = torch.clamp(term, min=0.0)
        phase_shift = 2.0 * np.pi * (1.0 / self.wavelengths) * torch.sqrt(term)
        return phase_shift


class FresnelPropagation(Propagation):
    """
    菲涅尔近似传播。
    unit_phase_shift = -λ * π * (fx² + fy²)
    """
    def __init__(self, input_shape, pitch, wavelengths, double_pad=False):
        super().__init__(input_shape, pitch, wavelengths, double_pad)
        self.register_buffer('unit_phase_shift', self._compute_unit_phase_shift())

    def _compute_unit_phase_shift(self):
        # 形状 (1, 1, H, W)，但需广播到 (1, C, H, W)
        squared_sum = self.fx ** 2 + self.fy ** 2
        phase_shift = -self.wavelengths * np.pi * squared_sum
        return phase_shift


def propagator_factory(input_shape: tuple,
                       pitch: float,
                       wavelengths: np.ndarray,
                       method: str = "as",
                       double_pad: bool = False) -> Propagation:
    """
    传播器工厂函数，替代原 tf_propagator。
    
    Args:
        input_shape: (H, W) 原始全息图分辨率。
        pitch:       像素尺寸 (mm)。
        wavelengths: 波长数组 (mm)。
        method:      传播方法，支持 'as'（角谱）或 'fresnel'（菲涅尔）。
        double_pad:  是否使用双倍填充。
    
    Returns:
        Propagation 子类的实例。
    """
    if method == "as":
        return ASPropagation(input_shape, pitch, wavelengths, double_pad)
    elif method == "fresnel":
        return FresnelPropagation(input_shape, pitch, wavelengths, double_pad)
    else:
        raise ValueError(f"Unsupported propagation method: {method}. Choose 'as' or 'fresnel'.")