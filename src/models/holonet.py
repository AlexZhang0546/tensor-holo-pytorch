"""
主 U‑Net 风格的全息预测网络（复数版本）。
所有内部运算均为复数，输出复数全息图，
再转换为振幅和相位以保持与外部实数模块的接口兼容。
"""

import torch
import torch.nn as nn
import numpy as np

from src.data.transforms import interleave_nonnative, deinterleave_nonnative
from src.layers.complex_layers import ComplexConv2d, ComplexBatchNorm2d, CReLU


class TensorHolographyNet(nn.Module):
    """
    复数主全息预测网络。
    输入:  (B, input_dim, H, W)   # NCHW 实数 RGBD 或 LDI
    输出:  amp (B, output_dim//2, H, W)  直接取复数振幅（值域由网络学习）
           phs (B, output_dim//2, H, W)  [0, 1] 归一化相位
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int = 6,       # 保持 6，内部实际输出 3 个复数通道
                 num_layers: int = 30,
                 num_filters_per_layer: int = 24,
                 interleave_rate: int = 1,
                 filter_width: int = 3,
                 bias_stddev: float = 0.01,
                 weight_var_scale: float = 0.25):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_filters = num_filters_per_layer
        self.interleave_rate = interleave_rate
        self.filter_width = filter_width

        # 复数输出通道数（例如 output_dim=6 → 3 个复数）
        num_complex_out = output_dim // 2

        # ---- 计算各层输入/输出通道数（复数通道数） ----
        in_dim_list = []
        out_dim_list = []
        for i in range(num_layers):
            if i == 0:
                # 第一层：输入实数转复数，通道数仍为 input_dim，经过 interleave 倍增
                in_dim = input_dim * (interleave_rate ** 2)
                out_dim = num_filters_per_layer
            elif i == num_layers - 1:
                # 最后一层：输出 num_complex_out 个复数，经过 deinterleave 前需要乘以 r^2
                in_dim = num_filters_per_layer + input_dim * (interleave_rate ** 2)
                out_dim = num_complex_out * (interleave_rate ** 2)
            else:
                in_dim = num_filters_per_layer
                out_dim = num_filters_per_layer
            in_dim_list.append(in_dim)
            out_dim_list.append(out_dim)

        # ---- 构建复数层（ComplexConv2d + ComplexBatchNorm2d + CReLU） ----
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            is_last = (i == num_layers - 1)

            conv = ComplexConv2d(
                in_channels=in_dim_list[i],
                out_channels=out_dim_list[i],
                kernel_size=filter_width,
                padding=filter_width // 2,   # SAME padding
                bias=True,
                weight_var_scale=weight_var_scale
            )
            bn = ComplexBatchNorm2d(out_dim_list[i])

            if is_last:
                # 最后一层不做激活，保持线性
                self.layers.append(nn.Sequential(conv, bn))
            else:
                self.layers.append(nn.Sequential(conv, bn, CReLU()))

    def _complex_interleave(self, r: int, x: torch.Tensor) -> torch.Tensor:
        """对复数张量进行空间->通道重排（interleave）。"""
        if r == 1:
            return x
        real = interleave_nonnative(r, x.real)
        imag = interleave_nonnative(r, x.imag)
        return torch.complex(real, imag)

    def _complex_deinterleave(self, r: int, x: torch.Tensor) -> torch.Tensor:
        """对复数张量进行通道->空间重排（deinterleave）。"""
        if r == 1:
            return x
        real = deinterleave_nonnative(r, x.real)
        imag = deinterleave_nonnative(r, x.imag)
        return torch.complex(real, imag)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: 实数输入 (B, input_dim, H, W)，值任意。
        Returns:
            amp: 振幅 (B, output_dim//2, H, W)
            phs: 相位 (B, output_dim//2, H, W)，范围 [0, 1]
        """
        # 1. 输入重归一化（与原代码一致）
        x = x - 0.5

        # 2. 实数 → 复数（虚部置零）
        x = torch.complex(x, torch.zeros_like(x))

        # 3. 可选 interleave（空间→通道）
        if self.interleave_rate > 1:
            x = self._complex_interleave(self.interleave_rate, x)
        x_in = x  # 保存用于跳跃连接

        # 4. 逐层计算
        prev_outputs = []
        for i in range(self.num_layers):
            # --- 构造输入 ---
            if i == 0:
                prev = x_in
            elif i < 3 or (i % 2 == 0):
                prev = prev_outputs[i-1]
            else:
                # 残差连接（复数加法直接支持）
                prev = prev_outputs[i-1] + prev_outputs[i-2]

            # 最后一层需拼接原始输入
            if i == self.num_layers - 1:
                prev = torch.cat([prev, x_in], dim=1)  # 复数拼接，通道维为 dim=1

            out = self.layers[i](prev)
            prev_outputs.append(out)

        # 5. 最后一层输出（复数场）
        field_complex = prev_outputs[-1]   # (B, out_dim, H, W)，复数

        # 6. 可选 deinterleave
        if self.interleave_rate > 1:
            field_complex = self._complex_deinterleave(self.interleave_rate, field_complex)
        # 现在 field_complex 形状: (B, num_complex_out, H, W)  e.g. (B, 3, H, W)

        # 7. 提取振幅和相位（兼容旧接口）
        amp = torch.abs(field_complex)
        phs = torch.angle(field_complex) / (2.0 * np.pi) + 0.5

        return amp, phs