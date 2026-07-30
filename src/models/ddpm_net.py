"""
复数 DDPM 校正网络（DDPMNet）。
用于对主网络输出经过 padding / 深度偏移后的振幅-相位拼接进行精细调整。
所有卷积、BN 均已替换为复数版本，输入/输出通过实虚部重构保持原接口。
"""

import torch
import torch.nn as nn
import numpy as np

from src.data.transforms import interleave_nonnative, deinterleave_nonnative
from src.layers.complex_layers import ComplexConv2d, ComplexBatchNorm2d, CReLU


class DDPMNet(nn.Module):
    """
    复数 DDPM 校正网络。
    输入:  amplitude + phase 拼接 (B, 6, H, W)，前3通道为振幅，后3通道为相位。
    输出:  调整后的 amplitude (B, 3, H, W) 范围 [0, sqrt(2)]
           phase       (B, 3, H, W) 范围 [0, 1]
    内部运算均为复数，输入被重新解释为复数场（实部=amp-0.5，虚部=phs-0.5），
    输出复数场后拆出振幅和相位。
    """
    def __init__(self,
                 input_dim: int = 6,
                 output_dim: int = 6,
                 num_layers: int = 8,
                 num_filters_per_layer: int = 8,
                 interleave_rate: int = 1,
                 filter_width: int = 3,
                 bias_stddev: float = 0.01,      # 保留参数以兼容旧接口，实际偏置按复数层默认
                 weight_var_scale: float = 0.25,
                 activation: nn.Module = CReLU,   # 复数激活
                 output_activation: nn.Module = None):  # 末层无激活
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.interleave_rate = interleave_rate
        self.weight_var_scale = weight_var_scale
        self.bias_stddev = bias_stddev

        # 转换为复数通道数（实虚部各占一半）
        c_in = input_dim // 2          # 3
        c_out = output_dim // 2        # 3
        nf = num_filters_per_layer     # 复数滤波器数
        r2 = interleave_rate ** 2

        # 计算每层的输入/输出复数通道数
        in_ch_list = []
        out_ch_list = []
        for i in range(num_layers):
            if i == 0:
                in_ch = c_in * r2
                out_ch = nf
            elif i == num_layers - 1:
                in_ch = nf + c_in * r2   # 跳跃连接拼接
                out_ch = c_out * r2
            else:
                in_ch = nf
                out_ch = nf
            in_ch_list.append(in_ch)
            out_ch_list.append(out_ch)

        # 构建复数卷积层（Conv + BN + 可选激活）
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            conv = ComplexConv2d(
                in_channels=in_ch_list[i],
                out_channels=out_ch_list[i],
                kernel_size=filter_width,
                padding=filter_width // 2,
                bias=True,
                weight_var_scale=self.weight_var_scale
            )
            bn = ComplexBatchNorm2d(out_ch_list[i])
            if is_last:
                self.layers.append(nn.Sequential(conv, bn))
            else:
                self.layers.append(nn.Sequential(conv, bn, CReLU()))

    # ---------- 复数 interleave / deinterleave ----------
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

    def forward(self, amp_phs: torch.Tensor):
        """
        Args:
            amp_phs: 实数张量 (B, 6, H, W)，前3通道为振幅，后3通道为相位。
        Returns:
            amp: (B, 3, H, W) 振幅，范围约 [0, sqrt(2)]
            phs: (B, 3, H, W) 相位，范围 [0, 1]
        """
        # 1. 减去中心值（与原实数网络一致），并构造复数输入
        x = amp_phs - 0.5  # (B, 6, H, W)
        x = torch.complex(x[:, :3], x[:, 3:])  # 实部：amp-0.5，虚部：phs-0.5

        # 2. 可选 interleave
        if self.interleave_rate > 1:
            x = self._complex_interleave(self.interleave_rate, x)
        x_in = x  # 保留跳跃连接

        # 3. 逐层计算（含跳跃/残差连接，逻辑与原网络一致）
        prev_outputs = []
        for i in range(self.num_layers):
            if i == 0:
                prev = x_in
            elif i < 3 or (i % 2 == 0):
                prev = prev_outputs[i-1]
            else:
                prev = prev_outputs[i-1] + prev_outputs[i-2]

            if i == self.num_layers - 1:
                prev = torch.cat([prev, x_in], dim=1)

            out = self.layers[i](prev)
            prev_outputs.append(out)

        field_complex = prev_outputs[-1]  # (B, c_out*r2, H, W)

        # 4. 可选 deinterleave
        if self.interleave_rate > 1:
            field_complex = self._complex_deinterleave(self.interleave_rate, field_complex)
        # 现在形状 (B, 3, H, W) 复数

        # 5. 拆回振幅和相位（保持与外部模块兼容）
        amp = torch.abs(field_complex)
        phs = torch.angle(field_complex) / (2.0 * np.pi) + 0.5

        return amp, phs