# src/models/ddpm_net.py
"""
复数 DDPM 校正网络（ComplexDDPMNet）。
输入:  复数全息场 (B, 3, H, W) —— 已由主网络输出并经传播得到的复数场。
输出:  校正后的复数全息场 (B, 3, H, W)，直接用于后续双相位编码等操作。
所有内部运算均基于复数模块（ComplexConv2d, ComplexBatchNorm2d, ComplexReLU）。
"""

import torch
import torch.nn as nn
import numpy as np

from src.data.transforms import interleave_nonnative, deinterleave_nonnative
from src.models.complex_layers import ComplexConv2d, ComplexBatchNorm2d, ComplexReLU


class ComplexDDPMNet(nn.Module):
    """
    复数 DDPM 校正网络。
    结构类似原实数 DDPMNet，但所有卷积/归一化/激活均替换为复数版本。
    跳跃连接策略保持不变，最后一层为线性（无激活），直接输出复数场。
    """
    def __init__(self,
                 input_dim: int = 3,                # 复数输入通道数（对应 RGB 三色）
                 output_dim: int = 3,               # 复数输出通道数
                 num_layers: int = 8,
                 num_filters_per_layer: int = 8,    # 中间层的复数通道数
                 interleave_rate: int = 1,
                 filter_width: int = 3,
                 bias_stddev: float = 0.01,
                 weight_var_scale: float = 0.25):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.interleave_rate = interleave_rate
        self.filter_width = filter_width

        # ---- 计算各层输入/输出通道数（均为复数通道） ----
        in_dim_list = []
        out_dim_list = []
        for i in range(num_layers):
            if i == 0:
                in_dim = input_dim * (interleave_rate ** 2)
                out_dim = num_filters_per_layer
            elif i == num_layers - 1:
                in_dim = num_filters_per_layer + input_dim * (interleave_rate ** 2)
                out_dim = output_dim * (interleave_rate ** 2)
            else:
                in_dim = num_filters_per_layer
                out_dim = num_filters_per_layer
            in_dim_list.append(in_dim)
            out_dim_list.append(out_dim)

        # ---- 构建复数层 ----
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            conv = ComplexConv2d(
                in_channels=in_dim_list[i],
                out_channels=out_dim_list[i],
                kernel_size=filter_width,
                padding=filter_width // 2,
                bias=True
            )
            bn = ComplexBatchNorm2d(out_dim_list[i])
            if is_last:
                # 最后一层去掉 BN，保持纯线性，避免强制归一化破坏物理振幅尺度
                self.layers.append(conv)
            else:
                self.layers.append(nn.Sequential(conv, bn, ComplexReLU(out_dim_list[i])))

    def _complex_interleave(self, r: int, x: torch.Tensor) -> torch.Tensor:
        """对复数张量进行空间->通道重排（interleave），分别处理实部虚部。"""
        if r == 1:
            return x
        real = interleave_nonnative(r, x.real)
        imag = interleave_nonnative(r, x.imag)
        return torch.complex(real, imag)

    def _complex_deinterleave(self, r: int, x: torch.Tensor) -> torch.Tensor:
        """对复数张量进行通道->空间重排（deinterleave），分别处理实部虚部。"""
        if r == 1:
            return x
        real = deinterleave_nonnative(r, x.real)
        imag = deinterleave_nonnative(r, x.imag)
        return torch.complex(real, imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 复数场 (B, 3, H, W)，通常为传播后的 holo_shifted。
        Returns:
            校正后的复数场 (B, 3, H, W)。
        """
        # 1. 可选 interleave
        if self.interleave_rate > 1:
            x = self._complex_interleave(self.interleave_rate, x)
        x_in = x  # 保存用于跳跃连接

        # 2. 逐层计算（跳跃连接与原 DDPMNet 相同）
        prev_outputs = []
        for i in range(self.num_layers):
            # 构造当前层的输入
            if i == 0:
                prev = x_in
            elif i < 3 or (i % 2 == 0):
                prev = prev_outputs[i-1]
            else:
                # 残差连接：复数加法直接支持
                prev = prev_outputs[i-1] + prev_outputs[i-2]

            # 最后一层拼接原始输入
            if i == self.num_layers - 1:
                prev = torch.cat([prev, x_in], dim=1)  # 复数拼接在通道维

            out = self.layers[i](prev)
            prev_outputs.append(out)

        # 3. 最后一层输出
        field_complex = prev_outputs[-1]   # (B, out_dim, H, W) 复数

        # 4. 可选 deinterleave
        if self.interleave_rate > 1:
            field_complex = self._complex_deinterleave(self.interleave_rate, field_complex)
        # 现在形状为 (B, 3, H, W)

        # 5. 直接返回复数场，不做额外的振幅/相位分离
        # 核心修正：将其作为残差叠加到原始输入上，网络仅学习物理光场的微小修正量
        return x + field_complex


# ---------- 保留旧版 DDPMNet（可选，以便兼容未改造部分） ----------
class DDPMNet(nn.Module):
    """
    原实数 DDPM 校正网络（保留用于参考或兼容，若全部替换可删除）。
    """
    def __init__(self,
                 input_dim: int = 6,
                 output_dim: int = 6,
                 num_layers: int = 8,
                 num_filters_per_layer: int = 8,
                 interleave_rate: int = 1,
                 filter_width: int = 3,
                 bias_stddev: float = 0.01,
                 weight_var_scale: float = 0.25,
                 activation: nn.Module = nn.ReLU,
                 output_activation: nn.Module = nn.Tanh):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.interleave_rate = interleave_rate
        self.weight_var_scale = weight_var_scale
        self.bias_stddev = bias_stddev

        in_dim_list = []
        out_dim_list = []
        for i in range(num_layers):
            if i == 0:
                in_dim = input_dim * (interleave_rate ** 2)
                out_dim = num_filters_per_layer
            elif i == num_layers - 1:
                in_dim = num_filters_per_layer + input_dim * (interleave_rate ** 2)
                out_dim = output_dim * (interleave_rate ** 2)
            else:
                in_dim = num_filters_per_layer
                out_dim = num_filters_per_layer
            in_dim_list.append(in_dim)
            out_dim_list.append(out_dim)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            act = output_activation() if is_last else activation()
            conv = nn.Conv2d(in_dim_list[i], out_dim_list[i],
                             kernel_size=filter_width,
                             padding=filter_width // 2,
                             bias=True)
            bn = nn.BatchNorm2d(out_dim_list[i])
            self.layers.append(nn.Sequential(conv, bn, act))

        self.reset_parameters()

    def reset_parameters(self):
        from src.utils.weight_init import init_weights_real
        for i, layer in enumerate(self.layers):
            conv = layer[0]
            fan_in = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
            fan_out = conv.out_channels * conv.kernel_size[0] * conv.kernel_size[1]
            init_weights_real(conv.weight, fan_in, fan_out, r=self.weight_var_scale)
            nn.init.trunc_normal_(conv.bias, std=self.bias_stddev)

    def forward(self, amp_phs: torch.Tensor):
        x = amp_phs - 0.5
        if self.interleave_rate > 1:
            x = interleave_nonnative(self.interleave_rate, x)
        x_in = x

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

        field = prev_outputs[-1]
        if self.interleave_rate > 1:
            field = deinterleave_nonnative(self.interleave_rate, field)
        amp = field[:, :3, :, :] * np.sqrt(0.5) + np.sqrt(0.5)
        phs = field[:, 3:, :, :] * 0.5 + 0.5
        return amp, phs