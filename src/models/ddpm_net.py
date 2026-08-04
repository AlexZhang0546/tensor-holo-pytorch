# src/models/ddpm_net.py
"""
复数 DDPM 校正网络（ComplexDDPMNet）。
输入为复数光场 (B, 3, H, W)，输出同样形状的复数光场。
内部使用复数卷积、批归一化、modReLU 激活以及可选的 interleave/deinterleave 操作。
"""

import torch
import torch.nn as nn
import numpy as np

from src.data.transforms import interleave_nonnative, deinterleave_nonnative
from src.models.complex_layers import ComplexConv2d, ComplexBatchNorm2d, ComplexReLU


class ComplexDDPMNet(nn.Module):
    """
    复数 DDPM 网络，用于对传播后的复数全息图进行精细化校正。
    
    输入:  (B, 3, H, W) 复数光场
    输出:  (B, 3, H, W) 校正后的复数光场
    """

    def __init__(self,
                 input_dim: int = 3,
                 output_dim: int = 3,
                 num_layers: int = 8,
                 num_filters_per_layer: int = 8,
                 interleave_rate: int = 1,
                 filter_width: int = 3,
                 bias_stddev: float = 0.01,
                 weight_var_scale: float = 0.25):
        """
        Args:
            input_dim: 输入复数通道数（默认 3）。
            output_dim: 输出复数通道数（默认 3）。
            num_layers: 网络总层数（默认 8）。
            num_filters_per_layer: 中间层滤波器数量（复数通道数）。
            interleave_rate: 空间下采样倍率（默认 1，不进行）。
            filter_width: 卷积核尺寸（默认 3）。
            bias_stddev: 偏置初始化标准差。
            weight_var_scale: 权重初始化方差缩放系数 r。
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_filters = num_filters_per_layer
        self.interleave_rate = interleave_rate
        self.filter_width = filter_width
        self.weight_var_scale = weight_var_scale
        self.bias_stddev = bias_stddev

        # 计算各层的输入/输出复数通道数
        in_dim_list = []
        out_dim_list = []
        for i in range(num_layers):
            if i == 0:
                # 第一层：输入经过 interleave 后通道数倍增
                in_dim = input_dim * (interleave_rate ** 2)
                out_dim = num_filters_per_layer
            elif i == num_layers - 1:
                # 最后一层：拼接原始输入，输出 output_dim * r^2，再 deinterleave
                in_dim = num_filters_per_layer + input_dim * (interleave_rate ** 2)
                out_dim = output_dim * (interleave_rate ** 2)
            else:
                in_dim = num_filters_per_layer
                out_dim = num_filters_per_layer
            in_dim_list.append(in_dim)
            out_dim_list.append(out_dim)

        # 构建复数层序列
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            conv = ComplexConv2d(
                in_channels=in_dim_list[i],
                out_channels=out_dim_list[i],
                kernel_size=filter_width,
                padding=filter_width // 2,  # SAME padding
                bias=True
            )
            bn = ComplexBatchNorm2d(out_dim_list[i])

            if is_last:
                # 最后一层不使用激活函数（线性输出）
                self.layers.append(nn.Sequential(conv, bn))
            else:
                self.layers.append(nn.Sequential(conv, bn, ComplexReLU(out_dim_list[i])))

        # 自定义权重/偏置初始化
        self._init_weights()

    def _init_weights(self):
        """使用与原项目一致的 Xavier 初始化和截断正态偏置。"""
        for i, layer_block in enumerate(self.layers):
            conv = layer_block[0]  # ComplexConv2d 是第一个模块
            fan_in = conv.conv_real.in_channels * self.filter_width * self.filter_width
            fan_out = conv.conv_real.out_channels * self.filter_width * self.filter_width

            # Xavier 幅度调整（与 util.py 中 tf_init_weights 一致）
            high = np.sqrt(self.weight_var_scale * 2.0 / (fan_in + fan_out))
            low = -high

            # 实部卷积权重与偏置
            with torch.no_grad():
                conv.conv_real.weight.uniform_(low, high)
                conv.conv_imag.weight.uniform_(low, high)
                if conv.conv_real.bias is not None:
                    conv.conv_real.bias.normal_(std=self.bias_stddev)
                if conv.conv_imag.bias is not None:
                    conv.conv_imag.bias.normal_(std=self.bias_stddev)

    def _complex_interleave(self, r: int, x: torch.Tensor) -> torch.Tensor:
        """对复数张量进行空间→通道重排（interleave）。"""
        if r == 1:
            return x
        real = interleave_nonnative(r, x.real)
        imag = interleave_nonnative(r, x.imag)
        return torch.complex(real, imag)

    def _complex_deinterleave(self, r: int, x: torch.Tensor) -> torch.Tensor:
        """对复数张量进行通道→空间重排（deinterleave）。"""
        if r == 1:
            return x
        real = deinterleave_nonnative(r, x.real)
        imag = deinterleave_nonnative(r, x.imag)
        return torch.complex(real, imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 复数输入 (B, 3, H, W)。

        Returns:
            torch.Tensor: 校正后的复数光场 (B, 3, H, W)。
        """
        # 1. 重归一化（与原始 TF 代码一致，将输入中心化）
        x = x - 0.5

        # 2. 可选 interleave（空间→通道）
        if self.interleave_rate > 1:
            x = self._complex_interleave(self.interleave_rate, x)
        x_in = x  # 保存用于跳跃连接及最后一层拼接

        # 3. 逐层前向传播，包含跳跃连接
        prev_outputs = []
        for i in range(self.num_layers):
            # 构建当前层输入
            if i == 0:
                prev = x_in
            elif i < 3 or (i % 2 == 0):
                # 前 3 层或偶数层（从 0 开始计数），直接使用上一层输出
                prev = prev_outputs[i - 1]
            else:
                # 奇数层（i>=3 且奇数），使用残差连接
                prev = prev_outputs[i - 1] + prev_outputs[i - 2]

            # 最后一层需拼接原始输入
            if i == self.num_layers - 1:
                prev = torch.cat([prev, x_in], dim=1)

            out = self.layers[i](prev)
            prev_outputs.append(out)

        # 4. 最后一层输出（复数场）
        field_complex = prev_outputs[-1]  # (B, out_dim*r^2, H/r, W/r) 或 (B, out_dim, H, W)

        # 5. 可选 deinterleave（恢复空间分辨率）
        if self.interleave_rate > 1:
            field_complex = self._complex_deinterleave(self.interleave_rate, field_complex)

        # 6. 直接返回复数光场（不再拆分实部/虚部）
        return field_complex