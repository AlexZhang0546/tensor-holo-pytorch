# src/layers/complex_layers.py
"""
复数神经网络基础层库。
包含：ComplexConv2d, ComplexBatchNorm2d, CReLU, ModReLU, complex_weight_init
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.utils.weight_init import init_weights_complex


class ComplexConv2d(nn.Module):
    """
    复数二维卷积。
    使用两个实数卷积核分别代表复数权重的实部和虚部，
    完全遵循复数乘法规则：
        output_real = conv_real(input_real) - conv_imag(input_imag)
        output_imag = conv_real(input_imag) + conv_imag(input_real)

    权重初始化通过 complex_weight_init 进行（独立 Xaiver-uniform）。
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 weight_var_scale: float = 0.5):
        """
        参数与 torch.nn.Conv2d 基本一致，额外：
            weight_var_scale: 用于复数初始化的方差因子 r。
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # 实部卷积核
        self.conv_real = nn.Conv2d(in_channels, out_channels,
                                   kernel_size, stride, padding, dilation,
                                   groups, bias=bias)
        # 虚部卷积核
        self.conv_imag = nn.Conv2d(in_channels, out_channels,
                                   kernel_size, stride, padding, dilation,
                                   groups, bias=bias)

        self.weight_var_scale = weight_var_scale
        self.reset_parameters()

    def reset_parameters(self):
        """使用复数 Xavier-uniform 初始化实部和虚部权重，偏置置零。"""
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        fan_out = self.out_channels * self.kernel_size * self.kernel_size
        init_weights_complex(self.conv_real.weight, self.conv_imag.weight,
                             fan_in, fan_out, r=self.weight_var_scale)
        if self.bias:
            nn.init.constant_(self.conv_real.bias, 0.)
            nn.init.constant_(self.conv_imag.bias, 0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 复数张量，形状 (B, C_in, H, W)，dtype=torch.complex64/complex128
        Returns:
            复数张量，形状 (B, C_out, H_out, W_out)
        """
        x_real = x.real
        x_imag = x.imag

        out_real = self.conv_real(x_real) - self.conv_imag(x_imag)
        out_imag = self.conv_real(x_imag) + self.conv_imag(x_real)

        return torch.complex(out_real, out_imag)


class ComplexBatchNorm2d(nn.Module):
    """
    复数批归一化（简化稳定版）：
    对实部和虚部分别使用独立的 nn.BatchNorm2d，共享 momentum（默认相同）。
    """
    def __init__(self, num_features: int, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.bn_real = nn.BatchNorm2d(num_features, eps, momentum, affine,
                                      track_running_stats)
        self.bn_imag = nn.BatchNorm2d(num_features, eps, momentum, affine,
                                      track_running_stats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 复数张量 (B, C, H, W)
        Returns:
            归一化后的复数张量
        """
        x_real = x.real
        x_imag = x.imag

        out_real = self.bn_real(x_real)
        out_imag = self.bn_imag(x_imag)

        return torch.complex(out_real, out_imag)


class CReLU(nn.Module):
    """复数 ReLU：对实部和虚部分别应用 ReLU。"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(F.relu(x.real), F.relu(x.imag))


class ModReLU(nn.Module):
    """
    基于幅度的复数 ReLU：
        ModReLU(z) = ReLU(|z| + b) * (z / |z|)
    其中 b 为可学习偏置（标量）。
    """
    def __init__(self, init_b: float = 0.0):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(init_b, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(x)
        scale = F.relu(mag + self.b) / (mag + 1e-7)
        return x * scale


def complex_weight_init(real_weight: torch.Tensor,
                        imag_weight: torch.Tensor,
                        fan_in: int,
                        fan_out: int,
                        r: float = 0.5,
                        seed: int = 0):
    """
    便捷的复数权重初始化函数，直接调用 src.utils.weight_init.init_weights_complex。
    """
    return init_weights_complex(real_weight, imag_weight, fan_in, fan_out, r, seed)