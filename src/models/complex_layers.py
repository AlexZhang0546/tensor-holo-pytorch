# src/models/complex_layers.py
"""
复值神经网络基础模块：
  - ComplexConv2d: 复数二维卷积（通过两个实卷积实现）
  - ComplexBatchNorm2d: 简化版复数批归一化（分别对实/虚部做 BN）
  - ComplexReLU: modReLU 激活函数
  - complex_xavier_uniform_: 复数 Xavier 初始化函数
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv2d(nn.Module):
    """
    复数二维卷积层。
    内部使用两个实值 nn.Conv2d 分别处理实部与虚部，
    通过复数乘法规则组合输出。
    
    输入：
        x : 复数张量 (B, C_in, H, W) 或实数张量。
            若为实数则自动视为实部 + 零虚部。
    输出：
        复数张量 (B, C_out, H', W')
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        # 两个独立的实卷积
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, bias)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, bias)
        self.reset_parameters()

    def reset_parameters(self):
        # 计算 fan_in / fan_out（与 Xavier 初始化一致）
        fan_in = self.conv_real.in_channels * self.conv_real.kernel_size[0] * self.conv_real.kernel_size[1]
        fan_out = self.conv_real.out_channels * self.conv_real.kernel_size[0] * self.conv_real.kernel_size[1]
        complex_xavier_uniform_(self.conv_real.weight, self.conv_imag.weight, fan_in, fan_out)
        if self.conv_real.bias is not None and self.conv_imag.bias is not None:
            nn.init.zeros_(self.conv_real.bias)
            nn.init.zeros_(self.conv_imag.bias)

    def forward(self, x):
        # 提取实部与虚部
        if torch.is_complex(x):
            x_real, x_imag = x.real, x.imag
        else:
            x_real, x_imag = x, torch.zeros_like(x)

        # 复数卷积： (A_real + j A_imag) * (W_real + j W_imag)
        out_real = self.conv_real(x_real) - self.conv_imag(x_imag)
        out_imag = self.conv_real(x_imag) + self.conv_imag(x_real)
        return torch.complex(out_real, out_imag)


class ComplexBatchNorm2d(nn.Module):
    """
    简化版复数批归一化。
    分别对实部和虚部使用标准 2D 批归一化。
    
    输入/输出：复数张量 (B, C, H, W)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.bn_real = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_imag = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        x_real, x_imag = x.real, x.imag
        out_real = self.bn_real(x_real)
        out_imag = self.bn_imag(x_imag)
        return torch.complex(out_real, out_imag)


class ComplexReLU(nn.Module):
    """
    modReLU 激活函数。
    
    modReLU(z) = ReLU(|z| + b) * (z / |z|)
    其中 b 是可学习的偏置参数，每个通道独立。
    
    输入/输出：复数张量 (B, C, H, W)
    """
    def __init__(self, num_features):
        super().__init__()
        # 每个通道一个可学习的偏置 b，初始化为 0
        self.b = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        abs_z = torch.abs(x)                           # (B, C, H, W)
        bias = self.b.view(1, -1, 1, 1)               # (1, C, 1, 1)
        mask = F.relu(abs_z + bias)                    # 非负部分
        # 避免除零
        return mask * (x / (abs_z + 1e-8))


def complex_xavier_uniform_(weight_real, weight_imag, fan_in, fan_out):
    """
    复数 Xavier 初始化（in‑place）。
    
    幅度: sqrt(2 / (fan_in + fan_out))
    相位: 均匀分布 U[-π, π]
    
    Args:
        weight_real: 实部权重张量 (nn.Parameter 的 data)
        weight_imag: 虚部权重张量 (nn.Parameter 的 data)
        fan_in: 输入单元数
        fan_out: 输出单元数
    """
    amplitude = math.sqrt(2.0 / (fan_in + fan_out))
    with torch.no_grad():
        # 生成与权重形状一致的随机相位
        phase = torch.empty_like(weight_real).uniform_(-math.pi, math.pi)
        weight_real.copy_(amplitude * torch.cos(phase))
        weight_imag.copy_(amplitude * torch.sin(phase))