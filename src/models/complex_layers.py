# src/models/complex_layers.py
"""
复值神经网络基础模块：
  - ComplexConv2d: 复数二维卷积（两个实卷积）
  - ComplexBatchNorm2d: 完整复数批归一化（白化+去相关，已修复广播错误）
  - SimpleComplexBatchNorm2d: 简化版复数批归一化（分别对实/虚部 BN，仅供对照）
  - ComplexReLU: modReLU 激活函数
  - complex_xavier_uniform_: 复数 Xavier 初始化
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
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, bias)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, bias)
        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.conv_real.in_channels * self.conv_real.kernel_size[0] * self.conv_real.kernel_size[1]
        fan_out = self.conv_real.out_channels * self.conv_real.kernel_size[0] * self.conv_real.kernel_size[1]
        # 使用原 TF 中的均匀分布初始化，方差缩放系数 r=0.25
        r = 0.25
        high = (r * 2.0 / (fan_in + fan_out)) ** 0.5
        with torch.no_grad():
            self.conv_real.weight.uniform_(-high, high)
            self.conv_imag.weight.uniform_(-high, high)
            if self.conv_real.bias is not None:
                self.conv_real.bias.normal_(std=0.01)
            if self.conv_imag.bias is not None:
                self.conv_imag.bias.normal_(std=0.01)

    def forward(self, x):
        if torch.is_complex(x):
            x_real, x_imag = x.real, x.imag
        else:
            x_real, x_imag = x, torch.zeros_like(x)

        out_real = self.conv_real(x_real) - self.conv_imag(x_imag)
        out_imag = self.conv_real(x_imag) + self.conv_imag(x_real)
        return torch.complex(out_real, out_imag)


class ComplexBatchNorm2d(nn.Module):
    """
    复数批归一化（白化 + 去相关）。
    实现参考：Deep Complex Networks (Trabelsi et al., ICLR 2018)。

    修复内容：
        - 修复均值/协方差分量维度索引错误；
        - 将统计量 reshape 为 (1, C, 1) 以正确广播到 (B, C, N) 的张量。
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight_rr = nn.Parameter(torch.ones(num_features))
            self.weight_ii = nn.Parameter(torch.ones(num_features))
            self.weight_ri = nn.Parameter(torch.zeros(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features, 2))
        else:
            self.register_parameter('weight_rr', None)
            self.register_parameter('weight_ii', None)
            self.register_parameter('weight_ri', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, 2))
            self.register_buffer('running_cov', torch.ones(num_features, 3))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_cov', None)
            self.register_buffer('num_batches_tracked', None)

    def _normalize_complex(self, x_real, x_imag,
                           mean_r, mean_i, V_rr, V_ii, V_ri):
        """
        白化步骤，所有分量已正确广播（形状 (1, C, 1) 或可广播）。
        """
        # 减去均值
        x_real = x_real - mean_r
        x_imag = x_imag - mean_i

        # 协方差矩阵的逆平方根元素计算
        det = V_rr * V_ii - V_ri * V_ri
        # 增加 clamp 避免浮点精度误差导致 det < 0 从而产生 NaN
        s = torch.sqrt(torch.clamp(det, min=0.0)) + self.eps
        trace = V_rr + V_ii
        t = torch.sqrt(trace + 2.0 * s) * s

        # 白化
        out_real = ((V_ii + s) * x_real - V_ri * x_imag) / t
        out_imag = (-V_ri * x_real + (V_rr + s) * x_imag) / t
        return out_real, out_imag

    def forward(self, input):
        """
        input: 复数张量 (B, C, H, W)
        """
        x_real = input.real
        x_imag = input.imag
        B, C, H, W = x_real.shape

        if self.training:
            N = B * H * W
            x_real_flat = x_real.permute(0, 2, 3, 1).reshape(N, C)
            x_imag_flat = x_imag.permute(0, 2, 3, 1).reshape(N, C)

            # 计算统计量
            mean_r = x_real_flat.mean(dim=0)   # (C,)
            mean_i = x_imag_flat.mean(dim=0)
            mean = torch.stack([mean_r, mean_i], dim=-1)   # (C, 2)

            x_real_centered = x_real_flat - mean_r
            x_imag_centered = x_imag_flat - mean_i

            V_rr = (x_real_centered * x_real_centered).sum(dim=0) / (N - 1)
            V_ii = (x_imag_centered * x_imag_centered).sum(dim=0) / (N - 1)
            V_ri = (x_real_centered * x_imag_centered).sum(dim=0) / (N - 1)

            cov = torch.stack([V_rr, V_ii, V_ri], dim=-1)   # (C, 3)

            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean.mul_(1 - self.momentum).add_(mean * self.momentum)
                    self.running_cov.mul_(1 - self.momentum).add_(cov * self.momentum)
                self.num_batches_tracked.add_(1)
        else:
            mean = self.running_mean      # (C, 2)
            cov = self.running_cov        # (C, 3)
            mean_r = mean[:, 0]           # (C,)
            mean_i = mean[:, 1]
            V_rr = cov[:, 0]
            V_ii = cov[:, 1]
            V_ri = cov[:, 2]

        # 将输入展平为 (B, C, N) 以进行逐点白化
        x_real_flat = x_real.view(B, C, -1)
        x_imag_flat = x_imag.view(B, C, -1)

        # 统计量重塑为 (1, C, 1) —— 与 (B, C, N) 兼容广播
        mean_r = mean_r.view(1, C, 1)
        mean_i = mean_i.view(1, C, 1)
        V_rr = V_rr.view(1, C, 1)
        V_ii = V_ii.view(1, C, 1)
        V_ri = V_ri.view(1, C, 1)

        # 白化
        out_real, out_imag = self._normalize_complex(
            x_real_flat, x_imag_flat,
            mean_r, mean_i, V_rr, V_ii, V_ri
        )

        # 仿射变换（参数也重塑为 (1, C, 1)）
        if self.affine:
            gamma_rr = self.weight_rr.view(1, C, 1)
            gamma_ii = self.weight_ii.view(1, C, 1)
            gamma_ri = self.weight_ri.view(1, C, 1)
            beta_r = self.bias[:, 0].view(1, C, 1)
            beta_i = self.bias[:, 1].view(1, C, 1)

            y_real = gamma_rr * out_real + gamma_ri * out_imag + beta_r
            y_imag = gamma_ri * out_real + gamma_ii * out_imag + beta_i
        else:
            y_real, y_imag = out_real, out_imag

        # 恢复空间形状
        y_real = y_real.view(B, C, H, W)
        y_imag = y_imag.view(B, C, H, W)
        return torch.complex(y_real, y_imag)


class SimpleComplexBatchNorm2d(nn.Module):
    """
    简化版复数批归一化（仅供对照实验，不推荐用于最终模型）。
    分别对实部和虚部执行独立的 2D BatchNorm，忽略了二者的相关性。
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.bn_real = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_imag = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        return torch.complex(self.bn_real(x.real), self.bn_imag(x.imag))


class ComplexReLU(nn.Module):
    """
    modReLU 激活函数。
    modReLU(z) = ReLU(|z| + b) * (z / |z|)
    b 为可学习偏置，每通道独立。
    """
    def __init__(self, num_features):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        abs_z = torch.abs(x)
        bias = self.b.view(1, -1, 1, 1)
        mask = F.relu(abs_z + bias)
        return mask * (x / (abs_z + 1e-8))


def complex_xavier_uniform_(weight_real, weight_imag, fan_in, fan_out):
    """
    复数 Xavier 初始化（in‑place）。
    幅度: sqrt(2 / (fan_in + fan_out))
    相位: 均匀分布 U[-π, π]
    """
    amplitude = math.sqrt(2.0 / (fan_in + fan_out))
    with torch.no_grad():
        phase = torch.empty_like(weight_real).uniform_(-math.pi, math.pi)
        weight_real.copy_(amplitude * torch.cos(phase))
        weight_imag.copy_(amplitude * torch.sin(phase))