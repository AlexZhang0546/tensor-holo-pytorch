# src/models/complex_layers.py
"""
复值神经网络基础模块：
  - ComplexConv2d: 复数二维卷积（两个实卷积）
  - ComplexBatchNorm2d: 完整复数批归一化（白化+去相关）
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
        # 两个独立的实卷积
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, bias)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, bias)
        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.conv_real.in_channels * self.conv_real.kernel_size[0] * self.conv_real.kernel_size[1]
        fan_out = self.conv_real.out_channels * self.conv_real.kernel_size[0] * self.conv_real.kernel_size[1]
        complex_xavier_uniform_(self.conv_real.weight, self.conv_imag.weight, fan_in, fan_out)
        if self.conv_real.bias is not None and self.conv_imag.bias is not None:
            nn.init.zeros_(self.conv_real.bias)
            nn.init.zeros_(self.conv_imag.bias)

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

    将每个复数特征视为 2D 实向量 (real, imag)：
        1. 减去均值（实部均值、虚部均值）；
        2. 计算 2×2 协方差矩阵 V；
        3. 用 V 的逆平方根矩阵进行白化（去相关 + 归一化方差）；
        4. 乘以可学习的缩放矩阵（γ_rr, γ_ri, γ_ii）并加上可学习的平移（β_r, β_i）。
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
            # 缩放参数：γ_rr, γ_ii, γ_ri (复数的 "协方差" 仅需三个独立参数)
            self.weight_rr = nn.Parameter(torch.ones(num_features))
            self.weight_ii = nn.Parameter(torch.ones(num_features))
            self.weight_ri = nn.Parameter(torch.zeros(num_features))
            # 平移参数：β_r, β_i
            self.bias = nn.Parameter(torch.zeros(num_features, 2))
        else:
            self.register_parameter('weight_rr', None)
            self.register_parameter('weight_ii', None)
            self.register_parameter('weight_ri', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            # 运行时统计量：均值 (C, 2) 和协方差 (C, 3) 的滑动平均
            self.register_buffer('running_mean', torch.zeros(num_features, 2))
            self.register_buffer('running_cov', torch.ones(num_features, 3))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_cov', None)
            self.register_buffer('num_batches_tracked', None)

    def _normalize_complex(self, x_real, x_imag, mean, cov):
        """
        给定均值 (N, C, 2) 和协方差 (N, C, 3)，进行白化。
        cov 的最后一维按顺序存放：V_rr, V_ii, V_ri。
        返回白化后的 (real, imag)。
        """
        # 减去均值
        x_real = x_real - mean[..., 0]
        x_imag = x_imag - mean[..., 1]

        # 协方差矩阵 V = [[V_rr, V_ri],
        #                  [V_ri, V_ii]]
        # 其逆平方根矩阵通过特征值分解计算，见论文公式。
        # 为了方便，直接计算 2x2 矩阵的平方根逆。
        # 该算法对每个样本独立计算（训练模式），需要向量化。
        # 这里采用论文中的方法：先计算 trace 和 determinant。
        V_rr = cov[..., 0]
        V_ii = cov[..., 1]
        V_ri = cov[..., 2]

        # 归一化因子 s = sqrt(V_rr * V_ii - V_ri^2) + eps
        det = V_rr * V_ii - V_ri * V_ri
        s = torch.sqrt(det) + self.eps

        # 逆平方根矩阵 R = [[(V_ii + s) / t, -V_ri / t],
        #                      [ -V_ri / t, (V_rr + s) / t]]
        # 其中 t = sqrt((V_rr + V_ii + 2*s) * det)
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
            # 计算当前 batch 的均值和协方差
            # 将实部和虚部展平到 (B*H*W, C)
            N = B * H * W
            x_real_flat = x_real.permute(0, 2, 3, 1).reshape(N, C)   # (N, C)
            x_imag_flat = x_imag.permute(0, 2, 3, 1).reshape(N, C)

            # 均值 (C, 2)
            mean_r = x_real_flat.mean(dim=0)   # (C,)
            mean_i = x_imag_flat.mean(dim=0)   # (C,)
            mean = torch.stack([mean_r, mean_i], dim=-1)   # (C, 2)

            # 中心化
            x_real_centered = x_real_flat - mean_r   # (N, C)
            x_imag_centered = x_imag_flat - mean_i

            # 协方差矩阵元素（C, 3）
            V_rr = (x_real_centered * x_real_centered).sum(dim=0) / (N - 1)  # 用样本方差
            V_ii = (x_imag_centered * x_imag_centered).sum(dim=0) / (N - 1)
            V_ri = (x_real_centered * x_imag_centered).sum(dim=0) / (N - 1)

            cov = torch.stack([V_rr, V_ii, V_ri], dim=-1)   # (C, 3)

            if self.track_running_stats:
                # 更新滑动平均
                with torch.no_grad():
                    self.running_mean.mul_(1 - self.momentum).add_(mean * self.momentum)
                    self.running_cov.mul_(1 - self.momentum).add_(cov * self.momentum)
                self.num_batches_tracked.add_(1)
        else:
            # 推理时使用累计统计量
            mean = self.running_mean      # (C, 2)
            cov = self.running_cov        # (C, 3)

        # 扩展均值和协方差到 (B, 1, 1, C, ...) 以便广播
        # 我们将输入重塑为 (B, C, -1)，做完后再恢复
        # 为了高效，保持 (B, C, H*W) 的形状
        x_real_flat = x_real.view(B, C, -1)   # (B, C, N_spatial)
        x_imag_flat = x_imag.view(B, C, -1)

        # 扩展 mean, cov 维度以匹配
        mean_expanded = mean.unsqueeze(0).unsqueeze(-1)   # (1, C, 2, 1)
        cov_expanded = cov.unsqueeze(0).unsqueeze(-1)     # (1, C, 3, 1)

        # 白化
        out_real, out_imag = self._normalize_complex(
            x_real_flat, x_imag_flat,
            mean_expanded, cov_expanded
        )   # 每个的形状是 (B, C, N_spatial)

        # 仿射变换（可学习的缩放与平移）
        if self.affine:
            # 缩放矩阵参数 shape: (C,)
            # 重建缩放矩阵 [[γ_rr, γ_ri], [γ_ri, γ_ii]]
            # 白化后向量为 (x̂_r, x̂_i)，需应用缩放：
            # [y_r, y_i]^T = [[γ_rr, γ_ri], [γ_ri, γ_ii]] * [x̂_r, x̂_i]^T + [β_r, β_i]^T
            # 注意：γ_ri 控制实部与虚部之间的混合。
            gamma_rr = self.weight_rr.view(1, -1, 1)   # (1, C, 1)
            gamma_ii = self.weight_ii.view(1, -1, 1)
            gamma_ri = self.weight_ri.view(1, -1, 1)
            beta_r = self.bias[:, 0].view(1, -1, 1)    # (1, C, 1)
            beta_i = self.bias[:, 1].view(1, -1, 1)

            # 应用线性变换
            y_real = gamma_rr * out_real + gamma_ri * out_imag + beta_r
            y_imag = gamma_ri * out_real + gamma_ii * out_imag + beta_i
        else:
            y_real, y_imag = out_real, out_imag

        # 恢复空间形状
        y_real = y_real.view(B, C, H, W)
        y_imag = y_imag.view(B, C, H, W)

        return torch.complex(y_real, y_imag)


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