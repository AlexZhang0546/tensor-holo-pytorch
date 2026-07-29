"""
主 U-Net 风格的全息预测网络（TensorHolographyNet）。
替代原 TensorFlow 中的 _build_model_vars 和 _build_graph。
"""

import torch
import torch.nn as nn
import numpy as np

# 引用前序步骤定义的 interleave/deinterleave 和权重初始化工具
from src.data.transforms import interleave_nonnative, deinterleave_nonnative
from src.utils.weight_init import init_weights_real


class TensorHolographyNet(nn.Module):
    """
    主全息预测网络。
    输入:  (B, input_dim, H, W)   # NCHW 格式，多通道 RGBD 或 LDI
    输出:  amp (B, output_dim//2, H, W)  范围 [0, sqrt(2)]
           phs (B, output_dim//2, H, W)  范围 [0, 1]
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int = 6,
                 num_layers: int = 30,
                 num_filters_per_layer: int = 24,
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
        self.num_filters = num_filters_per_layer
        self.interleave_rate = interleave_rate
        self.filter_width = filter_width
        self.bias_stddev = bias_stddev
        self.weight_var_scale = weight_var_scale

        # 计算每一层的输入/输出通道数
        # 与原始代码逻辑完全一致（考虑 interleave_rate 带来的通道倍增）
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

        # 构建各层（卷积 + BN + 激活，最后一层不同激活）
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            act = output_activation() if is_last else activation()

            conv = nn.Conv2d(in_dim_list[i], out_dim_list[i],
                             kernel_size=filter_width,
                             padding=filter_width // 2,  # SAME padding
                             bias=True)
            bn = nn.BatchNorm2d(out_dim_list[i])

            # 每一层封装为 Sequential
            self.layers.append(nn.Sequential(conv, bn, act))

        # 显式初始化权重
        self.reset_parameters()

    def reset_parameters(self):
        """
        使用与原始 TF 代码一致的 Xavier-uniform 初始化权重，
        bias 使用截断正态初始化（std = bias_stddev）。
        """
        for i, layer in enumerate(self.layers):
            conv = layer[0]  # nn.Conv2d
            bn = layer[1]    # nn.BatchNorm2d

            # 卷积权重初始化
            fan_in = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
            fan_out = conv.out_channels * conv.kernel_size[0] * conv.kernel_size[1]
            init_weights_real(conv.weight, fan_in, fan_out, r=self.weight_var_scale)

            # 偏置初始化：截断正态
            nn.init.trunc_normal_(conv.bias, std=self.bias_stddev)

            # BatchNorm 参数保持默认：gamma=1, beta=0（与原始代码一致，BN 未特殊初始化）

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: 输入张量 (B, input_dim, H, W)，值范围任意，内部会减去 0.5 进行重归一化。
        Returns:
            amp: 振幅 (B, output_dim//2, H, W)，范围 [0, sqrt(2)]
            phs: 相位 (B, output_dim//2, H, W)，范围 [0, 1]
        """
        # 1. 输入重归一化：减去 0.5（原代码中 renormalize_input=True）
        x = x - 0.5

        # 2. 可选 interleave（空间→通道）
        if self.interleave_rate > 1:
            x = interleave_nonnative(self.interleave_rate, x)
        x_in = x  # 保存用于最后一层的跳跃连接

        # 3. 逐层计算，并维护前两层的输出用于 skip connection
        prev_outputs = []  # 存储每一层的输出

        for i in range(self.num_layers):
            # --- 构建该层的输入（prev）---
            if i == 0:
                prev = x_in
            elif i < 3 or (i % 2 == 0):
                # i=1,2,4,6,... 直接取上一层输出
                prev = prev_outputs[i-1]
            else:
                # i=3,5,7,... 加上前两层（skip connection）
                prev = prev_outputs[i-1] + prev_outputs[i-2]

            # 最后一层还需拼接原始输入（沿通道维）
            if i == self.num_layers - 1:
                prev = torch.cat([prev, x_in], dim=1)  # NCHW，通道维是 dim=1

            # 通过当前层
            out = self.layers[i](prev)
            prev_outputs.append(out)

        # 最后一层输出
        field = prev_outputs[-1]

        # 4. 可选 deinterleave（通道→空间）
        if self.interleave_rate > 1:
            field = deinterleave_nonnative(self.interleave_rate, field)
        # 此时 field 形状: (B, output_dim, H, W)

        # 5. 分离振幅和相位，并缩放到指定范围
        amp_raw = field[:, :self.output_dim // 2, :, :]   # 前半通道为振幅
        phs_raw = field[:, self.output_dim // 2:, :, :]   # 后半通道为相位

        amp = amp_raw * (np.sqrt(0.5)) + np.sqrt(0.5)    # 缩放到 [0, sqrt(2)]
        phs = phs_raw * 0.5 + 0.5                         # 缩放到 [0, 1]

        return amp, phs