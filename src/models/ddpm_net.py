"""
DDPM 校正网络（DDPMNet）。
用于对主网络输出经过 padding / 深度偏移后的振幅-相位拼接进行精细调整。
"""

import torch
import torch.nn as nn
import numpy as np

from src.data.transforms import interleave_nonnative, deinterleave_nonnative
from src.utils.weight_init import init_weights_real


class DDPMNet(nn.Module):
    """
    DDPM 校正网络。
    输入:  amplitude + phase 拼接 (B, 6, H, W)，值范围同主网络输出。
    输出:  调整后的 amplitude (B, 3, H, W) 范围 [0, sqrt(2)]
           phase       (B, 3, H, W) 范围 [0, 1]
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

        # 计算各层通道数（逻辑与主网络相同）
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
        """与主网络相同的初始化方式"""
        for i, layer in enumerate(self.layers):
            conv = layer[0]
            fan_in = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
            fan_out = conv.out_channels * conv.kernel_size[0] * conv.kernel_size[1]
            init_weights_real(conv.weight, fan_in, fan_out, r=self.weight_var_scale)
            nn.init.trunc_normal_(conv.bias, std=self.bias_stddev)
        # BN 保持默认

    def forward(self, amp_phs: torch.Tensor):
        """
        Args:
            amp_phs: 张量 (B, 6, H, W)，前半 3 通道为振幅，后半 3 通道为相位。
        Returns:
            amp: (B, 3, H, W) 归一化到 [0, sqrt(2)]
            phs: (B, 3, H, W) 归一化到 [0, 1]
        """
        # 1. 归一化：减去 0.5（与原代码 renormalize_input_ddpm=True 一致）
        x = amp_phs - 0.5

        # 2. interleave（若需要）
        if self.interleave_rate > 1:
            x = interleave_nonnative(self.interleave_rate, x)
        x_in = x

        # 3. 逐层计算，含跳跃连接
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

        # 4. deinterleave
        if self.interleave_rate > 1:
            field = deinterleave_nonnative(self.interleave_rate, field)

        # 5. 分离并缩放
        amp = field[:, :3, :, :] * np.sqrt(0.5) + np.sqrt(0.5)
        phs = field[:, 3:, :, :] * 0.5 + 0.5
        return amp, phs