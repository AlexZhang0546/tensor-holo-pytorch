# src/models/real_ddpm_net.py
"""Paper-faithful DDPM network (real-valued amp/phase CNN).

This mirrors the TF reference implementation in main_v2.py
(_build_graph with with_postfix=True, used by _setup_train_ddpm):

  - input : concat([amp_shift, phs_shift], dim=1) = 6 channels,
            normalized by subtracting 0.5;
  - body  : num_layers x 3x3 conv + BN + ReLU (last layer: BN + tanh),
            with the exact skip structure of the reference:
              i == 0          -> prev = x_in
              i < 3 or i%2==0 -> prev = layers[i-1]
              else            -> prev = layers[i-1] + prev_layers[i-2]
            and the final concat of the (normalized) input at the last layer;
  - output: field = tanh(...) (6 channels),
            amp = field[:, :3] * sqrt(0.5) + sqrt(0.5)   (in [0, sqrt(2)])
            phs = field[:, 3:] * 0.5 + 0.5               (in [0, 1])

forward(x) returns the reconstructed complex field so the surrounding
pipeline is unchanged; forward_amp_phase(x) returns (amp, phs) directly.

The BN layers use the TF "inference" semantics that main_v2.py actually
runs with (tf.layers.batch_normalization defaults to training=False):
frozen running stats (mean 0, var 1) + trainable per-channel gamma/beta.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.optics.complex_utils import compl_val
from src.models.ddpm_net import ComplexDDPMNet


class _TFInferenceBN(nn.Module):
    """BatchNorm with tf.layers.batch_normalization(training=False) semantics.

    The TF reference never updates the moving statistics (they stay at the
    defaults: mean 0, variance 1); only gamma/beta are trainable.
    eps matches the TF default 0.001.
    """

    def __init__(self, num_features, eps=0.001):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        scale = self.gamma / math.sqrt(1.0 + self.eps)
        return x * scale.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)


def _truncated_normal_(tensor, std=0.01, bound=2.0):
    """Approximate tf.truncated_normal: normal clipped to +/- bound*std."""
    with torch.no_grad():
        torch.nn.init.normal_(tensor, mean=0.0, std=std)
        tensor.clamp_(-bound * std, bound * std)
    return tensor


def _init_conv(conv, in_dim, out_dim, weight_var_scale, bias_stddev):
    """TF tf_init_weights xavier (uniform) init:
    high = sqrt(weight_var_scale * 2 / (in_dim + out_dim)).
    """
    high = (weight_var_scale * 2.0 / (in_dim + out_dim)) ** 0.5
    with torch.no_grad():
        conv.weight.uniform_(-high, high)
    if conv.bias is not None:
        _truncated_normal_(conv.bias, std=bias_stddev)


class RealAmpPhaseDDPMNet(nn.Module):
    """Paper-faithful real-valued DDPM (see module docstring)."""

    def __init__(self, input_dim=3, output_dim=3, num_layers=8,
                 num_filters_per_layer=8, interleave_rate=1, filter_width=3,
                 bias_stddev=0.01, weight_var_scale=0.25, bn_mode='tf'):
        super().__init__()
        if interleave_rate != 1:
            raise ValueError('RealAmpPhaseDDPMNet only supports '
                             'interleave_rate=1 (matches the paper)')
        self.num_layers = num_layers
        self.filter_width = filter_width
        self.bn_mode = bn_mode

        real_in = input_dim * 2          # amp + phs
        real_out = output_dim * 2        # amp + phs

        in_dims = []
        out_dims = []
        for i in range(num_layers):
            if i == 0:
                in_dims.append(real_in)
                out_dims.append(num_filters_per_layer)
            elif i == num_layers - 1:
                in_dims.append(num_filters_per_layer + real_in)
                out_dims.append(real_out)
            else:
                in_dims.append(num_filters_per_layer)
                out_dims.append(num_filters_per_layer)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            conv = nn.Conv2d(in_dims[i], out_dims[i], filter_width,
                             padding=filter_width // 2, bias=True)
            _init_conv(conv, in_dims[i], out_dims[i],
                       weight_var_scale, bias_stddev)
            self.convs.append(conv)
            if bn_mode == 'tf':
                self.bns.append(_TFInferenceBN(out_dims[i]))
            else:
                self.bns.append(nn.BatchNorm2d(out_dims[i]))

    def forward_amp_phase(self, x):
        """Map a complex field (B,3,H,W) to (amp, phs).

        amp is in [0, sqrt(2)], phs in [0, 1] (paper conventions).
        """
        amp = x.abs()
        phs = torch.angle(x) / (2.0 * np.pi) + 0.5
        feat = torch.cat([amp, phs], dim=1) - 0.5
        x_in = feat

        layer_out = []
        prev_list = []
        for i in range(self.num_layers):
            if i == 0:
                prev = x_in
            elif i < 3 or (i % 2 == 0):
                prev = layer_out[i - 1]
            else:
                prev = layer_out[i - 1] + prev_list[i - 2]

            if i == self.num_layers - 1:
                prev = torch.cat([prev, x_in], dim=1)

            out = self.bns[i](self.convs[i](prev))
            if i == self.num_layers - 1:
                out = torch.tanh(out)
            else:
                out = F.relu(out)
            layer_out.append(out)
            prev_list.append(prev)

        field = layer_out[-1]                 # (B, 6, H, W)
        amp_out = field[:, :3] * math.sqrt(0.5) + math.sqrt(0.5)
        phs_out = field[:, 3:] * 0.5 + 0.5
        return amp_out, phs_out

    def forward(self, x):
        """Return the reconstructed complex field (B,3,H,W)."""
        amp_out, phs_out = self.forward_amp_phase(x)
        return compl_val(amp_out, (phs_out - 0.5) * 2.0 * np.pi)


def build_ddpm_net(ddpm_params, arch='real', bn_mode='tf'):
    """Construct the DDPM network.

    arch='real'    -> paper-faithful amp/phase CNN (RealAmpPhaseDDPMNet)
    arch='complex' -> legacy complex-valued CNN (ComplexDDPMNet)
    """
    if arch == 'real':
        return RealAmpPhaseDDPMNet(**ddpm_params, bn_mode=bn_mode)
    return ComplexDDPMNet(**ddpm_params)
