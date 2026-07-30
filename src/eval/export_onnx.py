# src/eval/export_onnx.py
"""
导出 ONNX 模型脚本（替代原 export_for_tensorrt）。
将训练好的 ComplexHoloNet（及可选的 ComplexDDPMNet）导出为 ONNX 格式，
由于 ONNX 不支持复数张量，故将输出拆分为实部与虚部两个实数张量。
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.holonet import ComplexHoloNet
from src.models.ddpm_net import ComplexDDPMNet
from typing import Optional


def export_onnx(
    holonet: nn.Module,
    ddpm_net: Optional[nn.Module] = None,
    output_path: str = "model.onnx",
    res_h: int = 1080,
    res_w: int = 1920,
    pad: int = 0,
    input_dim: int = 4,  # 4 for single RGBD
    device: str = "cpu"
):
    """
    将复数模型导出为 ONNX 格式，输出为复数场的实部（real_out）和虚部（imag_out）。
    若提供 DDPM 网络，则将 holonet 输出传入 ddpm_net，再拆分实部虚部。
    动态轴支持 batch 维度。
    """
    # 模型设为评估模式
    holonet.eval()
    if ddpm_net is not None:
        ddpm_net.eval()

    # 构造 dummy 输入（NCHW）
    dummy_rgbd = torch.randn(1, input_dim, res_h, res_w, device=device)

    class ExportModel(nn.Module):
        """封装 holonet 和可选的 ddpm_net，输出实部与虚部实数张量。"""
        def __init__(self, holonet, ddpm_net=None, pad=0):
            super().__init__()
            self.holonet = holonet
            self.ddpm_net = ddpm_net
            self.pad = pad

        def forward(self, x):
            # 复数主网络输出复数场 (B, 3, H, W)
            complex_field = self.holonet(x)

            # 添加 padding（对复数张量的实部与虚部分别填充）
            if self.pad > 0:
                real_padded = F.pad(complex_field.real, (self.pad, self.pad, self.pad, self.pad),
                                    mode='constant', value=0.0)
                imag_padded = F.pad(complex_field.imag, (self.pad, self.pad, self.pad, self.pad),
                                    mode='constant', value=0.0)
                complex_field = torch.complex(real_padded, imag_padded)

            if self.ddpm_net is not None:
                # DDPM 网络接受复数场，输出复数场
                complex_field = self.ddpm_net(complex_field)

            # 拆分为实部和虚部实数张量，便于 ONNX 导出
            return complex_field.real, complex_field.imag

    export_model = ExportModel(holonet, ddpm_net, pad).to(device)

    # 定义动态轴
    dynamic_axes = {
        "input": {0: "batch_size"},
        "real_out": {0: "batch_size"},
        "imag_out": {0: "batch_size"}
    }

    # 导出
    torch.onnx.export(
        export_model,
        dummy_rgbd,
        output_path,
        input_names=["input"],
        output_names=["real_out", "imag_out"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
        verbose=False
    )
    print(f"ONNX model exported to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ComplexHoloNet to ONNX")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to PyTorch checkpoint")
    parser.add_argument("--ddpm-ckpt-path", type=str, default=None, help="Separate DDPM checkpoint")
    parser.add_argument("--output", type=str, default="inference_graph_v2.onnx")
    parser.add_argument("--res-h", type=int, default=1080)
    parser.add_argument("--res-w", type=int, default=1920)
    parser.add_argument("--pad", type=int, default=0, help="Padding size")
    parser.add_argument("--input-dim", type=int, default=4, help="Input channels (4 for single RGBD)")
    parser.add_argument("--activate-ddpm", action="store_true")
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--num-filters-per-layer", type=int, default=24)

    args = parser.parse_args()

    device = "cpu"  # 导出一般用 CPU 即可
    holonet = ComplexHoloNet(
        input_dim=args.input_dim,
        num_layers=args.num_layers,
        num_filters_per_layer=args.num_filters_per_layer
    ).to(device)

    # 加载权重
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        holonet.load_state_dict(checkpoint['model_state_dict'])
    elif 'holonet_state_dict' in checkpoint:
        holonet.load_state_dict(checkpoint['holonet_state_dict'])
    else:
        holonet.load_state_dict(checkpoint)

    # 可选的 DDPM 网络（复数版本）
    ddpm_net = None
    if args.activate_ddpm:
        ddpm_net = ComplexDDPMNet(
            input_dim=3, output_dim=3, num_layers=8, num_filters_per_layer=8
        ).to(device)
        if args.ddpm_ckpt_path:
            ddpm_checkpoint = torch.load(args.ddpm_ckpt_path, map_location=device)
            if 'ddpm_net_state_dict' in ddpm_checkpoint:
                ddpm_net.load_state_dict(ddpm_checkpoint['ddpm_net_state_dict'])
            else:
                ddpm_net.load_state_dict(ddpm_checkpoint)
        else:
            # 尝试从主 checkpoint 加载
            if 'ddpm_net_state_dict' in checkpoint:
                ddpm_net.load_state_dict(checkpoint['ddpm_net_state_dict'])
            else:
                raise ValueError("DDPM weights not found. Provide --ddpm-ckpt-path.")

    export_onnx(
        holonet, ddpm_net, args.output,
        res_h=args.res_h, res_w=args.res_w,
        pad=args.pad, input_dim=args.input_dim
    )