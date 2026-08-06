# src/eval/evaluate.py
"""
单张图像推理脚本：对应原 TensorFlow 项目中 TensorHolographyModel.evaluate 方法。
支持 LDI 格式输入（由 active_max_ldi_layer 控制层数），
运行主网络（复数输出） + ComplexDDPM（可选） + 双相位编码（AA/BL/Maimone） + 物理光圈滤波，
最终保存相位图、振幅图及各通道相位图等，输出与 TF 版本像素对齐。

本版本适配复数主网络与复数 DDPM 网络，去掉 amp/phs 分离与重组步骤。
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.factory import build_main_net
from src.models.ddpm_net import ComplexDDPMNet          # 复数 DDPM 网络
from src.optics.propagation import propagator_factory
from src.optics.complex_utils import compl_val, compl_exp
from src.optics.dpm import aadpm, bldpm, dpm_maimone
from src.optics.aperture import filter_phs_only


def load_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'holonet_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['holonet_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded weights from {checkpoint_path}")


def build_propagator(res_h, res_w, pitch, wavelengths, pad=0, double_pad=True):
    h = res_h + 2 * pad
    w = res_w + 2 * pad
    return propagator_factory(
        input_shape=(h, w),
        pitch=pitch,
        wavelengths=wavelengths,
        method='as',
        double_pad=double_pad
    )


def process_rgbd(rgb_path, depth_path, res_h, res_w, active_max_ldi_layer=0):
    """
    读取 RGB 和深度图像，并根据 LDI 层数构造输入张量。
    返回形状为 (1, C, H, W) 的 float32 张量，值域 [0,1]。
    """
    rgb = cv2.resize(cv2.imread(rgb_path, cv2.IMREAD_COLOR)[:, :, ::-1], (res_w, res_h),
                     interpolation=cv2.INTER_CUBIC)
    rgb = np.transpose(rgb, (2, 0, 1)) / 255.0

    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image: {depth_path}")
    depth = cv2.resize(depth, (res_w, res_h), interpolation=cv2.INTER_CUBIC)
    depth = depth.astype(np.float32) / 255.0
    depth = depth[None, :, :]

    if active_max_ldi_layer == 0:
        rgbd_np = np.concatenate([rgb, depth], axis=0)
    else:
        rgbd_parts = [rgb, depth]
        base_rgb, ext = os.path.splitext(rgb_path)
        base_depth, _ = os.path.splitext(depth_path)
        for i in range(1, active_max_ldi_layer + 1):
            rgb_i_path = f"{base_rgb}_{i}{ext}"
            depth_i_path = f"{base_depth}_{i}{ext}"
            rgb_i = cv2.resize(cv2.imread(rgb_i_path, cv2.IMREAD_COLOR)[:, :, ::-1],
                               (res_w, res_h), interpolation=cv2.INTER_CUBIC)
            rgb_i = np.transpose(rgb_i, (2, 0, 1)) / 255.0
            depth_i = cv2.imread(depth_i_path, cv2.IMREAD_GRAYSCALE)
            if depth_i is None:
                raise FileNotFoundError(f"Could not read depth image: {depth_i_path}")
            depth_i = cv2.resize(depth_i, (res_w, res_h), interpolation=cv2.INTER_CUBIC)
            depth_i = depth_i.astype(np.float32) / 255.0
            depth_i = depth_i[None, :, :]
            rgbd_parts.append(rgb_i)
            rgbd_parts.append(depth_i)
        rgbd_np = np.concatenate(rgbd_parts, axis=0)

    rgbd_tensor = torch.from_numpy(rgbd_np).unsqueeze(0).float()
    return rgbd_tensor


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    hologram_params = {
        'wavelengths': np.array([0.000450, 0.000520, 0.000638]),
        'pitch': args.pitch,
        'res_h': args.eval_res_h,
        'res_w': args.eval_res_w,
        'depth_base': -3,
        'depth_scale': 6,
        'double_pad': True
    }

    input_dim = 4 * (args.active_max_ldi_layer + 1)
    # 复数主网络（按 arch 构建）
    holonet = build_main_net(
        arch=getattr(args, 'model_arch', 'holonet'),
        input_dim=input_dim,
        num_layers=args.num_layers,
        num_filters_per_layer=args.num_filters_per_layer,
        unet_depth=getattr(args, 'unet_depth', 3),
        unet_base_filters=getattr(args, 'unet_base_filters', 24),
        unet_attention=getattr(args, 'unet_attention', False),
    ).to(device)

    load_model(holonet, args.ckpt_path, device)

    # 复数 DDPM 网络（替换原实数 DDPM）
    ddpm_net = None
    if args.activate_ddpm and not args.bypass_ddpm_network:
        ddpm_net = ComplexDDPMNet(
            input_dim=3,               # 复数 RGB 三通道
            output_dim=3,
            num_layers=8,
            num_filters_per_layer=8,
            interleave_rate=1,
            filter_width=3,
            bias_stddev=0.01,
            weight_var_scale=0.25
        ).to(device)
        if args.ddpm_ckpt_path:
            ddpm_checkpoint = torch.load(args.ddpm_ckpt_path, map_location=device)
            if 'ddpm_net_state_dict' in ddpm_checkpoint:
                ddpm_net.load_state_dict(ddpm_checkpoint['ddpm_net_state_dict'])
                print("DDPM weights loaded from separate checkpoint.")
            else:
                # 兼容旧格式，直接作为 state_dict 加载
                ddpm_net.load_state_dict(ddpm_checkpoint)
        else:
            checkpoint = torch.load(args.ckpt_path, map_location=device)
            if 'ddpm_net_state_dict' in checkpoint:
                ddpm_net.load_state_dict(checkpoint['ddpm_net_state_dict'])
                print("DDPM weights loaded from main checkpoint.")
            else:
                raise ValueError("DDPM weights not found. Provide --ddpm-ckpt-path.")

    holonet.eval()
    if ddpm_net is not None:
        ddpm_net.eval()

    rgbd = process_rgbd(args.eval_rgb_path, args.eval_depth_path,
                        args.eval_res_h, args.eval_res_w,
                        active_max_ldi_layer=args.active_max_ldi_layer)
    rgbd = rgbd.to(device)

    pad = args.padding
    res_h, res_w = hologram_params['res_h'], hologram_params['res_w']
    propagator = build_propagator(res_h, res_w, hologram_params['pitch'],
                                  hologram_params['wavelengths'], pad, double_pad=True)
    propagator = propagator.to(device)
    wavelengths_tensor = torch.tensor(hologram_params['wavelengths'], device=device, dtype=torch.float32).view(1, -1, 1, 1)

    with torch.no_grad():
        # 主网络直接输出复数光场
        complex_field = holonet(rgbd)

    if pad > 0:
        complex_field = F.pad(complex_field, (pad, pad, pad, pad), mode='constant', value=0.0)

    # 深度偏移（保持复数）
    depth_shift = args.eval_depth_shift
    holo_out = complex_field
    holo_shifted = propagator(holo_out, depth_shift) * compl_exp(
        -2 * np.pi * depth_shift / wavelengths_tensor).to(torch.complex64)

    # 复数 DDPM 校正：直接输入复数场，输出校正后的复数场
    if ddpm_net is not None:
        holo_altered = ddpm_net(holo_shifted)        # (B, 3, H, W) 复数
    else:
        holo_altered = holo_shifted

    # 后续 DPM 与光圈滤波（均接受复数场）
    phs_max = [args.phs_max * np.pi] * 3

    if args.use_maimone_dpm:
        phs_only, amp_max = dpm_maimone(
            holo_altered,
            propagator=propagator,
            depth_shift=0.0,
            adaptive_phs_shift=args.adaptive_phs_shift,
            batch=1, num_channels=3,
            res_h=holo_altered.shape[2], res_w=holo_altered.shape[3],
            axis=3,
            phs_max=phs_max,
            amp_max=None,
            clamp=True,
            normalize=True,
            wavelength=hologram_params['wavelengths']
        )
    elif args.use_bldpm:
        phs_only, amp_max = bldpm(
            holo_altered,
            propagator=propagator,
            depth_shift=0.0,
            adaptive_phs_shift=args.adaptive_phs_shift,
            batch=1, num_channels=3,
            res_h=holo_altered.shape[2], res_w=holo_altered.shape[3],
            k=args.k,
            phs_max=phs_max,
            amp_max=None,
            clamp=True,
            normalize=True,
            wavelength=hologram_params['wavelengths']
        )
    else:
        phs_only, amp_max = aadpm(
            holo_altered,
            propagator=propagator,
            depth_shift=0.0,
            adaptive_phs_shift=args.adaptive_phs_shift,
            batch=1, num_channels=3,
            res_h=holo_altered.shape[2], res_w=holo_altered.shape[3],
            sigma=args.gaussian_sigma,
            kernel_width=args.gaussian_width,
            phs_max=phs_max,
            amp_max=None,
            clamp=True,
            normalize=True,
            wavelength=hologram_params['wavelengths']
        )

    amp_final, _ = filter_phs_only(
        phs_only,
        unnormalize_input=True,
        normalize_output=True,
        propagator=propagator,
        depth_shift=-depth_shift,
        batch=1, num_channels=3,
        res_h=holo_altered.shape[2], res_w=holo_altered.shape[3],
        radius=None,
        phs_max=phs_max,
        amp_max=amp_max,
        wavelength=hologram_params['wavelengths']
    )

    # 保存主网络输出的振幅和相位（从复数场提取）
    amp_out = torch.abs(complex_field)
    phs_out = torch.angle(complex_field) / (2.0 * np.pi) + 0.5

    amp_out_np = amp_out.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    phs_out_np = phs_out.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    phs_only_np = phs_only.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    amp_final_np = amp_final.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)

    os.makedirs(args.eval_output_path, exist_ok=True)

    cv2.imwrite(os.path.join(args.eval_output_path, "amp.png"),
                np.clip(amp_out_np * 255.0, 0.0, 255.0).astype(np.uint8))
    cv2.imwrite(os.path.join(args.eval_output_path, "phs.png"),
                np.clip(phs_out_np * 255.0, 0.0, 255.0).astype(np.uint8))
    cv2.imwrite(os.path.join(args.eval_output_path, "blue.png"),
                np.clip(phs_only_np[:, :, 0] * 255.0, 0.0, 255.0).astype(np.uint8))
    cv2.imwrite(os.path.join(args.eval_output_path, "green.png"),
                np.clip(phs_only_np[:, :, 1] * 255.0, 0.0, 255.0).astype(np.uint8))
    cv2.imwrite(os.path.join(args.eval_output_path, "red.png"),
                np.clip(phs_only_np[:, :, 2] * 255.0, 0.0, 255.0).astype(np.uint8))
    cv2.imwrite(os.path.join(args.eval_output_path, "amp_filtered.png"),
                np.clip(amp_final_np * 255.0, 0.0, 255.0).astype(np.uint8))

    print(f"Results saved to {args.eval_output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single-image hologram evaluation')
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument('--ddpm-ckpt-path', type=str, default=None)
    parser.add_argument('--activate-ddpm', action='store_true')
    parser.add_argument('--bypass-ddpm-network', action='store_true')
    parser.add_argument('--num-layers', type=int, default=30)
    parser.add_argument('--num-filters-per-layer', type=int, default=24)
    parser.add_argument('--active-max-ldi-layer', type=int, default=0)
    parser.add_argument('--model-arch', default='holonet',
                        choices=['holonet', 'unet'],
                        help='Main network architecture')
    parser.add_argument('--unet-depth', type=int, default=3,
                        help='ComplexUNet downsample levels')
    parser.add_argument('--unet-base-filters', type=int, default=24,
                        help='ComplexUNet base filters (shallowest level)')
    parser.add_argument('--unet-attention', action='store_true',
                        help='Enable bottleneck self-attention in ComplexUNet')
    parser.add_argument('--eval-res-h', type=int, default=1080)
    parser.add_argument('--eval-res-w', type=int, default=1920)
    parser.add_argument('--eval-rgb-path', type=str, required=True)
    parser.add_argument('--eval-depth-path', type=str, required=True)
    parser.add_argument('--eval-output-path', type=str, required=True)
    parser.add_argument('--eval-depth-shift', type=float, default=0.0)
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--use-maimone-dpm', action='store_true')
    parser.add_argument('--use-bldpm', action='store_true')
    parser.add_argument('--adaptive-phs-shift', action='store_true')
    parser.add_argument('--gaussian-sigma', type=float, default=0.0)
    parser.add_argument('--gaussian-width', type=int, default=3)
    parser.add_argument('--phs-max', type=float, default=2.0)
    parser.add_argument('--k', type=float, default=1.0)
    parser.add_argument('--pitch', type=float, default=0.008)
    args = parser.parse_args()
    evaluate(args)
