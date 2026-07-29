# src/eval/evaluate.py
"""
单张图像推理脚本：对应原 TensorFlow 项目中 TensorHolographyModel.evaluate 方法。
支持 LDI 格式输入（由 active_max_ldi_layer 控制层数），
运行主网络 + DDPM（可选）+ 双相位编码（AA/BL/Maimone）+ 物理光圈滤波，
最终保存相位图、振幅图及各通道相位图等，输出与 TF 版本像素对齐。

修复：深度图统一以 cv2.IMREAD_GRAYSCALE 读取，避免因单通道/三通道差异导致索引错误。
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.holonet import TensorHolographyNet
from src.models.ddpm_net import DDPMNet
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

    修复说明：
        - 深度图改用 cv2.IMREAD_GRAYSCALE 强制灰度读取，直接得到 (H, W) 二维数组，
          避免原代码假设三通道而可能出现的维度错误。
        - RGB 仍保持 BGR->RGB 转换，与原 TF 版本一致。
    """
    # 读取第一层 RGB (保持 BGR -> RGB 转换)
    rgb = cv2.resize(cv2.imread(rgb_path, cv2.IMREAD_COLOR)[:, :, ::-1], (res_w, res_h),
                     interpolation=cv2.INTER_CUBIC)  # (H, W, 3)
    rgb = np.transpose(rgb, (2, 0, 1)) / 255.0       # (3, H, W)

    # 读取深度图：强制灰度模式，避免通道歧义
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image: {depth_path}")
    depth = cv2.resize(depth, (res_w, res_h), interpolation=cv2.INTER_CUBIC)  # (H, W)
    depth = depth.astype(np.float32) / 255.0          # 归一化到 [0, 1]
    depth = depth[None, :, :]                         # (1, H, W)

    if active_max_ldi_layer == 0:
        rgbd_np = np.concatenate([rgb, depth], axis=0)
    else:
        # 多层 LDI：依次加载后缀为 _0, _1, ... 的 RGB 和深度图
        rgbd_parts = [rgb, depth]
        base_rgb, ext = os.path.splitext(rgb_path)
        base_depth, _ = os.path.splitext(depth_path)
        for i in range(1, active_max_ldi_layer + 1):
            rgb_i_path = f"{base_rgb}_{i}{ext}"
            depth_i_path = f"{base_depth}_{i}{ext}"

            # 读取下一层 RGB
            rgb_i = cv2.resize(cv2.imread(rgb_i_path, cv2.IMREAD_COLOR)[:, :, ::-1],
                               (res_w, res_h), interpolation=cv2.INTER_CUBIC)
            rgb_i = np.transpose(rgb_i, (2, 0, 1)) / 255.0

            # 读取下一层深度 (灰度)
            depth_i = cv2.imread(depth_i_path, cv2.IMREAD_GRAYSCALE)
            if depth_i is None:
                raise FileNotFoundError(f"Could not read depth image: {depth_i_path}")
            depth_i = cv2.resize(depth_i, (res_w, res_h), interpolation=cv2.INTER_CUBIC)
            depth_i = depth_i.astype(np.float32) / 255.0
            depth_i = depth_i[None, :, :]

            rgbd_parts.append(rgb_i)
            rgbd_parts.append(depth_i)

        rgbd_np = np.concatenate(rgbd_parts, axis=0)  # (C, H, W)

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
    holonet = TensorHolographyNet(
        input_dim=input_dim,
        output_dim=6,
        num_layers=args.num_layers,
        num_filters_per_layer=args.num_filters_per_layer,
        interleave_rate=1,
        filter_width=3,
        bias_stddev=0.01,
        weight_var_scale=0.25
    ).to(device)

    load_model(holonet, args.ckpt_path, device)

    ddpm_net = None
    if args.activate_ddpm and not args.bypass_ddpm_network:
        ddpm_net = DDPMNet(
            input_dim=6, output_dim=6,
            num_layers=8, num_filters_per_layer=8,
            interleave_rate=1, filter_width=3,
            bias_stddev=0.01, weight_var_scale=0.25
        ).to(device)
        if args.ddpm_ckpt_path:
            load_model(ddpm_net, args.ddpm_ckpt_path, device)
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
    wavelengths_tensor = torch.tensor(hologram_params['wavelengths'], device=device).view(1, -1, 1, 1)

    with torch.no_grad():
        amp_out, phs_out = holonet(rgbd)

    if pad > 0:
        amp_out = F.pad(amp_out, (pad, pad, pad, pad), mode='constant', value=0.0)
        phs_out = F.pad(phs_out, (pad, pad, pad, pad), mode='constant', value=0.5)

    depth_shift = args.eval_depth_shift
    holo_out = compl_val(amp_out, (phs_out - 0.5) * 2.0 * np.pi)
    holo_shifted = propagator(holo_out, depth_shift) * compl_exp(
        -2 * np.pi * depth_shift / wavelengths_tensor)
    amp_shifted = torch.abs(holo_shifted)
    phs_shifted = torch.angle(holo_shifted) / (2.0 * np.pi) + 0.5

    if ddpm_net is not None:
        amp_phs = torch.cat([amp_shifted, phs_shifted], dim=1)
        amp_altered, phs_altered = ddpm_net(amp_phs)
    else:
        amp_altered, phs_altered = amp_shifted, phs_shifted

    holo_altered = compl_val(amp_altered, (phs_altered - 0.5) * 2.0 * np.pi)
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

    # 转换并翻转（与原 TF 代码一致）
    amp_out_np = amp_out.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)[::-1, :, :]
    phs_out_np = phs_out.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)[::-1, :, :]
    phs_only_np = phs_only.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)[::-1, :, :]
    amp_final_np = amp_final.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)[::-1, :, :]

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