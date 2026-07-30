# src/eval/validate.py
"""
批量验证脚本（复数网络适配版）。
使用 ComplexHoloNet 和 ComplexDDPMNet，所有内部光场均为复数。
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.holonet import ComplexHoloNet
from src.models.ddpm_net import ComplexDDPMNet          # 复数 DDPM 网络
from src.data.dataset import create_dataloader
from src.optics.propagation import propagator_factory
from src.optics.complex_utils import compl_exp
from src.optics.dpm import aadpm
from src.optics.aperture import filter_phs_only
from src.utils.metrics import compute_ssim, compute_psnr


# ---- 传播算子构建 ----
def build_propagator(res_h, res_w, pitch, wavelengths, double_pad=True):
    return propagator_factory(
        input_shape=(res_h, res_w),
        pitch=pitch,
        wavelengths=wavelengths,
        method='as',
        double_pad=double_pad
    )

def build_propagator_padded(res_h, res_w, pitch, wavelengths, pad, double_pad=True):
    return propagator_factory(
        input_shape=(res_h + 2 * pad, res_w + 2 * pad),
        pitch=pitch,
        wavelengths=wavelengths,
        method='as',
        double_pad=double_pad
    )


# ---- 加载模型权重 ----
def load_model_weights(model, checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'holonet_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['holonet_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"Model weights loaded from {checkpoint_path}")


def validate_stage1(holonet, val_loader, hologram_params, device):
    """
    Stage1 验证：主网络输出复数场，直接比较振幅。
    """
    holonet.eval()
    ssim_list = []
    psnr_list = []

    with torch.no_grad():
        for batch in val_loader:
            rgbd = batch['rgbd'].to(device)
            amp_gt = batch['amp_4'].to(device)

            holo_out = holonet(rgbd)          # (B, 3, H, W) 复数
            amp_out = holo_out.abs()

            for i in range(amp_out.size(0)):
                ssim_val = compute_ssim(amp_out[i], amp_gt[i], data_range=1.0)
                psnr_val = compute_psnr(amp_out[i], amp_gt[i], data_range=1.0)
                ssim_list.append(ssim_val.item())
                psnr_list.append(psnr_val.item())

    ssim_arr = np.array(ssim_list)
    psnr_arr = np.array(psnr_list)

    print("\n===== Stage 1 Validation Results =====")
    print(f"SSIM Amp: mean={ssim_arr.mean():.4f}, std={ssim_arr.std():.4f}, "
          f"max={ssim_arr.max():.4f}, min={ssim_arr.min():.4f}")
    print(f"PSNR Amp: mean={psnr_arr.mean():.2f}, std={psnr_arr.std():.2f}, "
          f"max={psnr_arr.max():.2f}, min={psnr_arr.min():.2f}")
    return ssim_arr.mean(), psnr_arr.mean()


def validate_stage2(holonet, ddpm_net, val_loader, hologram_params, training_params, device):
    """
    Stage2 验证：复数 HoloNet -> padding -> 深度偏移 -> 复数 DDPM -> DPM -> 孔径滤波。
    ddpm_net 应为 ComplexDDPMNet 实例，输入输出均为复数场。
    """
    holonet.eval()
    if ddpm_net is not None:
        ddpm_net.eval()

    pad = training_params.get('padding', 0)
    depth_shift = training_params.get('depth_shift', 0.0)
    res_h, res_w = hologram_params['res_h'], hologram_params['res_w']
    wavelengths_np = hologram_params['wavelengths']
    wavelengths = torch.tensor(wavelengths_np, device=device).view(1, -1, 1, 1)

    propagator_pad = build_propagator_padded(res_h, res_w, hologram_params['pitch'],
                                             wavelengths_np, pad).to(device)

    ssim_list = []
    psnr_list = []

    with torch.no_grad():
        for batch in val_loader:
            rgbd = batch['rgbd'].to(device)
            amp_gt = batch['amp_4'].to(device)
            # 目标振幅 padding
            amp_gt_padded = F.pad(amp_gt, (pad, pad, pad, pad), mode='constant', value=0.0)

            # 1. 主网络输出复数场
            holo_mid = holonet(rgbd)                   # (B, 3, H, W)

            # 2. padding
            if pad > 0:
                holo_mid = F.pad(holo_mid, (pad, pad, pad, pad), mode='constant', value=0.0)

            # 3. 深度偏移
            holo_shifted = propagator_pad(holo_mid, depth_shift) * compl_exp(
                -2 * np.pi * depth_shift / wavelengths)

            # 4. 复数 DDPM 校正（如果存在），直接输入复数场，输出复数场
            if ddpm_net is not None:
                holo_altered = ddpm_net(holo_shifted)   # (B, 3, H, W) 复数
            else:
                holo_altered = holo_shifted

            # 5. 双相位编码 (AA-DPM)
            phs_only, amp_max = aadpm(
                holo_altered,
                propagator=propagator_pad,
                depth_shift=0.0,
                adaptive_phs_shift=False,
                batch=rgbd.size(0),
                num_channels=3,
                res_h=holo_altered.shape[2],
                res_w=holo_altered.shape[3],
                sigma=0.0,
                kernel_width=3,
                phs_max=None,
                amp_max=None,
                clamp=True,
                normalize=False,
                wavelength=wavelengths_np
            )

            # 6. 物理孔径滤波并反向传播
            amp_final, _ = filter_phs_only(
                phs_only,
                unnormalize_input=False,
                normalize_output=False,
                propagator=propagator_pad,
                depth_shift=-depth_shift,
                batch=rgbd.size(0),
                num_channels=3,
                res_h=holo_altered.shape[2],
                res_w=holo_altered.shape[3],
                radius=None,
                phs_max=None,
                amp_max=amp_max,
                wavelength=wavelengths_np
            )

            # 7. 与目标振幅比较
            for i in range(amp_final.size(0)):
                ssim_val = compute_ssim(amp_final[i], amp_gt_padded[i], data_range=1.0)
                psnr_val = compute_psnr(amp_final[i], amp_gt_padded[i], data_range=1.0)
                ssim_list.append(ssim_val.item())
                psnr_list.append(psnr_val.item())

    ssim_arr = np.array(ssim_list)
    psnr_arr = np.array(psnr_list)

    print("\n===== Stage 2 Validation Results =====")
    print(f"SSIM Amp: mean={ssim_arr.mean():.4f}, std={ssim_arr.std():.4f}, "
          f"max={ssim_arr.max():.4f}, min={ssim_arr.min():.4f}")
    print(f"PSNR Amp: mean={psnr_arr.mean():.2f}, std={psnr_arr.std():.2f}, "
          f"max={psnr_arr.max():.2f}, min={psnr_arr.min():.2f}")
    return ssim_arr.mean(), psnr_arr.mean()


def main():
    parser = argparse.ArgumentParser(description='Validate Complex Holonet / DDPM model')
    parser.add_argument('--val-mode', type=str, required=True, choices=['stage1', 'stage2'],
                        help='Validation stage')
    parser.add_argument('--model-name', default='full_loss', type=str)
    parser.add_argument('--dataset-res', default=192, type=int)
    parser.add_argument('--pitch', default=0.008, type=float)
    parser.add_argument('--num-layers', default=30, type=int)
    parser.add_argument('--num-filters-per-layer', default=24, type=int)
    parser.add_argument('--batch', default=2, type=int)
    parser.add_argument('--padding', default=0, type=int)
    parser.add_argument('--depth-shift', default=12.0, type=float)
    parser.add_argument('--activate-ddpm', action='store_true')
    parser.add_argument('--bypass-ddpm-network', action='store_true')
    parser.add_argument('--ckpt-path', type=str, required=True)
    parser.add_argument('--ddpm-ckpt-path', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    hologram_params = {
        "wavelengths": np.array([0.000450, 0.000520, 0.000638]),
        "pitch": args.pitch,
        "res_h": args.dataset_res,
        "res_w": args.dataset_res,
        "depth_base": -3,
        "depth_scale": 6,
        "double_pad": True
    }

    training_params = {
        "batch": args.batch,
        "padding": args.padding,
        "depth_shift": args.depth_shift,
    }

    # 主网络：ComplexHoloNet
    holonet = ComplexHoloNet(
        input_dim=4,
        num_layers=args.num_layers,
        num_filters_per_layer=args.num_filters_per_layer,
        interleave_rate=1,
        filter_width=3,
        bias_stddev=0.01,
        weight_var_scale=0.25
    ).to(device)
    load_model_weights(holonet, args.ckpt_path, device)

    # 复数 DDPM 网络
    ddpm_net = None
    if args.mode == 'stage2' and args.activate_ddpm and not args.bypass_ddpm_network:
        ddpm_net = ComplexDDPMNet(
            input_dim=3, output_dim=3,          # 复数通道为 3（RGB）
            num_layers=8, num_filters_per_layer=8,
            interleave_rate=1, filter_width=3,
            bias_stddev=0.01, weight_var_scale=0.25
        ).to(device)
        if args.ddpm_ckpt_path:
            load_model_weights(ddpm_net, args.ddpm_ckpt_path, device)
        else:
            checkpoint = torch.load(args.ckpt_path, map_location=device)
            if 'ddpm_net_state_dict' in checkpoint:
                ddpm_net.load_state_dict(checkpoint['ddpm_net_state_dict'])
                print("DDPM weights loaded from main checkpoint.")
            else:
                raise ValueError("DDPM weights not found. Provide --ddpm-ckpt-path.")

    # 数据加载
    cur_dir = os.getcwd()
    val_tfrecord = os.path.join(cur_dir, "data", f"validate_{args.dataset_res}_v2",
                                "validate_04.tfrecord")
    labels = ["amp_4", "phs_4", "img_0", "depth_0"]
    val_dataset_params = {
        "res_h": args.dataset_res,
        "res_w": args.dataset_res,
        "sample_count": 100
    }
    val_loader = create_dataloader(
        tfrecord_path=val_tfrecord,
        dataset_params=val_dataset_params,
        labels=labels,
        active_max_ldi_layer=0,
        batch_size=args.batch,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    if args.val_mode == 'stage1':
        validate_stage1(holonet, val_loader, hologram_params, device)
    else:
        validate_stage2(holonet, ddpm_net, val_loader, hologram_params, training_params, device)

if __name__ == '__main__':
    main()