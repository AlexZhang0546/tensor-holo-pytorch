# src/eval/validate.py
"""
批量验证脚本：对应原 TensorFlow 项目中的 validate_stage_1 和 validate_stage_2 方法。
加载指定阶段的模型权重，在验证集上计算振幅图的 SSIM/PSNR，并输出统计结果。
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys

# 添加项目根目录到路径，确保 src 模块可导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.holonet import TensorHolographyNet
from src.models.ddpm_net import DDPMNet
from src.data.dataset import create_dataloader
from src.optics.propagation import propagator_factory
from src.optics.complex_utils import compl_val, compl_exp
from src.optics.dpm import aadpm
from src.optics.aperture import filter_phs_only
from src.utils.metrics import compute_ssim, compute_psnr  # 如果单独提取，也可直接调用内部实现


# ---- 辅助函数：构建传播算子（无 padding / 有 padding） ----
def build_propagator(res_h, res_w, pitch, wavelengths, double_pad=True):
    """构建角谱传播算子，用于无 padding 的验证 (stage1)"""
    return propagator_factory(
        input_shape=(res_h, res_w),
        pitch=pitch,
        wavelengths=wavelengths,
        method='as',
        double_pad=double_pad
    )

def build_propagator_padded(res_h, res_w, pitch, wavelengths, pad, double_pad=True):
    """构建带 padding 的角谱传播算子，用于 stage2 验证"""
    return propagator_factory(
        input_shape=(res_h + 2 * pad, res_w + 2 * pad),
        pitch=pitch,
        wavelengths=wavelengths,
        method='as',
        double_pad=double_pad
    )


# ---- 加载模型权重 ----
def load_model_weights(model, checkpoint_path, device):
    """从指定 .pth 文件加载模型权重（支持主网络和 DDPM 网络）"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 兼容 stage1 和 stage2 的保存格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'holonet_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['holonet_state_dict'])
    else:
        # 直接尝试加载全部 key（可能直接是 state_dict）
        model.load_state_dict(checkpoint)
    print(f"Model weights loaded from {checkpoint_path}")


def validate_stage1(holonet, val_loader, hologram_params, device):
    """
    Stage1 验证：只使用主网络输出，比较振幅图的 SSIM 和 PSNR。
    返回平均 SSIM、PSNR 以及各自的标准差和极值。
    """
    holonet.eval()
    ssim_list = []
    psnr_list = []

    with torch.no_grad():
        for batch in val_loader:
            rgbd = batch['rgbd'].to(device)         # (B, C, H, W)
            amp_gt = batch['amp_4'].to(device)      # (B, 3, H, W)
            phs_gt = batch['phs_4'].to(device)

            amp_out, phs_out = holonet(rgbd)        # 预测振幅/相位

            # 计算每张图的 SSIM/PSNR（data_range 与原 TF 一致，使用 1.0）
            for i in range(amp_out.size(0)):
                # 注意：原代码振幅范围 [0, sqrt(2)]，但 SSIM 使用了 max_val=1.0，此处保持一致
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
    Stage2 验证：执行完整的 DDPM 流程，包括 padding、深度偏移、DDPM 校正、
    双相位编码（AA-DPM）和物理孔径滤波，最终比较滤波后振幅与原始目标振幅的 SSIM/PSNR。
    若 ddpm_net 为 None，则 bypass DDPM，直接使用偏移后的场进行编码和滤波。
    """
    holonet.eval()
    if ddpm_net is not None:
        ddpm_net.eval()

    # 参数提取
    pad = training_params.get('padding', 0)
    depth_shift = training_params.get('depth_shift', 0.0)
    res_h, res_w = hologram_params['res_h'], hologram_params['res_w']
    wavelengths_np = hologram_params['wavelengths']
    wavelengths = torch.tensor(wavelengths_np, device=device).view(1, -1, 1, 1)

    # 构建带 padding 的传播算子
    propagator_pad = build_propagator_padded(res_h, res_w, hologram_params['pitch'],
                                             wavelengths_np, pad)
    propagator_pad = propagator_pad.to(device)

    ssim_list = []
    psnr_list = []

    with torch.no_grad():
        for batch in val_loader:
            rgbd = batch['rgbd'].to(device)
            amp_gt = batch['amp_4'].to(device)
            phs_gt = batch['phs_4'].to(device)

            # 目标全息图也需要 padding（与原代码一致）
            amp_gt_padded = torch.nn.functional.pad(
                amp_gt, (pad, pad, pad, pad), mode='constant', value=0.0)
            phs_gt_padded = torch.nn.functional.pad(
                phs_gt, (pad, pad, pad, pad), mode='constant', value=0.5)

            # 1. 主网络输出
            amp_mid, phs_mid = holonet(rgbd)

            # 2. padding 并深度偏移
            amp_mid_padded = torch.nn.functional.pad(
                amp_mid, (pad, pad, pad, pad), mode='constant', value=0.0)
            phs_mid_padded = torch.nn.functional.pad(
                phs_mid, (pad, pad, pad, pad), mode='constant', value=0.5)

            holo_mid = compl_val(amp_mid_padded, (phs_mid_padded - 0.5) * 2.0 * np.pi)
            holo_shifted = propagator_pad(holo_mid, depth_shift) * compl_exp(
                -2 * np.pi * depth_shift / wavelengths)
            amp_shifted = torch.abs(holo_shifted)
            phs_shifted = torch.angle(holo_shifted) / (2.0 * np.pi) + 0.5

            # 3. DDPM 校正（若存在）
            if ddpm_net is not None:
                amp_phs = torch.cat([amp_shifted, phs_shifted], dim=1)
                amp_altered, phs_altered = ddpm_net(amp_phs)
            else:
                # bypass：直接使用偏移后的结果
                amp_altered, phs_altered = amp_shifted, phs_shifted

            # 4. 双相位编码 (AA-DPM) —— 与原 stage2 验证流程一致
            holo_altered = compl_val(amp_altered, (phs_altered - 0.5) * 2.0 * np.pi)
            phs_only, amp_max = aadpm(
                holo_altered,
                propagator=propagator_pad,
                depth_shift=0.0,            # 已在目标平面
                adaptive_phs_shift=False,
                batch=rgbd.size(0),
                num_channels=3,
                res_h=holo_altered.shape[2],
                res_w=holo_altered.shape[3],
                sigma=0.0,                  # 与原代码一致，sigma=0
                kernel_width=3,
                phs_max=None,
                amp_max=None,
                clamp=True,
                normalize=False,
                wavelength=wavelengths_np
            )

            # 5. 物理孔径滤波，并传回原始深度（取消偏移）
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

            # 6. 与目标振幅（已 padding）比较 SSIM/PSNR
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
    parser = argparse.ArgumentParser(description='Validate trained Holonet / DDPM model')
    parser.add_argument('--mode', type=str, required=True, choices=['stage1', 'stage2'],
                        help='Validation stage')
    parser.add_argument('--model-name', default='full_loss', type=str, help='Model name')
    parser.add_argument('--dataset-res', default=192, type=int, help='Dataset resolution')
    parser.add_argument('--pitch', default=0.008, type=float, help='Pixel pitch in mm')
    parser.add_argument('--num-layers', default=30, type=int, help='Number of layers')
    parser.add_argument('--num-filters-per-layer', default=24, type=int,
                        help='Number of filters per layer')
    parser.add_argument('--batch', default=2, type=int, help='Batch size')
    parser.add_argument('--padding', default=0, type=int, help='Padding for stage2')
    parser.add_argument('--depth-shift', default=12.0, type=float, help='Depth shift for stage2')
    parser.add_argument('--activate-ddpm', action='store_true',
                        help='Use DDPM network in stage2')
    parser.add_argument('--bypass-ddpm-network', action='store_true',
                        help='Bypass DDPM network in stage2 (only main net)')
    parser.add_argument('--ckpt-path', type=str, required=True,
                        help='Path to the checkpoint file (.pth)')
    # 可选：DDPM 网络单独的 checkpoint（若不与主网络保存在一起）
    parser.add_argument('--ddpm-ckpt-path', type=str, default=None,
                        help='Separate DDPM checkpoint (only needed if not in main ckpt)')
    args = parser.parse_args()

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 参数准备
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
        "num_top_depth_for_img_loss": 15,  # 这些参数在验证时不使用，但保留完整性
        "num_random_depth_for_img_loss": 5,
        "depth_dependent_weight_scale": 0.35,
        "num_hist_bins": 200
    }

    # 构建主网络
    holonet = TensorHolographyNet(
        input_dim=4,                 # 单层 LDI
        output_dim=6,
        num_layers=args.num_layers,
        num_filters_per_layer=args.num_filters_per_layer,
        interleave_rate=1,
        filter_width=3,
        bias_stddev=0.01,
        weight_var_scale=0.25
    ).to(device)
    load_model_weights(holonet, args.ckpt_path, device)

    # 若 stage2 且需要 DDPM
    ddpm_net = None
    if args.mode == 'stage2' and args.activate_ddpm and not args.bypass_ddpm_network:
        ddpm_net = DDPMNet(
            input_dim=6,
            output_dim=6,
            num_layers=8,
            num_filters_per_layer=8,
            interleave_rate=1,
            filter_width=3,
            bias_stddev=0.01,
            weight_var_scale=0.25
        ).to(device)
        if args.ddpm_ckpt_path:
            load_model_weights(ddpm_net, args.ddpm_ckpt_path, device)
        else:
            # 若未单独指定，尝试从同一 checkpoint 加载（主 ckpt 中可能包含 ddpm_net_state_dict）
            checkpoint = torch.load(args.ckpt_path, map_location=device)
            if 'ddpm_net_state_dict' in checkpoint:
                ddpm_net.load_state_dict(checkpoint['ddpm_net_state_dict'])
                print("DDPM weights loaded from the same checkpoint.")
            else:
                raise ValueError("DDPM weights not found in checkpoint. Provide --ddpm-ckpt-path.")

    # 数据加载
    cur_dir = os.getcwd()  # 项目根目录
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

    # 执行验证
    if args.mode == 'stage1':
        validate_stage1(holonet, val_loader, hologram_params, device)
    else:
        validate_stage2(holonet, ddpm_net, val_loader, hologram_params, training_params, device)


if __name__ == '__main__':
    main()