# src/train/stage2.py
"""
阶段二训练脚本：DDPM 网络校正与联合微调（复数版本）。
适配复数主网络 ComplexHoloNet 与复数 DDPM 网络 ComplexDDPMNet，
直接处理复数全息场，移除 compl_val 构造步骤。

修正：
  - 双相位编码(aadpm)和光圈滤波(filter_phs_only)必须传入正确的phs_max参数，
    并开启归一化/反归一化流程，否则相位数值越界导致训练崩溃。
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.holonet import ComplexHoloNet
from src.models.ddpm_net import ComplexDDPMNet
from src.data.dataset import THDataset, create_dataloader
from src.optics.propagation import propagator_factory
from src.optics.complex_utils import compl_val, compl_exp
from src.optics.dpm import aadpm
from src.optics.aperture import filter_phs_only
from src.losses.focal_stack import compute_focal_stack_loss
from src.utils.metrics import compute_ssim, compute_psnr
from src.losses.ddpm_loss import compute_ddpm_phase_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Stage 2 Training (DDPM)')
    parser.add_argument('--model-name', default='full_loss', type=str)
    parser.add_argument('--dataset-res', default=192, type=int)
    parser.add_argument('--pitch', default=0.008, type=float)
    parser.add_argument('--num-layers', default=30, type=int)
    parser.add_argument('--num-filters-per-layer', default=24, type=int)
    parser.add_argument('--epoch-to-start-ddpm', default=3000, type=int,
                        help='(informational) epoch at which stage2 begins')
    parser.add_argument('--stage2-epochs', default=50, type=int,
                        help='Number of epochs for identity pretraining')
    parser.add_argument('--joint-epochs', default=200, type=int,
                        help='Number of epochs for joint training')
    parser.add_argument('--batch', default=2, type=int)
    parser.add_argument('--learning-rate', default=1e-4, type=float)
    parser.add_argument('--num-iter-per-test', default=500, type=int)
    parser.add_argument('--restore-stage1', action='store_true',
                        help='Load stage1 checkpoint before starting')
    parser.add_argument('--restore-stage2', action='store_true',
                        help='Resume stage2 training from checkpoint')
    parser.add_argument('--stage1-ckpt', default=None, type=str,
                        help='Path to stage1 checkpoint .pth file')
    parser.add_argument('--stage2-ckpt-dir', default=None, type=str,
                        help='Directory to save stage2 checkpoints')
    parser.add_argument('--activate-ddpm', action='store_true',
                        help='Use DDPM network')
    parser.add_argument('--bypass-ddpm-network', action='store_true',
                        help='If true, DDPM network is not used, only main network is trained.')
    parser.add_argument('--padding', default=0, type=int,
                        help='Padding for out-of-frame diffraction')
    parser.add_argument('--depth-shift', default=12.0, type=float,
                        help='Depth shift from midpoint hologram (mm)')
    return parser.parse_args()


# ----------------------------------------------------------------------
# 工具函数：复数张量 padding
# ----------------------------------------------------------------------
def complex_pad(x: torch.Tensor, pad: int, mode: str = 'constant',
                real_value: float = 0.0, imag_value: float = 0.0) -> torch.Tensor:
    """
    对复数张量 (..., H, W) 进行对称 padding。
    分别对实部、虚部进行填充，再组合为复数张量。
    """
    if pad == 0:
        return x
    pad_tuple = (pad, pad, pad, pad)  # 左,右,上,下
    real_padded = F.pad(x.real, pad_tuple, mode=mode, value=real_value)
    imag_padded = F.pad(x.imag, pad_tuple, mode=mode, value=imag_value)
    return torch.complex(real_padded, imag_padded)


# ----------------------------------------------------------------------
# 构建带 padding 的传播算子
# ----------------------------------------------------------------------
def build_propagator_padded(hologram_params, pad):
    res_h = hologram_params['res_h']
    res_w = hologram_params['res_w']
    return propagator_factory(
        input_shape=(res_h + 2 * pad, res_w + 2 * pad),
        pitch=hologram_params['pitch'],
        wavelengths=hologram_params['wavelengths'],
        method='as',
        double_pad=True
    )


# ----------------------------------------------------------------------
# 恒等预训练损失（复数版本）
# ----------------------------------------------------------------------
def identity_loss(holo_altered: torch.Tensor, holo_shifted: torch.Tensor, loss_fn):
    """
    计算 DDPM 网络的恒等损失，要求输出复数场与输入尽可能接近。
    """
    loss_real = loss_fn(holo_altered.real, holo_shifted.real)
    loss_imag = loss_fn(holo_altered.imag, holo_shifted.imag)
    total = loss_real + loss_imag

    amp_altered = torch.abs(holo_altered)
    amp_shifted = torch.abs(holo_shifted)
    ssim_amp = compute_ssim(amp_altered, amp_shifted, data_range=1.0)
    return total, ssim_amp

"""
# ----------------------------------------------------------------------
# 完整前向传播与损失计算（联合训练/验证复用，复数版本）
# ----------------------------------------------------------------------
def _run_stage2_forward(
    rgbd, amp_gt, phs_gt,
    holonet, ddpm_net,
    propagator_pad, depth_shift, pad,
    hologram_params, training_params, loss_params, loss_fn,
    bypass_ddpm=False
):
    执行 stage2 完整前向传播，所有网络输出均为复数场。
    
    修正说明：
      双相位编码(aadpm)和光圈滤波(filter_phs_only)现在接收正确的phs_max，
      并启用归一化/反归一化流程，避免相位值域越界。
    device = rgbd.device
    wavelengths_tensor = torch.tensor(hologram_params['wavelengths'], device=device).view(1, -1, 1, 1)

    # 1. 主网络输出复数全息场 (B, 3, H, W)
    holo_mid = holonet(rgbd)  # 复数

    # 2. padding
    holo_mid_padded = complex_pad(holo_mid, pad)

    # 3. 深度偏移
    holo_shifted = propagator_pad(holo_mid_padded, depth_shift) * \
                   compl_exp(-2 * np.pi * depth_shift / wavelengths_tensor)

    # 4. DDPM 校正或 bypass
    if ddpm_net is not None and not bypass_ddpm:
        # 复数 DDPM 网络直接接收复数场
        holo_altered = ddpm_net(holo_shifted)
        # 提取相位用于正则（转为 [0,1] 归一化相位）
        phs_for_reg = torch.angle(holo_altered) / (2.0 * np.pi) + 0.5
    else:
        holo_altered = holo_shifted
        phs_for_reg = None

    # ----- 关键修复：获取相位最大值并传递给后续模块 -----
    # 默认三个通道均为 2π，与 evaluate.py 保持一致
    phs_max = loss_params.get('phs_max', [2.0 * np.pi] * 3)

    # 5. 双相位编码 (AA-DPM)，现在传入 phs_max 并开启 normalize
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
        phs_max=None,#phs_max,
        amp_max=None,
        clamp=True,
        normalize=False,#True,
        wavelength=hologram_params['wavelengths']
    )

    # 6. 物理光圈滤波并反向传播回 midpoint
    amp_final, phs_final = filter_phs_only(
        phs_only,
        unnormalize_input=False,
        normalize_output=False,#True,
        propagator=propagator_pad,
        depth_shift=-depth_shift,
        batch=rgbd.size(0),
        num_channels=3,
        res_h=holo_altered.shape[2],
        res_w=holo_altered.shape[3],
        radius=None,
        phs_max=None,#phs_max,
        amp_max=amp_max,
        wavelength=hologram_params['wavelengths']
    )

    # 7. 目标全息图构造并 padding
    holo_gt = compl_val(amp_gt, (phs_gt - 0.5) * 2.0 * np.pi)
    holo_gt_padded = complex_pad(holo_gt, pad)

    # 8. 焦栈损失（将最终振幅/相位构造为复数场）
    holo_out = compl_val(amp_final, phs_final)
    fs_loss, fs_tv, ssim_img, psnr_img = compute_focal_stack_loss(
        holo_out, holo_gt_padded, rgbd, propagator_pad,
        hologram_params, training_params, loss_fn, pad=pad
    )

    # 9. 振幅图 SSIM/PSNR（裁剪有效区域后与目标振幅比较）
    amp_crop = amp_final[:, :, pad:pad + hologram_params['res_h'], pad:pad + hologram_params['res_w']]
    amp_gt_crop = amp_gt[:, :, pad:pad + hologram_params['res_h'], pad:pad + hologram_params['res_w']]
    ssim_amp = compute_ssim(amp_crop, amp_gt_crop, data_range=1.0)
    psnr_amp = compute_psnr(amp_crop, amp_gt_crop, data_range=1.0)

    # 10. 组合总损失
    w_fs = loss_params.get('weight_fs', 1.0)
    w_fs_tv = loss_params.get('weight_fs_tv', 1.0)
    total_loss = w_fs * fs_loss + w_fs_tv * fs_tv
    mean_loss = torch.tensor(0.0, device=device)
    std_loss = torch.tensor(0.0, device=device)

    if phs_for_reg is not None:
        std_loss, mean_loss = compute_ddpm_phase_loss(
            phs_for_reg, pad=pad,
            res_h=hologram_params['res_h'], res_w=hologram_params['res_w']
        )
        w_std = loss_params.get('weight_std', 0.02)
        w_mean = loss_params.get('weight_mean', 0.03)
        total_loss = total_loss + w_std * std_loss + w_mean * mean_loss

    return {
        'loss': total_loss,
        'fs_loss': fs_loss,
        'fs_tv': fs_tv,
        'ssim_amp': ssim_amp,
        'psnr_amp': psnr_amp,
        'ssim_img': ssim_img,
        'psnr_img': psnr_img,
        'mean_loss': mean_loss,
        'std_loss': std_loss
    }
"""
def _run_stage2_forward(
    rgbd, amp_gt, phs_gt,
    holonet, ddpm_net,
    propagator_pad, depth_shift, pad,
    hologram_params, training_params, loss_params, loss_fn,
    bypass_ddpm=False
):
    """
    诊断版本：跳过 DPM 与滤波，输出所有中间振幅的统计与 SSIM。
    """
    device = rgbd.device
    wavelengths_np = hologram_params['wavelengths']
    wavelengths_tensor = torch.tensor(wavelengths_np, device=device).view(1, -1, 1, 1)

    # ---------- 1. 主网络输出 ----------
    holo_mid = holonet(rgbd)                            # (B,3,H,W) complex
    amp_mid = holo_mid.abs()
    amp_gt_original = amp_gt                           # (B,3,H,W) real, range [0, sqrt(2)]

    # 主网络直接 SSIM（无 pad，无 shift）
    ssim_mid = compute_ssim(amp_mid, amp_gt_original, data_range=1.0)
    print(f"[DIAG] Mid-plane SSIM (no pad, no shift): {ssim_mid.item():.6f}")
    print(f"[DIAG] amp_mid mean: {amp_mid.mean().item():.6f}, max: {amp_mid.max().item():.6f}")
    print(f"[DIAG] amp_gt  mean: {amp_gt_original.mean().item():.6f}, max: {amp_gt_original.max().item():.6f}")

    # ---------- 2. Padding ----------
    holo_mid_padded = complex_pad(holo_mid, pad)        # 对复数场填 0（实部虚部均填0）
    amp_mid_padded = holo_mid_padded.abs()
    print(f"[DIAG] After pad ({pad}): amp_mid_padded mean: {amp_mid_padded.mean().item():.6f}, max: {amp_mid_padded.max().item():.6f}")

    # ---------- 3. 深度偏移 ----------
    # 传播 + 相位补偿（与原 TF 完全一致）
    holo_shifted = propagator_pad(holo_mid_padded, depth_shift) * \
                   compl_exp(-2 * np.pi * depth_shift / wavelengths_tensor)
    amp_shifted = holo_shifted.abs()
    print(f"[DIAG] After shift ({depth_shift} mm): amp_shifted mean: {amp_shifted.mean().item():.6f}, max: {amp_shifted.max().item():.6f}")

    # ---------- 4. 目标构造并 padding ----------
    holo_gt = compl_val(amp_gt_original, (phs_gt - 0.5) * 2.0 * np.pi)
    holo_gt_padded = complex_pad(holo_gt, pad)
    amp_gt_padded = holo_gt_padded.abs()
    print(f"[DIAG] Target amp (padded) mean: {amp_gt_padded.mean().item():.6f}, max: {amp_gt_padded.max().item():.6f}")

    # ---------- 5. 直接计算焦栈损失与振幅 SSIM（跳过 DDPM/DPM/滤波）----------
    # 注意：这里直接用 holo_shifted 作为最终输出
    fs_loss, fs_tv, ssim_img, psnr_img = compute_focal_stack_loss(
        holo_shifted, holo_gt_padded, rgbd, propagator_pad,
        hologram_params, training_params, loss_fn, pad=pad
    )

    # 裁剪有效区域后计算振幅 SSIM
    res_h = hologram_params['res_h']
    res_w = hologram_params['res_w']
    amp_crop = holo_shifted.abs()[:, :, pad:pad+res_h, pad:pad+res_w]
    amp_gt_crop = amp_gt_original[:, :, pad:pad+res_h, pad:pad+res_w] if pad > 0 else amp_gt_original
    ssim_amp = compute_ssim(amp_crop, amp_gt_crop, data_range=1.0)
    psnr_amp = compute_psnr(amp_crop, amp_gt_crop, data_range=1.0)

    print(f"[DIAG] Bypass DPM/Filter - SSIM amp: {ssim_amp.item():.6f}, PSNR amp: {psnr_amp.item():.2f}")
    print(f"[DIAG] FS loss: {fs_loss.item():.6f}, FS TV: {fs_tv.item():.6f}")

    # ---------- 6. 返回字典（保持接口兼容） ----------
    return {
        'loss': fs_loss,            # 临时只用 fs_loss
        'fs_loss': fs_loss,
        'fs_tv': fs_tv,
        'ssim_amp': ssim_amp,
        'psnr_amp': psnr_amp,
        'ssim_img': ssim_img,
        'psnr_img': psnr_img,
        'mean_loss': torch.tensor(0.0),
        'std_loss': torch.tensor(0.0)
    }

# ----------------------------------------------------------------------
# 保存与加载 checkpoint
# ----------------------------------------------------------------------
def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(filename):
    return torch.load(filename, map_location='cpu')


# ----------------------------------------------------------------------
# 主训练函数
# ----------------------------------------------------------------------
def train_stage2(
    model_params, ddpm_params, hologram_params, training_params, loss_params,
    train_loader, val_loader, device,
    stage1_ckpt_path, stage2_ckpt_dir,
    restore_stage2=False,
    bypass_ddpm=False
):
    # ---------- 构建模型（复数版本） ----------
    holonet = ComplexHoloNet(**model_params).to(device)
    ddpm_net = ComplexDDPMNet(**ddpm_params).to(device) if not bypass_ddpm else None

    # ---------- 加载 stage1 权重 ----------
    if stage1_ckpt_path and os.path.exists(stage1_ckpt_path):
        print(f"Loading stage1 checkpoint from {stage1_ckpt_path}")
        ckpt = load_checkpoint(stage1_ckpt_path)
        holonet.load_state_dict(ckpt['model_state_dict'])
    else:
        print("WARNING: No stage1 checkpoint provided or file not found. Starting with random holonet weights.")

    # ---------- 传播算子 ----------
    pad = training_params.get('padding', 0)
    propagator_pad = build_propagator_padded(hologram_params, pad).to(device)
    depth_shift = training_params.get('depth_shift', 0.0)

    # ---------- 损失函数 ----------
    loss_type = loss_params.get('loss_type', 'l1')
    loss_fn = F.l1_loss if loss_type == 'l1' else F.mse_loss

    # ---------- 优化器 ----------
    if ddpm_net is not None and not bypass_ddpm:
        optimizer_identity = optim.Adam(
            ddpm_net.parameters(),
            lr=training_params.get('learning_rate', 1e-4),
            betas=(0.9, 0.99), eps=1e-8
        )
    else:
        optimizer_identity = None

    # 联合优化器
    params_joint = list(holonet.parameters())
    if ddpm_net is not None and not bypass_ddpm:
        params_joint += list(ddpm_net.parameters())
    optimizer_joint = optim.Adam(
        params_joint,
        lr=training_params.get('learning_rate', 1e-4),
        betas=(0.9, 0.99), eps=1e-8
    )

    # ---------- 训练状态 ----------
    identity_epochs = training_params.get('identity_epochs', 50)
    joint_epochs = training_params.get('joint_epochs', 200)
    start_epoch_identity = 0
    start_epoch_joint = 0
    global_step = 0

    # 尝试恢复 stage2 训练
    if restore_stage2:
        identity_ckpt = os.path.join(stage2_ckpt_dir, 'stage2_identity_latest.pth')
        joint_ckpt = os.path.join(stage2_ckpt_dir, 'stage2_joint_latest.pth')

        if os.path.exists(joint_ckpt):
            print(f"Resuming joint training from {joint_ckpt}")
            ckpt = load_checkpoint(joint_ckpt)
            holonet.load_state_dict(ckpt['holonet_state_dict'])
            if ddpm_net is not None and 'ddpm_net_state_dict' in ckpt:
                ddpm_net.load_state_dict(ckpt['ddpm_net_state_dict'])
            optimizer_joint.load_state_dict(ckpt['optimizer_joint_state_dict'])
            start_epoch_joint = ckpt['epoch'] + 1
            global_step = ckpt.get('global_step', 0)
            start_epoch_identity = identity_epochs  # 跳过 identity 阶段
        elif os.path.exists(identity_ckpt) and ddpm_net is not None:
            print(f"Resuming identity pretraining from {identity_ckpt}")
            ckpt = load_checkpoint(identity_ckpt)
            ddpm_net.load_state_dict(ckpt['ddpm_net_state_dict'])
            optimizer_identity.load_state_dict(ckpt['optimizer_identity_state_dict'])
            start_epoch_identity = ckpt['epoch'] + 1
            global_step = ckpt.get('global_step', 0)
        else:
            print("No valid stage2 checkpoint found, starting from scratch.")

    # ---------- 阶段 2a: Identity 预训练 ----------
    if ddpm_net is not None and start_epoch_identity < identity_epochs:
        print("Starting identity pretraining...")
        holonet.eval()
        ddpm_net.train()
        for epoch in range(start_epoch_identity, identity_epochs):
            epoch_loss = 0.0
            epoch_ssim = 0.0
            for batch_idx, batch_data in enumerate(train_loader):
                rgbd = batch_data['rgbd'].to(device)
                with torch.no_grad():
                    holo_mid = holonet(rgbd)
                holo_mid_padded = complex_pad(holo_mid, pad)
                wavelengths_tensor = torch.tensor(hologram_params['wavelengths'], device=device).view(1, -1, 1, 1)
                holo_shifted = propagator_pad(holo_mid_padded, depth_shift) * \
                               compl_exp(-2 * np.pi * depth_shift / wavelengths_tensor)

                holo_altered = ddpm_net(holo_shifted)

                total_loss, ssim_amp = identity_loss(holo_altered, holo_shifted, loss_fn)

                optimizer_identity.zero_grad()
                total_loss.backward()
                optimizer_identity.step()

                epoch_loss += total_loss.item()
                epoch_ssim += ssim_amp.item()

                if (batch_idx + 1) % 50 == 0:
                    print(f"Identity Epoch {epoch:3d} | Step {batch_idx:4d} | Loss {total_loss.item():.6f} | SSIM {ssim_amp.item():.4f}")

                global_step += 1

            avg_loss = epoch_loss / len(train_loader)
            avg_ssim = epoch_ssim / len(train_loader)
            print(f"Identity Epoch {epoch:3d} finished | Avg Loss {avg_loss:.6f} | Avg SSIM {avg_ssim:.4f}")

            # 保存 identity checkpoint
            save_checkpoint({
                'epoch': epoch,
                'global_step': global_step,
                'ddpm_net_state_dict': ddpm_net.state_dict(),
                'optimizer_identity_state_dict': optimizer_identity.state_dict(),
            }, os.path.join(stage2_ckpt_dir, f'stage2_identity_epoch_{epoch:04d}.pth'))
            save_checkpoint({
                'epoch': epoch,
                'global_step': global_step,
                'ddpm_net_state_dict': ddpm_net.state_dict(),
                'optimizer_identity_state_dict': optimizer_identity.state_dict(),
            }, os.path.join(stage2_ckpt_dir, 'stage2_identity_latest.pth'))

        print("Identity pretraining completed.")

    # ---------- 阶段 2b: 联合训练 ----------
    holonet.train()
    if ddpm_net is not None:
        ddpm_net.train()

    print("Starting joint training...")
    for epoch in range(start_epoch_joint, joint_epochs):
        epoch_loss = 0.0
        epoch_fs = 0.0
        epoch_tv = 0.0
        epoch_ssim_amp = 0.0
        epoch_mean = 0.0
        epoch_std = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            rgbd = batch_data['rgbd'].to(device)
            amp_gt = batch_data['amp_4'].to(device)
            phs_gt = batch_data['phs_4'].to(device)

            # 使用修复后的前向函数
            outputs = _run_stage2_forward(
                rgbd, amp_gt, phs_gt,
                holonet, ddpm_net,
                propagator_pad, depth_shift, pad,
                hologram_params, training_params, loss_params, loss_fn,
                bypass_ddpm=bypass_ddpm
            )
            total_loss = outputs['loss']

            optimizer_joint.zero_grad()
            total_loss.backward()
            optimizer_joint.step()

            # 记录指标
            epoch_loss += total_loss.item()
            epoch_fs += outputs['fs_loss'].item()
            epoch_tv += outputs['fs_tv'].item()
            epoch_ssim_amp += outputs['ssim_amp'].item()
            if not bypass_ddpm and ddpm_net is not None:
                epoch_mean += outputs['mean_loss'].item()
                epoch_std += outputs['std_loss'].item()

            if (batch_idx + 1) % 50 == 0:
                log_str = (f"Joint Epoch {epoch:3d} | Step {batch_idx:4d} | Loss {total_loss.item():.6f} | "
                           f"FS {outputs['fs_loss'].item():.6f} | TV {outputs['fs_tv'].item():.6f} | "
                           f"SSIM_amp {outputs['ssim_amp'].item():.4f}")
                if not bypass_ddpm and ddpm_net is not None:
                    log_str += f" | Mean {outputs['mean_loss'].item():.6f} | Std {outputs['std_loss'].item():.6f}"
                print(log_str)

            global_step += 1

            # ---------- 定期验证 ----------
            if global_step % training_params.get('num_iter_per_test', 500) == 0 and global_step > 0:
                holonet.eval()
                if ddpm_net is not None:
                    ddpm_net.eval()

                val_stats = {'loss': 0.0, 'fs_loss': 0.0, 'fs_tv': 0.0,
                             'ssim_amp': 0.0, 'ssim_img': 0.0, 'mean_loss': 0.0, 'std_loss': 0.0}
                num_val_batches = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_rgbd = val_batch['rgbd'].to(device)
                        val_amp_gt = val_batch['amp_4'].to(device)
                        val_phs_gt = val_batch['phs_4'].to(device)

                        val_outputs = _run_stage2_forward(
                            val_rgbd, val_amp_gt, val_phs_gt,
                            holonet, ddpm_net,
                            propagator_pad, depth_shift, pad,
                            hologram_params, training_params, loss_params, loss_fn,
                            bypass_ddpm=bypass_ddpm
                        )
                        for k in val_stats:
                            val_stats[k] += val_outputs[k].item()
                        num_val_batches += 1

                for k in val_stats:
                    val_stats[k] /= num_val_batches

                print(f"--- Validation at step {global_step} ---")
                print(f"Loss: {val_stats['loss']:.6f} | FS: {val_stats['fs_loss']:.6f} | TV: {val_stats['fs_tv']:.6f} | "
                      f"SSIM_amp: {val_stats['ssim_amp']:.4f} | SSIM_img: {val_stats['ssim_img']:.4f}")
                if not bypass_ddpm and ddpm_net is not None:
                    print(f"Mean: {val_stats['mean_loss']:.6f} | Std: {val_stats['std_loss']:.6f}")

                holonet.train()
                if ddpm_net is not None:
                    ddpm_net.train()
                    ddpm_net.eval()

        # 每个 epoch 结束后保存 checkpoint
        save_dict = {
            'epoch': epoch,
            'global_step': global_step,
            'holonet_state_dict': holonet.state_dict(),
            'optimizer_joint_state_dict': optimizer_joint.state_dict(),
        }
        if ddpm_net is not None:
            save_dict['ddpm_net_state_dict'] = ddpm_net.state_dict()
        save_checkpoint(save_dict, os.path.join(stage2_ckpt_dir, f'stage2_joint_epoch_{epoch:04d}.pth'))
        save_checkpoint(save_dict, os.path.join(stage2_ckpt_dir, 'stage2_joint_latest.pth'))

        # 打印 epoch 平均指标
        avg_loss = epoch_loss / len(train_loader)
        avg_fs = epoch_fs / len(train_loader)
        avg_tv = epoch_tv / len(train_loader)
        avg_ssim = epoch_ssim_amp / len(train_loader)
        log_str = (f"Joint Epoch {epoch:3d} summary | Loss {avg_loss:.6f} | "
                   f"FS {avg_fs:.6f} | TV {avg_tv:.6f} | SSIM_amp {avg_ssim:.4f}")
        if not bypass_ddpm and ddpm_net is not None:
            avg_mean = epoch_mean / len(train_loader)
            avg_std = epoch_std / len(train_loader)
            log_str += f" | Mean {avg_mean:.6f} | Std {avg_std:.6f}"
        print(log_str)

    print("Stage 2 training completed.")


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 配置参数
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
        "identity_epochs": args.stage2_epochs,
        "joint_epochs": args.joint_epochs,
        "learning_rate": args.learning_rate,
        "decay_type": None,
        "decay_params": None,
        "num_iter_per_test": args.num_iter_per_test,
        "num_top_depth_for_img_loss": 15,
        "num_random_depth_for_img_loss": 5,
        "depth_dependent_weight_scale": 0.35,
        "num_hist_bins": 200,
        "depth_shift": args.depth_shift,
        "padding": args.padding,
    }

    # 复数主网络参数
    model_params = {
        "input_dim": 4,
        "num_layers": args.num_layers,
        "num_filters_per_layer": args.num_filters_per_layer,
        "interleave_rate": 1,
        "filter_width": 3,
        "bias_stddev": 0.01,
        "weight_var_scale": 0.25
    }

    # 复数 DDPM 网络参数
    ddpm_params = {
        "input_dim": 3,
        "output_dim": 3,
        "num_layers": 8,
        "num_filters_per_layer": 8,
        "interleave_rate": 1,
        "filter_width": 3,
        "bias_stddev": 0.01,
        "weight_var_scale": 0.25
    }

    # 损失参数（包含phs_max，传递给前向函数中的光学模块）
    loss_params = {
        "loss_type": "l1",
        "weight_holo": 1.0,
        "weight_fs": float(training_params["num_top_depth_for_img_loss"] + training_params["num_random_depth_for_img_loss"]),
        "weight_fs_tv": float(training_params["num_top_depth_for_img_loss"] + training_params["num_random_depth_for_img_loss"]),
        "weight_std": 0.02,
        "weight_mean": 0.03,
        "phs_max": [2 * np.pi, 2 * np.pi, 2 * np.pi]   # 关键：相位最大值
    }

    labels = ["amp_4", "phs_4", "img_0", "depth_0"]
    cur_dir = os.getcwd()
    train_tfrecord = os.path.join(cur_dir, "data", f"train_{args.dataset_res}_v2", "train_04.tfrecord")
    val_tfrecord   = os.path.join(cur_dir, "data", f"test_{args.dataset_res}_v2", "test_04.tfrecord")

    dataset_params = {"res_h": args.dataset_res, "res_w": args.dataset_res, "sample_count": 3800}
    val_dataset_params = {"res_h": args.dataset_res, "res_w": args.dataset_res, "sample_count": 100}

    train_loader = create_dataloader(
        train_tfrecord, dataset_params, labels,
        active_max_ldi_layer=0, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True
    )
    val_loader = create_dataloader(
        val_tfrecord, val_dataset_params, labels,
        active_max_ldi_layer=0, batch_size=args.batch, shuffle=False, num_workers=1, drop_last=True
    )

    if args.stage2_ckpt_dir is None:
        ckpt_name = (f"ckpt_{args.model_name}_pitch_{int(args.pitch*1000)}_"
                     f"layers_{args.num_layers}_filters_{args.num_filters_per_layer}_"
                     f"ddpm_{int(args.depth_shift)}")
        if args.bypass_ddpm_network:
            ckpt_name += "_bypass"
        stage2_ckpt_dir = os.path.join(cur_dir, "model", ckpt_name)
    else:
        stage2_ckpt_dir = args.stage2_ckpt_dir
    os.makedirs(stage2_ckpt_dir, exist_ok=True)

    train_stage2(
        model_params=model_params,
        ddpm_params=ddpm_params,
        hologram_params=hologram_params,
        training_params=training_params,
        loss_params=loss_params,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        stage1_ckpt_path=args.stage1_ckpt if args.restore_stage1 else None,
        stage2_ckpt_dir=stage2_ckpt_dir,
        restore_stage2=args.restore_stage2,
        bypass_ddpm=args.bypass_ddpm_network or not args.activate_ddpm
    )


if __name__ == '__main__':
    main()