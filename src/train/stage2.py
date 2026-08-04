# src/train/stage2.py
"""
阶段二训练脚本：在 Stage 1 主网络（ComplexHoloNet）的基础上，
添加复数 DDPM 网络进行端到端优化（包含身份预训练和联合微调）。
所有光场均使用复数张量，避免振幅/相位拆分。
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# 允许相对导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.holonet import ComplexHoloNet
from src.models.ddpm_net import ComplexDDPMNet          # 复数 DDPM 网络（假定已实现）
from src.data.dataset import create_dataloader
from src.optics.propagation import propagator_factory
from src.optics.complex_utils import compl_val, compl_exp
from src.optics.dpm import aadpm
from src.optics.aperture import filter_phs_only
from src.losses.complex_losses import complex_holo_loss, complex_ddpm_phase_loss
from src.losses.focal_stack import compute_focal_stack_loss
from src.utils.metrics import compute_ssim, compute_psnr


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def complex_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
    """对复数张量进行零填充（四周填充相同宽度）。"""
    if pad == 0:
        return x
    return F.pad(x, (pad, pad, pad, pad), mode='constant', value=0.0)


def build_propagator_padded(res_h, res_w, pitch, wavelengths, pad, double_pad=True):
    """构建带有额外 padding 的角谱传播算子。"""
    return propagator_factory(
        input_shape=(res_h + 2 * pad, res_w + 2 * pad),
        pitch=pitch,
        wavelengths=wavelengths,
        method='as',
        double_pad=double_pad,
    )


def identity_loss(pred_complex: torch.Tensor,
                  target_complex: torch.Tensor,
                  loss_type: str = 'l1') -> torch.Tensor:
    """
    DDPM 身份预训练损失：要求 DDPM 输出与输入一致。
    直接比较两个复数场。
    """
    return complex_holo_loss(pred_complex, target_complex,
                             loss_type=loss_type, method='magnitude_phase')


# ----------------------------------------------------------------------
# 前向传播核心
# ----------------------------------------------------------------------
def _run_stage2_forward(holonet, ddpm_net, rgbd, target_complex,
                        propagator_pad, hologram_params, training_params):
    """
    执行 Stage2 的完整前向过程：
    1. 主网络预测中间全息图
    2. padding 目标与预测
    3. 深度偏移（传播 + 补偿）
    4. 复数 DDPM 校正（可选）
    5. 双相位编码（AA-DPM）
    6. 物理孔径滤波与反向传播，得到最终复数场

    返回字典包含所有必要中间量。
    """
    pad = training_params.get('padding', 0)
    depth_shift = training_params.get('depth_shift', 0.0)
    wavelengths = hologram_params['wavelengths']
    device = rgbd.device
    wlen_tensor = torch.tensor(wavelengths, device=device, dtype=torch.float32).view(1, -1, 1, 1)

    # 1. 主网络预测（复数，无归一化）
    holo_mid = holonet(rgbd)

    # 2. 对预测与目标进行相同 padding
    if pad > 0:
        holo_mid_pad = complex_pad(holo_mid, pad)
        target_pad = complex_pad(target_complex, pad)
    else:
        holo_mid_pad = holo_mid
        target_pad = target_complex

    # 3. 深度偏移（正向传播 + 相位补偿）
    holo_shifted = propagator_pad(holo_mid_pad, depth_shift) * compl_exp(
        -2 * np.pi * depth_shift / wlen_tensor).to(torch.complex64)

    # 4. 复数 DDPM 校正
    bypass = training_params.get('bypass_ddpm_network', False)
    if ddpm_net is not None and not bypass:
        holo_altered = ddpm_net(holo_shifted)
    else:
        holo_altered = holo_shifted

    # 5. 双相位编码（AA-DPM，不进行相位归一化，保持弧度）
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
        wavelength=wavelengths,
    )

    # 6. 物理孔径滤波并反向传播，得到最终复数场
    amp_filtered, phs_filtered = filter_phs_only(
        phs_only,
        unnormalize_input=False,
        normalize_output=False,   # 输出相位保持弧度
        propagator=propagator_pad,
        depth_shift=-depth_shift,
        batch=rgbd.size(0),
        num_channels=3,
        res_h=holo_altered.shape[2],
        res_w=holo_altered.shape[3],
        radius=None,
        phs_max=None,
        amp_max=amp_max,
        wavelength=wavelengths,
    )
    final_complex = compl_val(amp_filtered, phs_filtered)

    return {
        'holo_altered': holo_altered,
        'final_complex': final_complex,
        'target_padded': target_pad,
        'holo_shifted': holo_shifted,
        'amp_max': amp_max,
    }


# ----------------------------------------------------------------------
# 模型保存与恢复
# ----------------------------------------------------------------------
def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(filepath, device=None):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No checkpoint found at {filepath}")
    print(f"Loading checkpoint from {filepath}")
    return torch.load(filepath, map_location=device)


# ----------------------------------------------------------------------
# 训练主逻辑
# ----------------------------------------------------------------------
def train_stage2(args, hologram_params, training_params, loss_params, device):
    # ---------- 构建模型 ----------
    holonet = ComplexHoloNet(
        input_dim=4 * (args.active_max_ldi_layer + 1),
        num_layers=args.num_layers,
        num_filters_per_layer=args.num_filters_per_layer,
        interleave_rate=1,
        filter_width=3,
        bias_stddev=0.01,
        weight_var_scale=0.25,
    ).to(device)

    # 加载 Stage1 权重
    print(f"Loading Stage1 checkpoint from {args.stage1_ckpt}")
    s1_ckpt = load_checkpoint(args.stage1_ckpt, device)
    if 'model_state_dict' in s1_ckpt:
        holonet.load_state_dict(s1_ckpt['model_state_dict'])
    elif 'holonet_state_dict' in s1_ckpt:
        holonet.load_state_dict(s1_ckpt['holonet_state_dict'])
    else:
        holonet.load_state_dict(s1_ckpt)
    print("Stage1 weights loaded successfully.")

    # 复数 DDPM 网络
    ddpm_net = None
    if args.activate_ddpm and not args.bypass_ddpm_network:
        ddpm_net = ComplexDDPMNet(
            input_dim=3,
            output_dim=3,
            num_layers=8,
            num_filters_per_layer=8,
            interleave_rate=1,
            filter_width=3,
            bias_stddev=0.01,
            weight_var_scale=0.25,
        ).to(device)
        print("DDPM network created.")

    # ---------- 传播算子（带 padding） ----------
    res_h, res_w = hologram_params['res_h'], hologram_params['res_w']
    pad = training_params['padding']
    propagator_pad = build_propagator_padded(
        res_h, res_w, hologram_params['pitch'],
        hologram_params['wavelengths'], pad
    ).to(device)

    # ---------- 数据加载 ----------
    active_layer = args.active_max_ldi_layer
    labels = ["amp_4", "phs_4"]
    for i in range(active_layer + 1):
        labels.extend([f"img_{i}", f"depth_{i}"])

    cur_dir = os.getcwd()
    train_tfrecord = os.path.join(cur_dir, "data", f"train_{args.dataset_res}_v2",
                                  f"train_{active_layer}4.tfrecord")
    val_tfrecord   = os.path.join(cur_dir, "data", f"test_{args.dataset_res}_v2",
                                  f"test_{active_layer}4.tfrecord")

    dataset_params = {"res_h": res_h, "res_w": res_w, "sample_count": 3800}
    val_dataset_params = {"res_h": res_h, "res_w": res_w, "sample_count": 100}

    train_loader = create_dataloader(
        train_tfrecord, dataset_params, labels,
        active_max_ldi_layer=active_layer,
        batch_size=args.batch, shuffle=True,
        num_workers=2, drop_last=True,
    )
    val_loader = create_dataloader(
        val_tfrecord, val_dataset_params, labels,
        active_max_ldi_layer=0,
        batch_size=args.batch, shuffle=False,
        num_workers=1, drop_last=True,
    )

    # ---------- 优化器 ----------
    params = list(holonet.parameters())
    if ddpm_net is not None:
        params += list(ddpm_net.parameters())
    optimizer = optim.Adam(params, lr=args.learning_rate,
                           betas=(0.9, 0.99), eps=1e-8)

    start_epoch = 0
    global_step = 0
    identity_epochs = args.stage2_epochs          # 身份预训练 epoch 数
    joint_epochs = args.joint_epochs              # 联合训练 epoch 数
    total_epochs = identity_epochs + joint_epochs

    # 恢复 Stage2 训练状态
    if args.restore_stage2 and args.stage2_ckpt_dir:
        ckpt_path = os.path.join(args.stage2_ckpt_dir, 'stage2_latest.pth')
        if os.path.isfile(ckpt_path):
            checkpoint = load_checkpoint(ckpt_path, device)
            holonet.load_state_dict(checkpoint['holonet_state_dict'])
            if ddpm_net is not None and 'ddpm_net_state_dict' in checkpoint:
                ddpm_net.load_state_dict(checkpoint['ddpm_net_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step']
            print(f"Resumed Stage2 from epoch {start_epoch}, step {global_step}")

    # 确定 checkpoint 保存目录
    if args.stage2_ckpt_dir is None:
        ckpt_dir = os.path.join(cur_dir, "model",
                                f"stage2_{args.model_name}_pad{pad}_shift{args.depth_shift}")
    else:
        ckpt_dir = args.stage2_ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---------- 训练循环 ----------
    print("Start Stage 2 Training!")
    for epoch in range(start_epoch, total_epochs):
        # 根据当前 epoch 确定训练阶段
        if epoch < identity_epochs and ddpm_net is not None and not args.bypass_ddpm_network:
            phase = 'identity'
            # 冻结主网络，只训练 DDPM
            for p in holonet.parameters():
                p.requires_grad = False
            for p in ddpm_net.parameters():
                p.requires_grad = True
        else:
            phase = 'joint'
            # 全部参数可训练
            for p in holonet.parameters():
                p.requires_grad = True
            if ddpm_net is not None:
                for p in ddpm_net.parameters():
                    p.requires_grad = True

        epoch_loss = 0.0
        epoch_aux = {}  # 辅助指标

        for batch_idx, batch_data in enumerate(train_loader):
            rgbd = batch_data['rgbd'].to(device)
            target_complex = batch_data['target_complex'].to(device)

            # 前向传播
            fwd_out = _run_stage2_forward(
                holonet, ddpm_net, rgbd, target_complex,
                propagator_pad, hologram_params, training_params,
            )

            if phase == 'identity':
                # 身份损失：要求 DDPM 输出等于其输入
                loss = identity_loss(fwd_out['holo_shifted'], fwd_out['holo_altered'],
                                     loss_type='l1')
                # 计算振幅图的 SSIM 和 PSNR
                amp_pred = fwd_out['final_complex'].abs()
                amp_gt   = fwd_out['target_padded'].abs()
                ssim_val = compute_ssim(amp_pred, amp_gt)
                psnr_val = compute_psnr(amp_pred, amp_gt)
            else:
                # 焦栈损失（基于最终复数场）
                fs_loss, fs_tv_loss, ssim_img, psnr_img = compute_focal_stack_loss(
                    fwd_out['final_complex'], fwd_out['target_padded'],
                    rgbd, propagator_pad, hologram_params, training_params,
                    loss_fn=F.l1_loss,
                    pad=pad,
                )
                # DDPM 相位正则
                std_loss, mean_loss = complex_ddpm_phase_loss(
                    fwd_out['holo_altered'],
                    pad=pad, res_h=res_h, res_w=res_w,
                )
                total_loss = (loss_params['weight_fs'] * fs_loss +
                              loss_params['weight_fs_tv'] * fs_tv_loss +
                              loss_params['weight_std'] * std_loss +
                              loss_params['weight_mean'] * mean_loss)
                loss = total_loss

                # 记录详细损失
                epoch_aux.setdefault('fs', 0.0)
                epoch_aux.setdefault('fs_tv', 0.0)
                epoch_aux.setdefault('std', 0.0)
                epoch_aux.setdefault('mean', 0.0)
                epoch_aux.setdefault('ssim_img', 0.0)
                epoch_aux.setdefault('psnr_img', 0.0)
                epoch_aux['fs'] += fs_loss.item()
                epoch_aux['fs_tv'] += fs_tv_loss.item()
                epoch_aux['std'] += std_loss.item()
                epoch_aux['mean'] += mean_loss.item()
                epoch_aux['ssim_img'] += ssim_img.item()
                epoch_aux['psnr_img'] += psnr_img.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            epoch_loss += loss.item()

            # 控制台输出
            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                log_str = (f"Epoch {epoch:3d} [{phase:>8s}] Batch {batch_idx:4d}/{len(train_loader)} "
                           f"| Loss {loss.item():.6f} | SSIM {ssim_val:.4f} | PSNR {psnr_val:.2f}")
                if phase == 'joint':
                    log_str += (f" | FS {fs_loss.item():.6f} | FS_TV {fs_tv_loss.item():.6f} "
                                f"| Std {std_loss.item():.4f} | Mean {mean_loss.item():.4f}")
                print(log_str)

            # 定期验证
            if global_step % args.num_iter_per_test == 0 and global_step > 0:
                holonet.eval()
                if ddpm_net is not None:
                    ddpm_net.eval()

                val_loss = 0.0
                val_ssim = 0.0
                val_psnr = 0.0
                num_val = len(val_loader)
                with torch.no_grad():
                    for v_batch in val_loader:
                        v_rgbd = v_batch['rgbd'].to(device)
                        v_target = v_batch['target_complex'].to(device)
                        v_fwd = _run_stage2_forward(
                            holonet, ddpm_net, v_rgbd, v_target,
                            propagator_pad, hologram_params, training_params,
                        )
                        if phase == 'identity':
                            v_loss = identity_loss(v_fwd['holo_shifted'], v_fwd['holo_altered'])
                        else:
                            fs_l, fs_tv, ssim_i, psnr_i = compute_focal_stack_loss(
                                v_fwd['final_complex'], v_fwd['target_padded'],
                                v_rgbd, propagator_pad, hologram_params, training_params,
                                loss_fn=F.l1_loss, pad=pad,
                            )
                            std_l, mean_l = complex_ddpm_phase_loss(
                                v_fwd['holo_altered'], pad=pad, res_h=res_h, res_w=res_w,
                            )
                            v_loss = (loss_params['weight_fs'] * fs_l +
                                      loss_params['weight_fs_tv'] * fs_tv +
                                      loss_params['weight_std'] * std_l +
                                      loss_params['weight_mean'] * mean_l)
                            val_ssim += ssim_i.item()
                            val_psnr += psnr_i.item()
                        val_loss += v_loss.item()
                val_loss /= num_val
                val_ssim /= num_val
                val_psnr /= num_val
                print(f"--- Validation at step {global_step} ({phase}) ---")
                print(f"Val Loss: {val_loss:.6f}", end="")
                if phase == 'joint':
                    print(f" | SSIM_img: {val_ssim:.4f} | PSNR_img: {val_psnr:.2f}")
                else:
                    print()
                holonet.train()
                if ddpm_net is not None:
                    ddpm_net.train()

        # 每个 epoch 结束时的统计与保存
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch:3d} [{phase:>8s}] Average Loss: {avg_loss:.6f}")
        if phase == 'joint':
            for k in epoch_aux:
                print(f"  {k}: {epoch_aux[k] / len(train_loader):.6f}")

        # 保存 checkpoint
        state = {
            'epoch': epoch,
            'global_step': global_step,
            'holonet_state_dict': holonet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if ddpm_net is not None:
            state['ddpm_net_state_dict'] = ddpm_net.state_dict()
        save_checkpoint(state, os.path.join(ckpt_dir, f'stage2_epoch_{epoch:04d}.pth'))
        save_checkpoint(state, os.path.join(ckpt_dir, 'stage2_latest.pth'))

    print("Stage 2 Training completed.")


# ----------------------------------------------------------------------
# 参数解析与主函数
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Stage 2 Training (Complex DDPM)')
    parser.add_argument('--model-name', default='full_loss', type=str)
    parser.add_argument('--dataset-res', default=192, type=int)
    parser.add_argument('--pitch', default=0.008, type=float)
    parser.add_argument('--num-layers', default=30, type=int)
    parser.add_argument('--num-filters-per-layer', default=24, type=int)
    parser.add_argument('--batch', default=2, type=int)
    parser.add_argument('--learning-rate', default=1e-4, type=float)
    parser.add_argument('--stage1-ckpt', type=str, required=True,
                        help='Path to Stage1 checkpoint')
    parser.add_argument('--activate-ddpm', action='store_true',
                        help='Use DDPM network')
    parser.add_argument('--bypass-ddpm-network', action='store_true',
                        help='Bypass DDPM network (even if activated)')
    parser.add_argument('--padding', default=0, type=int,
                        help='Padding for hologram')
    parser.add_argument('--depth-shift', default=12.0, type=float,
                        help='Depth shift in mm')
    parser.add_argument('--stage2-ckpt-dir', default=None, type=str,
                        help='Directory to save Stage2 checkpoints')
    parser.add_argument('--restore-stage2', action='store_true',
                        help='Restore Stage2 training from checkpoint')
    parser.add_argument('--stage2-epochs', default=50, type=int,
                        help='Number of identity pretraining epochs')
    parser.add_argument('--joint-epochs', default=200, type=int,
                        help='Number of joint training epochs')
    parser.add_argument('--restore-stage1', action='store_true',
                        help='(Reserved)')
    parser.add_argument('--epoch-to-start-ddpm', default=3000, type=int,
                        help='(Not used in Stage2 standalone)')
    parser.add_argument('--num-iter-per-test', default=500, type=int,
                        help='Validation frequency (in steps)')
    parser.add_argument('--active-max-ldi-layer', type=int, default=0,
                        help='Maximum LDI layer index (0 for single RGBD)')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    hologram_params = {
        "wavelengths": np.array([0.000450, 0.000520, 0.000638]),
        "pitch": args.pitch,
        "res_h": args.dataset_res,
        "res_w": args.dataset_res,
        "depth_base": -3,
        "depth_scale": 6,
        "double_pad": True,
    }

    training_params = {
        "restore_trained_model": args.restore_stage1,
        "batch": args.batch,
        "padding": args.padding,
        "depth_shift": args.depth_shift,
        "learning_rate": args.learning_rate,
        "decay_type": None,
        "decay_params": None,
        "num_iter_per_test": args.num_iter_per_test,
        "num_top_depth_for_img_loss": 15,
        "num_random_depth_for_img_loss": 5,
        "depth_dependent_weight_scale": 0.35,
        "num_hist_bins": 200,
        "bypass_ddpm_network": args.bypass_ddpm_network,
    }

    loss_params = {
        "loss_type": "l1",
        "weight_fs": float(training_params["num_top_depth_for_img_loss"] +
                           training_params["num_random_depth_for_img_loss"]),
        "weight_fs_tv": float(training_params["num_top_depth_for_img_loss"] +
                              training_params["num_random_depth_for_img_loss"]),
        "weight_std": 0.02,
        "weight_mean": 0.03,
    }

    train_stage2(args, hologram_params, training_params, loss_params, device)


if __name__ == '__main__':
    main()