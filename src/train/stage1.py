# src/train/stage1.py
"""
阶段一训练脚本：训练主网络（ComplexHoloNet，输出复数光场），
使用复数全息图损失 + 焦栈损失 + TV 损失的组合进行优化。

对应原始 TensorFlow 代码中 train() 方法里的 stage 1 部分，
但已改为复数输出，直接输出复数场。
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # 可选

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.holonet import ComplexHoloNet             # 复数主网络
from src.data.dataset import THDataset, create_dataloader
from src.optics.propagation import propagator_factory
from src.losses.complex_losses import complex_holo_loss   # 复数全息损失
from src.losses.focal_stack import compute_focal_stack_loss
from src.utils.metrics import compute_ssim, compute_psnr


def parse_args():
    parser = argparse.ArgumentParser(description='Stage 1 Training (Complex Output)')
    parser.add_argument('--model-name', default='full_loss', type=str)
    parser.add_argument('--dataset-res', default=192, type=int)
    parser.add_argument('--pitch', default=0.008, type=float)
    parser.add_argument('--num-layers', default=30, type=int)
    parser.add_argument('--num-filters-per-layer', default=24, type=int)
    parser.add_argument('--num-epochs', default=4050, type=int)
    parser.add_argument('--batch', default=2, type=int)
    parser.add_argument('--learning-rate', default=1e-4, type=float)
    parser.add_argument('--num-iter-per-test', default=1000, type=int)
    parser.add_argument('--restore', action='store_true', help='Restore from checkpoint')
    parser.add_argument('--ckpt-dir', default=None, type=str, help='Checkpoint directory')
    parser.add_argument('--active-max-ldi-layer', type=int, default=0,
                        help='Maximum LDI layer index (0 for single RGBD)')
    return parser.parse_args()


def build_model(model_params):
    """根据参数字典构建复数主网络"""
    model = ComplexHoloNet(
        input_dim=model_params['input_dim'],
        num_layers=model_params['num_layers'],
        num_filters_per_layer=model_params['num_filters_per_layer'],
        interleave_rate=model_params.get('interleave_rate', 1),
        filter_width=model_params.get('filter_width', 3),
        bias_stddev=model_params.get('bias_stddev', 0.01),
        weight_var_scale=model_params.get('weight_var_scale', 0.25)
    )
    return model


def build_propagator(hologram_params):
    """构建无 padding 的角谱传播算子（用于 stage1 损失计算）。"""
    return propagator_factory(
        input_shape=(hologram_params['res_h'], hologram_params['res_w']),
        pitch=hologram_params['pitch'],
        wavelengths=hologram_params['wavelengths'],
        method='as',
        double_pad=True
    )


def combine_loss(
    holo_out,                    # 复数预测场 (B, 3, H, W)
    target_complex,             # 复数目标场 (B, 3, H, W)
    rgbd,                       # 输入 RGBD (B, C_in, H, W)
    propagator,                 # 传播算子
    hologram_params,            # 光学参数
    training_params,            # 训练参数
    loss_fn,                    # 像素损失函数 (l1_loss / mse_loss)
    loss_type,                  # 'l1' or 'l2'
    loss_params,                # 损失权重等
    pad=0
):
    """
    将全息图损失与焦栈损失组合为总损失，并返回所有 metric。

    参数:
        holo_out:     复数预测场 (B, 3, H, W)
        target_complex: 复数目标场 (B, 3, H, W)
        rgbd:         输入 RGBD
        propagator:   传播算子
        ...           其他配置

    返回:
        total_loss, holo_loss, fs_loss, fs_tv_loss,
        ssim_amp, psnr_amp, ssim_img, psnr_img
    """
    # 1. 裁剪后的复数场用于全息损失（保持与原来裁剪一致）
    if pad > 0:
        holo_out_crop = holo_out[:, :, pad:-pad, pad:-pad]
        target_crop   = target_complex[:, :, pad:-pad, pad:-pad]
    else:
        holo_out_crop = holo_out
        target_crop   = target_complex

    # 2. 全息图复数损失
    holo_loss = complex_holo_loss(holo_out_crop, target_crop,
                                  loss_type=loss_type, method='magnitude_phase')

    # 3. 焦栈损失（内部会处理传播和裁剪，需传入未裁剪的复数场）
    fs_loss, fs_tv_loss, ssim_img, psnr_img = compute_focal_stack_loss(
        holo_out, target_complex, rgbd, propagator,
        hologram_params, training_params, loss_fn, pad=pad
    )

    # 4. 振幅图 SSIM / PSNR（在裁剪后的振幅上计算）
    amp_pred = holo_out_crop.abs()
    amp_gt   = target_crop.abs()
    ssim_amp = compute_ssim(amp_pred, amp_gt, data_range=1.0)
    psnr_amp = compute_psnr(amp_pred, amp_gt, data_range=1.0)

    # 5. 总损失组合
    weight_holo = loss_params.get('weight_holo', 1.0)
    weight_fs   = loss_params.get('weight_fs', 1.0)
    weight_fs_tv = loss_params.get('weight_fs_tv', 1.0)
    total_loss = weight_holo * holo_loss + weight_fs * fs_loss + weight_fs_tv * fs_tv_loss

    return total_loss, holo_loss, fs_loss, fs_tv_loss, ssim_amp, psnr_amp, ssim_img, psnr_img


def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        print(f"Loading checkpoint from {filename}")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        return start_epoch, global_step
    else:
        raise FileNotFoundError(f"No checkpoint found at {filename}")


def train_stage1(
    model_params, hologram_params, training_params, loss_params,
    train_loader, val_loader, device, ckpt_dir, restore=False
):
    # ---------- 构建模型、传播算子、优化器 ----------
    model = build_model(model_params).to(device)
    propagator = build_propagator(hologram_params).to(device)

    loss_type = loss_params.get('loss_type', 'l1')
    loss_fn = F.l1_loss if loss_type == 'l1' else F.mse_loss

    optimizer = optim.Adam(
        model.parameters(),
        lr=training_params.get('learning_rate', 1e-4),
        betas=(0.9, 0.99),
        eps=1e-8
    )

    scheduler = None
    if training_params.get('decay_type') == 'polynomial':
        scheduler = optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=training_params['num_epochs'] * len(train_loader),
            power=training_params.get('decay_power', 1.0)
        )

    start_epoch = 0
    global_step = 0
    num_epochs = training_params.get('num_epochs', 100)
    num_iter_per_test = training_params.get('num_iter_per_test', 1000)

    if restore:
        ckpt_path = os.path.join(ckpt_dir, 'stage1_latest.pth')
        start_epoch, global_step = load_checkpoint(model, optimizer, ckpt_path)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch}, global step {global_step}")

    # ---------- 训练循环 ----------
    model.train()
    print("Start Stage 1 Training! (Complex output)")
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        epoch_holo_loss = 0.0
        epoch_fs_loss = 0.0
        epoch_fs_tv = 0.0
        epoch_ssim_amp = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            rgbd = batch_data['rgbd'].to(device)                  # (B, C_in, H, W)
            target_complex = batch_data['target_complex'].to(device)  # 复数目标

            # 前向传播 → 直接输出复数场
            holo_out = model(rgbd)   # (B, 3, H, W) complex64

            # 计算组合损失
            total_loss, holo_loss, fs_loss, fs_tv, ssim_amp, _, _, _ = combine_loss(
                holo_out, target_complex, rgbd,
                propagator, hologram_params, training_params,
                loss_fn, loss_type, loss_params, pad=0
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            epoch_loss += total_loss.item()
            epoch_holo_loss += holo_loss.item()
            epoch_fs_loss += fs_loss.item()
            epoch_fs_tv += fs_tv.item()
            epoch_ssim_amp += ssim_amp.item()

            global_step += 1

            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                print(f"Epoch {epoch:3d} | Step {batch_idx:4d}/{len(train_loader)} | "
                      f"Loss {total_loss.item():.6f} | Holo {holo_loss.item():.6f} | "
                      f"FS {fs_loss.item():.6f} | FS_TV {fs_tv.item():.6f} | "
                      f"SSIM_amp {ssim_amp.item():.4f}")

            # 定期验证
            if global_step % num_iter_per_test == 0 and global_step > 0:
                model.eval()
                val_holo_loss = 0.0
                val_fs_loss = 0.0
                val_ssim_amp = 0.0
                val_ssim_img = 0.0
                num_val_batches = len(val_loader)
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_rgbd = val_batch['rgbd'].to(device)
                        val_target = val_batch['target_complex'].to(device)

                        val_holo_out = model(val_rgbd)
                        _, v_holo, v_fs, _, v_ssim_amp, _, v_ssim_img, _ = combine_loss(
                            val_holo_out, val_target, val_rgbd,
                            propagator, hologram_params, training_params,
                            loss_fn, loss_type, loss_params, pad=0
                        )
                        val_holo_loss += v_holo.item()
                        val_fs_loss += v_fs.item()
                        val_ssim_amp += v_ssim_amp.item()
                        val_ssim_img += v_ssim_img.item()

                val_holo_loss /= num_val_batches
                val_fs_loss /= num_val_batches
                val_ssim_amp /= num_val_batches
                val_ssim_img /= num_val_batches
                print(f"--- Validation at step {global_step} ---")
                print(f"Avg Holo Loss: {val_holo_loss:.6f} | Avg FS Loss: {val_fs_loss:.6f} | "
                      f"SSIM Amp: {val_ssim_amp:.4f} | SSIM Img: {val_ssim_img:.4f}")
                model.train()

        # 每个 epoch 结束打印平均损失
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_holo_loss = epoch_holo_loss / len(train_loader)
        avg_fs_loss = epoch_fs_loss / len(train_loader)
        avg_fs_tv = epoch_fs_tv / len(train_loader)
        avg_ssim_amp = epoch_ssim_amp / len(train_loader)
        print(f"Epoch {epoch:3d} finished | Avg Loss {avg_epoch_loss:.6f} | "
              f"Holo {avg_holo_loss:.6f} | FS {avg_fs_loss:.6f} | FS_TV {avg_fs_tv:.6f} | "
              f"SSIM {avg_ssim_amp:.4f}")

        # 保存 checkpoint
        ckpt_path = os.path.join(ckpt_dir, f'stage1_epoch_{epoch:04d}.pth')
        save_checkpoint({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, ckpt_path)
        save_checkpoint({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(ckpt_dir, 'stage1_latest.pth'))

    print("Stage 1 Training completed.")


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    active_layer = args.active_max_ldi_layer
    labels = ["amp_4", "phs_4"]
    for i in range(active_layer + 1):
        labels.extend([f"img_{i}", f"depth_{i}"])

    cur_dir = os.getcwd()
    train_tfrecord = os.path.join(cur_dir, "data", f"train_{args.dataset_res}_v2",
                                  f"train_{active_layer}4.tfrecord")
    val_tfrecord   = os.path.join(cur_dir, "data", f"test_{args.dataset_res}_v2",
                                  f"test_{active_layer}4.tfrecord")

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
        "restore_trained_model": args.restore,
        "batch": args.batch,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "decay_type": None,
        "decay_params": None,
        "num_iter_per_test": args.num_iter_per_test,
        "num_top_depth_for_img_loss": 15,
        "num_random_depth_for_img_loss": 5,
        "depth_dependent_weight_scale": 0.35,
        "num_hist_bins": 200,
        "depth_shift": 0.0
    }

    model_params = {
        "name": args.model_name,
        "input_dim": 4 * (active_layer + 1),    # 复数网络不需要 output_dim
        "num_layers": args.num_layers,
        "num_filters_per_layer": args.num_filters_per_layer,
        "interleave_rate": 1,
        "filter_width": 3,
        "bias_stddev": 0.01,
        "weight_var_scale": 0.25,
        "renormalize_input": True
    }

    loss_params = {
        "loss_type": "l1",
        "weight_holo": 1.0,
        "weight_fs": float(training_params["num_top_depth_for_img_loss"] +
                           training_params["num_random_depth_for_img_loss"]),
        "weight_fs_tv": float(training_params["num_top_depth_for_img_loss"] +
                              training_params["num_random_depth_for_img_loss"]),
        "weight_std": 0.02,
        "weight_mean": 0.03
    }

    dataset_params = {
        "res_h": args.dataset_res,
        "res_w": args.dataset_res,
        "sample_count": 3800
    }
    val_dataset_params = {
        "res_h": args.dataset_res,
        "res_w": args.dataset_res,
        "sample_count": 100
    }

    train_loader = create_dataloader(
        tfrecord_path=train_tfrecord,
        dataset_params=dataset_params,
        labels=labels,
        active_max_ldi_layer=active_layer,
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    val_loader = create_dataloader(
        tfrecord_path=val_tfrecord,
        dataset_params=val_dataset_params,
        labels=labels,
        active_max_ldi_layer=0,
        batch_size=args.batch,
        shuffle=False,
        num_workers=1,
        drop_last=True
    )

    if args.ckpt_dir is None:
        ckpt_name = (f"ckpt_{args.model_name}_pitch_{int(args.pitch*1000)}_"
                     f"layers_{args.num_layers}_filters_{args.num_filters_per_layer}_stage1")
        ckpt_dir = os.path.join(cur_dir, "model", ckpt_name)
    else:
        ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    train_stage1(
        model_params=model_params,
        hologram_params=hologram_params,
        training_params=training_params,
        loss_params=loss_params,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        ckpt_dir=ckpt_dir,
        restore=args.restore
    )


if __name__ == '__main__':
    main()