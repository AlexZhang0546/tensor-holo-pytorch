# src/train/stage1.py
"""
阶段一训练脚本：仅训练主网络（TensorHolographyNet），
使用全息图损失 + 焦栈损失 + TV 损失的组合进行优化。

对应原始 TensorFlow 代码中 train() 方法里的 stage 1 部分。
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
from torch.utils.tensorboard import SummaryWriter  # 可选，用于记录曲线

# 将项目根目录加入路径，确保绝对导入有效
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.holonet import TensorHolographyNet
from src.data.dataset import THDataset, create_dataloader
from src.optics.propagation import propagator_factory
from src.optics.complex_utils import compl_val
from src.losses.holo_loss import compute_holo_loss
from src.losses.focal_stack import compute_focal_stack_loss
from src.utils.metrics import compute_ssim, compute_psnr


def parse_args():
    """命令行参数解析（与原始 main_v2.py 中一致，仅保留 stage1 相关部分）。"""
    parser = argparse.ArgumentParser(description='Stage 1 Training')
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
    parser.add_argument('--active-max-ldi-layer', type=int, default=0, help='Maximum LDI layer index (0 for single RGBD)')
    # 更多的参数可以从默认配置读取，这里只列举关键部分
    return parser.parse_args()


def build_model(model_params):
    """根据参数字典构建主网络"""
    model = TensorHolographyNet(
        input_dim=model_params['input_dim'],
        output_dim=model_params['output_dim'],
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
    amp_out, phs_out, amp_gt, phs_gt, rgbd,
    propagator, hologram_params, training_params, loss_fn, loss_type, loss_params, pad=0
):
    """
    将全息图损失与焦栈损失组合为总损失，并返回所有 metric。

    参数:
        amp_out, phs_out: 预测的振幅/相位 (B,3,H,W)，值域 [0,√2] / [0,1]
        amp_gt, phs_gt: 目标振幅/相位
        rgbd: 输入 RGBD (B, C_in, H, W)，包含深度图
        propagator: 传播算子
        hologram_params: 光学参数字典
        training_params: 训练参数字典
        loss_fn: 像素损失函数 (l1_loss / mse_loss)
        pad: 裁剪边距（stage1 为 0）

    返回:
        loss: 总损失标量
        holo_loss: 全息损失
        fs_loss, fs_tv_loss: 焦栈损失分量
        ssim_amp, psnr_amp: 振幅图 SSIM/PSNR
        ssim_img, psnr_img: 焦栈图像平均 SSIM/PSNR
    """
    # 构造复数全息图
    holo_out = compl_val(amp_out, (phs_out - 0.5) * 2.0 * np.pi)
    holo_gt  = compl_val(amp_gt,  (phs_gt  - 0.5) * 2.0 * np.pi)

    # 1. 全息图振幅-相位损失
    holo_loss = compute_holo_loss(amp_out, phs_out, amp_gt, phs_gt,
                                  pad=pad, loss_type=loss_type)

    # 2. 焦栈损失（内部会进行深度采样与传播）
    fs_loss, fs_tv_loss, ssim_img, psnr_img = compute_focal_stack_loss(
        holo_out, holo_gt, rgbd, propagator,
        hologram_params, training_params, loss_fn, pad=pad
    )

    # 3. 振幅图 SSIM / PSNR（在裁剪后计算，与原始一致）
    if pad > 0:
        amp_out_crop = amp_out[:, :, pad:-pad, pad:-pad]
        amp_gt_crop  = amp_gt[:, :, pad:-pad, pad:-pad]
    else:
        amp_out_crop = amp_out
        amp_gt_crop  = amp_gt
    ssim_amp = compute_ssim(amp_out_crop, amp_gt_crop, data_range=1.0)
    psnr_amp = compute_psnr(amp_out_crop, amp_gt_crop, data_range=1.0)

    # 4. 组合总损失
    weight_holo = loss_params.get('weight_holo', 1.0)
    weight_fs = loss_params.get('weight_fs', 1.0)
    weight_fs_tv = loss_params.get('weight_fs_tv', 1.0)
    total_loss = weight_holo * holo_loss + weight_fs * fs_loss + weight_fs_tv * fs_tv_loss

    return total_loss, holo_loss, fs_loss, fs_tv_loss, ssim_amp, psnr_amp, ssim_img, psnr_img


def save_checkpoint(state, filename):
    """保存模型、优化器、epoch 等信息"""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, filename):
    """加载 checkpoint，返回起始 epoch 和 global_step"""
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
    """
    执行阶段一训练。

    参数:
        model_params, hologram_params, training_params, loss_params: 配置字典
        train_loader, val_loader: 训练与验证 DataLoader
        device: torch.device
        ckpt_dir: checkpoint 存储目录
        restore: 是否从已有 checkpoint 恢复训练
    """
    # ---------- 构建模型、传播算子、优化器 ----------
    model = build_model(model_params).to(device)
    propagator = build_propagator(hologram_params).to(device)

    # 选择损失函数（原代码默认 L1）
    loss_type = loss_params.get('loss_type', 'l1')
    loss_fn = F.l1_loss if loss_type == 'l1' else F.mse_loss

    # Adam 优化器（复刻原始参数）
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_params.get('learning_rate', 1e-4),
        betas=(0.9, 0.99),
        eps=1e-8
    )

    # 学习率调度（原代码 decay_type=None，因此固定学习率，这里保留接口）
    scheduler = None
    if training_params.get('decay_type') == 'polynomial':
        # 若需要使用多项式衰减，可通过 torch.optim.lr_scheduler 实现
        scheduler = optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=training_params['num_epochs'] * len(train_loader),
            power=training_params.get('decay_power', 1.0)
        )

    # 训练状态
    start_epoch = 0
    global_step = 0
    num_epochs = training_params.get('num_epochs', 100)
    num_iter_per_test = training_params.get('num_iter_per_test', 1000)

    # 恢复训练
    if restore:
        ckpt_path = os.path.join(ckpt_dir, 'stage1_latest.pth')
        start_epoch, global_step = load_checkpoint(model, optimizer, ckpt_path)
        start_epoch += 1  # 从下一个 epoch 开始
        print(f"Resumed from epoch {start_epoch}, global step {global_step}")

    # ---------- 训练循环 ----------
    model.train()
    print("Start Stage 1 Training!")
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        epoch_holo_loss = 0.0
        epoch_fs_loss = 0.0
        epoch_fs_tv = 0.0
        epoch_ssim_amp = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            # 取出数据并移至设备
            rgbd = batch_data['rgbd'].to(device)      # (B, C_in, H, W)
            amp_gt = batch_data['amp_4'].to(device)   # (B, 3, H, W)
            phs_gt = batch_data['phs_4'].to(device)

            # 前向传播
            amp_out, phs_out = model(rgbd)

            # 计算组合损失
            total_loss, holo_loss, fs_loss, fs_tv, ssim_amp, _, _, _ = combine_loss(
                amp_out, phs_out, amp_gt, phs_gt, rgbd,
                propagator, hologram_params, training_params, loss_fn, loss_type, loss_params, pad=0
            )

            # 反向传播与优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            # 累计统计
            epoch_loss += total_loss.item()
            epoch_holo_loss += holo_loss.item()
            epoch_fs_loss += fs_loss.item()
            epoch_fs_tv += fs_tv.item()
            epoch_ssim_amp += ssim_amp.item()

            global_step += 1

            # 控制台日志（仿照原 TF 格式）
            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                print(f"Epoch {epoch:3d} | Step {batch_idx:4d}/{len(train_loader)} | "
                      f"Loss {total_loss.item():.6f} | Holo {holo_loss.item():.6f} | "
                      f"FS {fs_loss.item():.6f} | FS_TV {fs_tv.item():.6f} | "
                      f"SSIM_amp {ssim_amp.item():.4f}")

            # ---------- 定期验证 ----------
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
                        val_amp_gt = val_batch['amp_4'].to(device)
                        val_phs_gt = val_batch['phs_4'].to(device)

                        val_amp_out, val_phs_out = model(val_rgbd)
                        _, v_holo, v_fs, _, v_ssim_amp, _, v_ssim_img, _ = combine_loss(
                            val_amp_out, val_phs_out, val_amp_gt, val_phs_gt, val_rgbd,
                            propagator, hologram_params, training_params, loss_fn, loss_type, loss_params, pad=0
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
                model.train()  # 切换回训练模式

        # 每个 epoch 结束后打印平均损失
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_holo_loss = epoch_holo_loss / len(train_loader)
        avg_fs_loss = epoch_fs_loss / len(train_loader)
        avg_fs_tv = epoch_fs_tv / len(train_loader)
        avg_ssim_amp = epoch_ssim_amp / len(train_loader)
        print(f"Epoch {epoch:3d} finished | Avg Loss {avg_epoch_loss:.6f} | "
              f"Holo {avg_holo_loss:.6f} | FS {avg_fs_loss:.6f} | FS_TV {avg_fs_tv:.6f} | "
              f"SSIM {avg_ssim_amp:.4f}")

        # 保存 checkpoint（每个 epoch 或定期保存）
        ckpt_path = os.path.join(ckpt_dir, f'stage1_epoch_{epoch:04d}.pth')
        save_checkpoint({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, ckpt_path)

        # 同时保存一个最新的（覆盖）
        save_checkpoint({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(ckpt_dir, 'stage1_latest.pth'))

    print("Stage 1 Training completed.")


def main():
    args = parse_args()
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据集标签（与原项目一致）
    active_layer = args.active_max_ldi_layer
    labels = ["amp_4", "phs_4"]  # 固定部分
    for i in range(active_layer + 1):
        labels.extend([f"img_{i}", f"depth_{i}"])

    # 数据集路径（根据原项目结构）
    cur_dir = os.getcwd()
    train_tfrecord = os.path.join(cur_dir, "data", f"train_{args.dataset_res}_v2",
                                  f"train_{active_layer}4.tfrecord")
    val_tfrecord   = os.path.join(cur_dir, "data", f"test_{args.dataset_res}_v2",
                                  f"test_{active_layer}4.tfrecord")

    # -------------------- 配置参数（等同于原项目中的字典） --------------------
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
        "decay_type": None,               # 无衰减
        "decay_params": None,
        "num_iter_per_test": args.num_iter_per_test,
        "num_top_depth_for_img_loss": 15,
        "num_random_depth_for_img_loss": 5,
        "depth_dependent_weight_scale": 0.35,
        "num_hist_bins": 200,
        "depth_shift": 0.0                # stage1 无深度偏移
    }

    model_params = {
        "name": args.model_name,
        "input_dim": 4 * (active_layer + 1), 
        "output_dim": 6,
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

    

    # 数据加载参数
    dataset_params = {
        "res_h": args.dataset_res,
        "res_w": args.dataset_res,
        "sample_count": 3800   # 训练集数量，与原代码一致
    }
    val_dataset_params = {
        "res_h": args.dataset_res,
        "res_w": args.dataset_res,
        "sample_count": 100
    }

    # 创建 DataLoader
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

    # checkpoint 存储目录
    if args.ckpt_dir is None:
        ckpt_name = f"ckpt_{args.model_name}_pitch_{int(args.pitch*1000)}_layers_{args.num_layers}_filters_{args.num_filters_per_layer}_stage1"
        ckpt_dir = os.path.join(cur_dir, "model", ckpt_name)
    else:
        ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    # 启动训练
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