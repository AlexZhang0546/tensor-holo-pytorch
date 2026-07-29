"""
焦栈感知损失（focal stack loss）。
从深度图采样焦点平面，传播预测和目标全息图到该平面，
计算加权图像损失（L1/L2）、总变差（TV）损失，以及 SSIM / PSNR 指标。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable, Tuple
from src.utils.metrics import compute_ssim, compute_psnr

# 假设传播算子符合接口：propagator(cpx: Tensor, z_dist: float) -> Tensor


def _crop_margin_4d(x: torch.Tensor, margin: int) -> torch.Tensor:
    """对 4D 张量 (B, C, H, W) 进行中心裁剪，去掉每边 margin 个像素。"""
    if margin == 0:
        return x
    return x[:, :, margin:-margin, margin:-margin]


def _compute_tv_loss(img_in: torch.Tensor,
                     img_gt: torch.Tensor,
                     loss_fn: Callable) -> torch.Tensor:
    """
    计算两张图像的总变差损失。
    分别对宽度方向和高度方向计算梯度，然后计算 L1/L2 损失的平均值。
    """
    # 宽度方向梯度 (右 - 左)
    dx_in = img_in[:, :, :, 1:] - img_in[:, :, :, :-1]
    dx_gt = img_gt[:, :, :, 1:] - img_gt[:, :, :, :-1]
    # 高度方向梯度 (下 - 上)
    dy_in = img_in[:, :, 1:, :] - img_in[:, :, :-1, :]
    dy_gt = img_gt[:, :, 1:, :] - img_gt[:, :, :-1, :]

    tv_loss = 0.5 * loss_fn(dx_in, dx_gt) + 0.5 * loss_fn(dy_in, dy_gt)
    return tv_loss


def _get_depth_dependent_weight(
    depth_map: torch.Tensor,       # (1, 1, H, W) 或 (B, 1, H, W)
    depth_to_focus: float,         # 标量焦点深度
    depth_diff_max: float,         # 深度差异上限（depth_scale）
    weight_scale: float = 0.35
) -> torch.Tensor:
    """
    计算深度依赖的注意力权重，用于加权图像损失。
    depth_diff = (depth_diff_max - |depth_map - depth_to_focus|) * weight_scale
    weight = softmax-like (exp 后归一化到最大值为 1)
    """
    diff = depth_diff_max - torch.abs(depth_map - depth_to_focus)
    weighted = torch.exp(diff * weight_scale)
    # 归一化使最大值为 1（保持数值稳定）
    max_val = weighted.amax(dim=(2, 3), keepdim=True)  # 空间维度最大值
    depth_weight = weighted / (max_val + 1e-8)
    return depth_weight


def _img_diff_at_depth(
    holo_out_slice: torch.Tensor,   # (1, C, H, W) 复数
    holo_gt_slice: torch.Tensor,    # 同上
    depth_map_slice: torch.Tensor,  # (1, 1, H, W) 深度图
    depth_to_focus: float,          # 焦点深度（已缩放至物理距离）
    propagator: Callable,
    pad: int,
    loss_fn: Callable,
    depth_diff_max: float,
    depth_weight_scale: float = 0.35
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    在单个焦点平面上计算图像损失、TV 损失，并返回原始（未加权）振幅图像。
    """
    # 传播到焦点平面（负号：逆向传播）
    img_gt_complex = propagator(holo_gt_slice, -depth_to_focus)
    img_out_complex = propagator(holo_out_slice, -depth_to_focus)

    img_gt = torch.abs(img_gt_complex)   # 振幅图像
    img_out = torch.abs(img_out_complex)

    # 裁剪边缘
    img_gt_cropped = _crop_margin_4d(img_gt, pad)
    img_out_cropped = _crop_margin_4d(img_out, pad)

    # 深度依赖权重
    depth_weight = _get_depth_dependent_weight(
        depth_map_slice, depth_to_focus, depth_diff_max, depth_weight_scale
    )
    # 裁剪权重以匹配图像尺寸
    depth_weight = _crop_margin_4d(depth_weight, pad)

    # 加权图像
    weighted_gt = img_gt_cropped * depth_weight
    weighted_out = img_out_cropped * depth_weight

    # 图像损失（L1 或 L2）
    img_loss = loss_fn(weighted_gt, weighted_out)

    # TV 损失（在加权图像上计算）
    tv_loss = _compute_tv_loss(weighted_gt, weighted_out, loss_fn)

    return img_loss, tv_loss, img_gt_cropped, img_out_cropped


def _sample_focus_depths(
    depth_maps: torch.Tensor,      # (B, 1, H, W)
    num_hist_bins: int = 200,
    num_top: int = 15,
    num_random: int = 5
) -> torch.Tensor:
    """
    从一批深度图中采样焦点深度。
    返回形状 (B, num_top + num_random) 的张量，值在 [0, 1] 之间。
    """
    B = depth_maps.shape[0]
    device = depth_maps.device
    N_focus = num_top + num_random
    depth_to_focus_list = []

    for i in range(B):
        depth_flat = depth_maps[i].flatten()  # (H*W,)
        hist = torch.histc(depth_flat, bins=num_hist_bins, min=0.0, max=1.0)
        # 按计数降序排序，返回索引（0 ~ bins-1）
        sorted_idx = torch.argsort(hist, descending=True).float()
        # 为每个 bin 添加随机偏移（0 ~ 1），再归一化
        offset = torch.rand(1, device=device)
        idx = (sorted_idx + offset) / num_hist_bins

        top_depths = idx[:num_top]
        # 从剩余 bin 中随机选择 num_random 个
        rest = idx[num_top:]
        perm = torch.randperm(rest.size(0), device=device)
        random_depths = rest[perm[:num_random]]

        sample = torch.cat([top_depths, random_depths], dim=0)  # (N_focus,)
        depth_to_focus_list.append(sample)

    return torch.stack(depth_to_focus_list, dim=0)  # (B, N_focus)


# ---------- 主焦栈损失函数 ----------
def compute_focal_stack_loss(
    holo_out: torch.Tensor,          # (B, 3, H, W) 复数
    holo_gt: torch.Tensor,           # 同上
    rgbd: torch.Tensor,              # (B, input_dim, H, W) 包含深度图在通道 3
    propagator: Callable,
    hologram_params: dict,
    training_params: dict,
    loss_fn: Callable,
    pad: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算焦栈损失及相关指标。

    Args:
        holo_out, holo_gt: 预测与目标复数全息图 (B, 3, H, W)。
        rgbd: 输入 RGBD 数据，假设 depth 在通道索引 3（单层 LDI）。
        propagator: 传播函数，propagator(cpx, distance) -> cpx。
        hologram_params: 包含 'depth_scale', 'depth_base'。
        training_params: 包含 'num_hist_bins', 'num_top_depth_for_img_loss',
                         'num_random_depth_for_img_loss', 'depth_dependent_weight_scale'。
        loss_fn: 像素级损失（如 F.l1_loss / F.mse_loss）。
        pad: 边缘裁剪量。

    Returns:
        fs_loss: 平均焦栈图像损失。
        fs_tv_loss: 平均焦栈 TV 损失。
        ssim_img_loss: 平均 SSIM（加权前图像）。
        psnr_img_loss: 平均 PSNR。
    """
    # 提取深度图（假设 depth 在通道 3，形状 (B, 1, H, W)）
    depth = rgbd[:, 3:4, :, :]   # (B, 1, H, W)

    # 从参数中读取数值
    depth_scale = hologram_params['depth_scale']
    depth_base  = hologram_params['depth_base']
    num_hist_bins = training_params.get('num_hist_bins', 200)
    num_top = training_params.get('num_top_depth_for_img_loss', 15)
    num_random = training_params.get('num_random_depth_for_img_loss', 5)
    depth_weight_scale = training_params.get('depth_dependent_weight_scale', 0.35)

    B = holo_out.shape[0]

    # 1. 采样焦点深度（原始范围 [0, 1]）
    depth_to_focus_norm = _sample_focus_depths(depth, num_hist_bins, num_top, num_random)
    N_focus = depth_to_focus_norm.shape[1]  # num_top + num_random

    # 2. 将深度值缩放到实际物理范围
    depth_phys = depth * depth_scale + depth_base               # (B, 1, H, W)
    depth_to_focus_phys = depth_to_focus_norm * depth_scale + depth_base  # (B, N_focus)

    # 3. 逐样本、逐焦点计算损失
    fs_loss = torch.tensor(0.0, device=holo_out.device)
    fs_tv_loss = torch.tensor(0.0, device=holo_out.device)
    ssim_sum = torch.tensor(0.0, device=holo_out.device)
    psnr_sum = torch.tensor(0.0, device=holo_out.device)

    for i in range(B):
        for j in range(N_focus):
            img_loss, tv_loss, img_gt_cropped, img_out_cropped = _img_diff_at_depth(
                holo_out_slice=holo_out[i:i+1],        # (1, 3, H, W)
                holo_gt_slice=holo_gt[i:i+1],
                depth_map_slice=depth_phys[i:i+1],
                depth_to_focus=depth_to_focus_phys[i, j].item(),
                propagator=propagator,
                pad=pad,
                loss_fn=loss_fn,
                depth_diff_max=depth_scale,            # 深度差异上限
                depth_weight_scale=depth_weight_scale
            )
            fs_loss += img_loss
            fs_tv_loss += tv_loss

            # SSIM 和 PSNR（在未加权图像上，使用 data_range=1.0 与原 TF 保持一致）
            # 注意：原代码 amp 范围为 [0, √2]，但 SSIM 仍用 max_val=1.0，
            # 这里为忠实迁移保留该行为。
            ssim_sum += compute_ssim(img_gt_cropped, img_out_cropped, data_range=1.0)
            psnr_sum += compute_psnr(img_gt_cropped, img_out_cropped, data_range=1.0)

    # 4. 归一化（除以总焦点数）
    normalize_scale = float(B * N_focus)
    fs_loss = fs_loss / normalize_scale
    fs_tv_loss = fs_tv_loss / normalize_scale
    ssim_img_loss = ssim_sum / normalize_scale
    psnr_img_loss = psnr_sum / normalize_scale

    return fs_loss, fs_tv_loss, ssim_img_loss, psnr_img_loss