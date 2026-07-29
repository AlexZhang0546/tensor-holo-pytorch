"""
PyTorch 版本的 interleave / deinterleave 工具函数。
等效替代原 TensorFlow 中的 tf_interleave_nonnative / tf_deinterleave_nonnative，
完全基于 torch.reshape + torch.permute 实现，与 TensorRT 兼容。
"""

import torch


def interleave_nonnative(r: int, x: torch.Tensor) -> torch.Tensor:
    """
    将空间像素按 r×r 的块重新排列到通道维度（space‑to‑depth 的一种实现）。

    输入形状: (batch, depth, height, width)
    输出形状: (batch, depth * r * r, height // r, width // r)

    Args:
        r: 下采样倍率（必须整除 height 和 width）。
        x: 输入张量，形状 (B, C, H, W)。

    Returns:
        重排后的张量。
    """
    if r == 1:
        return x

    B, C, H, W = x.shape
    reduced_H = H // r
    reduced_W = W // r

    # 将 H 和 W 分别拆分为 (reduced_H, r) 和 (reduced_W, r)
    # 形状变为 (B, C, reduced_H, r, reduced_W, r)
    y = x.reshape(B, C, reduced_H, r, reduced_W, r)

    # 转置为 (B, r, r, C, reduced_H, reduced_W)
    # 注意：原 TF 代码使用 perm [0, 3, 5, 1, 2, 4]
    z = y.permute(0, 3, 5, 1, 2, 4).contiguous()

    # 展平前三个维度 (B, r*r*C, reduced_H, reduced_W)
    out = z.reshape(B, -1, reduced_H, reduced_W)
    return out


def deinterleave_nonnative(r: int, x: torch.Tensor) -> torch.Tensor:
    """
    interleave_nonnative 的逆操作，将通道维度重新展开回空间维度（depth‑to‑space）。

    输入形状: (batch, depth, height, width)  其中 depth 应等于原始通道数 * r * r
    输出形状: (batch, depth // (r * r), height * r, width * r)

    Args:
        r: 上采样倍率。
        x: 输入张量，形状 (B, C', H', W')，C' = C * r * r。

    Returns:
        恢复空间分辨率的张量。
    """
    if r == 1:
        return x

    B, C_prime, H, W = x.shape
    C = C_prime // (r * r)          # 原始通道数
    expanded_H = H * r
    expanded_W = W * r

    # 重塑为 (B, r, r, C, H, W)
    y = x.reshape(B, r, r, C, H, W)

    # 转置为 (B, C, H, r, W, r)  — 对应原 TF 的 [0,3,4,1,5,2]
    z = y.permute(0, 3, 4, 1, 5, 2).contiguous()

    # 最后重塑为 (B, C, expanded_H, expanded_W)
    out = z.reshape(B, C, expanded_H, expanded_W)
    return out