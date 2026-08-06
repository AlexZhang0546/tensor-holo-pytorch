"""
DDPM 阶段专属损失项：相位均值和标准差的正则项。
对应原 TF 代码 _get_loss 中当 y_out_phs_shifted 不为 None 时
添加的 std_loss 和 mean_loss。
"""

import torch


def compute_ddpm_phase_loss(
    phs_out_shifted: torch.Tensor,
    pad: int = 0,
    res_h: int = None,
    res_w: int = None
):
    """
    计算 DDPM 校正阶段相位图的统计正则损失。

    Args:
        phs_out_shifted: 经过 DDPM 网络调整后的相位图（双相位编码前），
                         形状 (B, 3, H_pad, W_pad)，值域 [0, 1]。
        pad:            边缘填充量（需要在计算前裁剪）。
        res_h, res_w:   原始（未填充）的高度和宽度，用于裁剪区域。
                        若为 None，则根据 pad 自动推断（要求 phs_out_shifted 尺寸足够）。

    Returns:
        std_loss:  各通道空间标准差的均值。
        mean_loss: 各通道空间均值偏离 0.5 的绝对值的平均。
    """
    if pad > 0:
        if res_h is None:
            res_h = phs_out_shifted.shape[2] - 2 * pad
        if res_w is None:
            res_w = phs_out_shifted.shape[3] - 2 * pad
        # 裁剪中心有效区域
        phs_cropped = phs_out_shifted[:, :, pad:pad + res_h, pad:pad + res_w]
    else:
        phs_cropped = phs_out_shifted

    # 空间维度（H, W）的标准差，然后对通道和 batch 取均值
    std_per_channel = phs_cropped.std(dim=(2, 3))          # (B, C)
    std_loss = std_per_channel.mean()

    # 空间均值偏移量（减去 0.5 后取绝对值），再对通道和 batch 取均值
    mean_per_channel = phs_cropped.mean(dim=(2, 3))        # (B, C)
    mean_shift = (mean_per_channel - 0.5).abs()
    mean_loss = mean_shift.mean()

    return std_loss, mean_loss