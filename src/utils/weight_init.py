"""
PyTorch 版本的权重初始化函数。
替代原 TensorFlow 中的 tf_init_weights，可作为模型参数初始化工具。
"""

import math
import torch
import numpy as np


def _xavier_uniform_range(fan_in: int, fan_out: int, r: float = 0.5):
    """
    计算 Xavier‑uniform 初始化的上下界。

    high = sqrt(r * 2.0 / (fan_in + fan_out))
    """
    high = math.sqrt(r * 2.0 / (fan_in + fan_out))
    return -high, high


def init_weights_real(tensor: torch.Tensor,
                      fan_in: int,
                      fan_out: int,
                      r: float = 0.5,
                      seed: int = None) -> torch.Tensor:
    low, high = _xavier_uniform_range(fan_in, fan_out, r)
    with torch.no_grad():
        if seed is not None:
            generator = torch.Generator(device=tensor.device)
            generator.manual_seed(seed)
            tensor.uniform_(low, high, generator=generator)
        else:
            tensor.uniform_(low, high)   # 使用全局随机状态
    return tensor


def init_weights_complex(real_tensor: torch.Tensor,
                         imag_tensor: torch.Tensor,
                         fan_in: int,
                         fan_out: int,
                         r: float = 0.5,
                         seed: int = 0):
    """
    对复数权重（分别用两个实数张量表示实部和虚部）进行独立初始化。

    Args:
        real_tensor: 实部张量。
        imag_tensor: 虚部张量。
        fan_in, fan_out, r, seed: 同 init_weights_real。

    Returns:
        (real_tensor, imag_tensor)
    """
    # 为避免实部虚部完全相同，使用不同的种子
    init_weights_real(real_tensor, fan_in, fan_out, r, seed)
    init_weights_real(imag_tensor, fan_in, fan_out, r, seed + 1)
    return real_tensor, imag_tensor


def init_weights(tensor: torch.Tensor,
                 fan_in: int,
                 fan_out: int,
                 r: float = 0.5,
                 seed: int = 0,
                 is_complex: bool = False,
                 imag_tensor: torch.Tensor = None):
    """
    统一接口：根据 is_complex 标志初始化实数或复数权重。

    Args:
        tensor:      权重张量（实数情况）。
        fan_in:      输入单元数。
        fan_out:     输出单元数。
        r:           方差系数。
        seed:        随机种子。
        is_complex:  是否为复数权重。
        imag_tensor: 若 is_complex=True，则需提供虚部张量。

    Returns:
        初始化后的张量；复数情况返回 (real, imag)。
    """
    if is_complex:
        if imag_tensor is None:
            raise ValueError("For complex weights, imag_tensor must be provided.")
        return init_weights_complex(tensor, imag_tensor, fan_in, fan_out, r, seed)
    else:
        return init_weights_real(tensor, fan_in, fan_out, r, seed)


# ---------- 便捷函数：直接用于 nn.Module 的权重初始化 ----------
def holonet_weight_init(module: torch.nn.Module,
                        r: float = 0.5,
                        seed: int = 0):
    """
    对 HoloNet / DDPM‑Net 的卷积层进行自定义 Xavier 初始化。
    适合在模型定义中调用 self.apply(holonet_weight_init)。

    该函数会跳过 bias（偏置另有初始化方式，如常数或正态），
    仅对 Conv2d 的 weight 进行初始化。
    """
    if isinstance(module, torch.nn.Conv2d):
        fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
        fan_out = module.out_channels * module.kernel_size[0] * module.kernel_size[1]
        init_weights_real(module.weight, fan_in, fan_out, r=r, seed=seed)