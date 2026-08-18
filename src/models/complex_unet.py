"""
多尺度复数 U-Net 主网络（ComplexUNet）。

编码器-解码器结构，全链路复数运算：
  - 编码器：stem 卷积 + 逐级复数残差块 + stride=2 复数卷积下采样；
  - 瓶颈：复数残差块 + 可选复数自注意力；
  - 解码器：复数转置卷积上采样 + 拼接对应编码器特征（跳跃连接）+ 残差块；
  - 输出：复数场 (B, output_dim, H, W)，无激活。

输入预处理与 ComplexHoloNet 保持一致（先减 0.5 再构造复数）；输入分辨率
任意（内部按 2^depth 对齐 padding，输出裁剪回原尺寸）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.complex_layers import (
    ComplexConv2d,
    ComplexConvTranspose2d,
    ComplexBatchNorm2d,
    ComplexReLU,
)


class ComplexResBlock(nn.Module):
    """
    复数残差块：conv -> BN -> modReLU -> conv -> BN -> modReLU，再与输入恒等相加。
    """
    def __init__(self, channels, filter_width=3, bias_stddev=0.01,
                 weight_var_scale=0.25):
        super().__init__()
        pad = filter_width // 2
        self.conv1 = ComplexConv2d(channels, channels, filter_width,
                                   padding=pad, bias=True)
        self.bn1 = ComplexBatchNorm2d(channels)
        self.act1 = ComplexReLU(channels)
        self.conv2 = ComplexConv2d(channels, channels, filter_width,
                                   padding=pad, bias=True)
        self.bn2 = ComplexBatchNorm2d(channels)
        self.act2 = ComplexReLU(channels)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        return out + x


class ComplexSelfAttention(nn.Module):
    """
    复数自注意力（瓶颈层可选）。
    以 Re(q·k) 作为注意力权重（缩放点积），对复数 v 加权求和，
    再接 1x1 复数卷积并保留残差连接。
    """
    def __init__(self, channels):
        super().__init__()
        self.q = ComplexConv2d(channels, channels, 1, bias=False)
        self.k = ComplexConv2d(channels, channels, 1, bias=False)
        self.v = ComplexConv2d(channels, channels, 1, bias=False)
        self.out = ComplexConv2d(channels, channels, 1, bias=True)
        self.scale = channels ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).flatten(2).transpose(1, 2)      # (B, HW, C)
        k = self.k(x).flatten(2)                      # (B, C, HW)
        v = self.v(x).flatten(2).transpose(1, 2)      # (B, HW, C)
        scores = torch.real(q @ k) * self.scale       # (B, HW, HW)
        attn = torch.softmax(scores, dim=-1).to(v.dtype)
        out = attn @ v                                # (B, HW, C)
        out = out.transpose(1, 2).view(B, C, H, W)
        return self.out(out) + x


class ComplexUNet(nn.Module):
    """
    多尺度复数 U-Net 主网络。

    Args:
        input_dim: 实数输入通道数（RGBD 为 4，多层 LDI 为 4*(LDI+1)）。
        output_dim: 复数输出通道数（默认 3，RGB 三色光场）。
        depth: 下采样级数（2~3 次即 2^depth 分辨率缩减）。
        base_filters: 最浅层复数通道数，逐级翻倍。
        filter_width: 卷积核宽度（默认 3）。
        bias_stddev / weight_var_scale: 与 ComplexHoloNet 对齐的初始化参数（保留）。
        use_attention: 瓶颈层是否使用复数自注意力。

    输入:  (B, input_dim, H, W) 实数 RGBD/LDI
    输出:  complex_field (B, output_dim, H, W)，分辨率与输入相同，无激活。
    """
    def __init__(self,
                 input_dim: int = 4,
                 output_dim: int = 3,
                 depth: int = 3,
                 base_filters: int = 24,
                 filter_width: int = 3,
                 bias_stddev: float = 0.01,
                 weight_var_scale: float = 0.25,
                 use_attention: bool = False,
                 out_bn: bool = False,
                 stem_skip: bool = False,
                 refine_blocks: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.depth = depth
        self.base_filters = base_filters
        self.use_attention = use_attention
        self.out_bn_enabled = out_bn
        self.stem_skip_enabled = stem_skip
        self.refine_blocks = refine_blocks
        pad = filter_width // 2

        # ---- 编码器 ----
        self.stem_conv = ComplexConv2d(input_dim, base_filters, filter_width,
                                       padding=pad, bias=True)
        self.stem_bn = ComplexBatchNorm2d(base_filters)
        self.stem_act = ComplexReLU(base_filters)

        self.encoder_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        self.enc_channels = [base_filters]
        ch = base_filters
        for _ in range(depth):
            self.encoder_blocks.append(ComplexResBlock(
                ch, filter_width, bias_stddev, weight_var_scale))
            next_ch = ch * 2
            self.down_convs.append(ComplexConv2d(
                ch, next_ch, filter_width, stride=2, padding=pad, bias=True))
            ch = next_ch
            self.enc_channels.append(ch)

        # ---- 瓶颈 ----
        self.bottleneck = ComplexResBlock(
            ch, filter_width, bias_stddev, weight_var_scale)
        self.attention = ComplexSelfAttention(ch) if use_attention else None

        # ---- 解码器 ----
        self.up_convs = nn.ModuleList()
        self.reduce_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in range(depth, 0, -1):
            in_ch = self.enc_channels[i]
            out_ch = self.enc_channels[i - 1]
            self.up_convs.append(ComplexConvTranspose2d(
                in_ch, out_ch, filter_width, stride=2, padding=pad,
                output_padding=1, bias=True))
            self.reduce_convs.append(ComplexConv2d(
                out_ch * 2, out_ch, filter_width, padding=pad, bias=True))
            self.decoder_blocks.append(ComplexResBlock(
                out_ch, filter_width, bias_stddev, weight_var_scale))

        # ---- full-res refine blocks (optional, before output head) ----
        if refine_blocks > 0:
            self.refine = nn.ModuleList([
                ComplexResBlock(base_filters, filter_width, bias_stddev,
                                weight_var_scale)
                for _ in range(refine_blocks)
            ])
        else:
            self.refine = None

        # ---- 输出头（线性，无激活） ----
        self.out_conv = ComplexConv2d(base_filters, output_dim, filter_width,
                                      padding=pad, bias=True)
        if stem_skip:
            # full-res stem bypass: inject input high-freq info into output.
            # zero-initialized so resuming from an existing ckpt is a no-op.
            self.stem_skip_conv = ComplexConv2d(base_filters, output_dim, 1,
                                                bias=True)
            with torch.no_grad():
                self.stem_skip_conv.conv_real.weight.zero_()
                self.stem_skip_conv.conv_imag.weight.zero_()
                self.stem_skip_conv.conv_real.bias.zero_()
                self.stem_skip_conv.conv_imag.bias.zero_()
        else:
            self.stem_skip_conv = None
        if out_bn:
            # 与 ComplexHoloNet 最后一层一致：conv + BN（无激活）
            self.out_bn = ComplexBatchNorm2d(output_dim)
        else:
            self.out_bn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 输入重归一化（与 ComplexHoloNet 一致）
        x = x - 0.5
        x = torch.complex(x, torch.zeros_like(x))

        # 2. 对齐到 2^depth 的整数倍（仅右侧/底部 padding，最后裁剪）
        orig_h, orig_w = x.shape[-2:]
        mult = 2 ** self.depth
        pad_h = (mult - orig_h % mult) % mult
        pad_w = (mult - orig_w % mult) % mult
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # 3. 编码器（记录各级跳跃连接特征）
        skips = []
        h = self.stem_act(self.stem_bn(self.stem_conv(x)))
        skips.append(h)
        for i in range(self.depth):
            h = self.encoder_blocks[i](h)
            skips.append(h)
            h = self.down_convs[i](h)

        # 4. 瓶颈
        h = self.bottleneck(h)
        if self.attention is not None:
            h = self.attention(h)

        # 5. 解码器（上采样 + 跳跃连接 + 残差块）
        for j in range(self.depth):
            h = self.up_convs[j](h)
            skip = skips[self.depth - j]
            h = torch.cat([h, skip], dim=1)
            h = self.decoder_blocks[j](self.reduce_convs[j](h))

        # 6. 输出头并裁剪回原始分辨率
        # 6. full-res refine (optional)
        if self.refine is not None:
            for blk in self.refine:
                h = blk(h)

        # 7. output head: decoder out + full-res stem bypass (optional)
        out = self.out_conv(h)
        if self.stem_skip_conv is not None:
            out = out + self.stem_skip_conv(skips[0])
        if self.out_bn is not None:
            out = self.out_bn(out)
        if pad_h or pad_w:
            out = out[..., :orig_h, :orig_w]
        return out
