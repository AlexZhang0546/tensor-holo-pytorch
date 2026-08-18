"""
主网络工厂：按 arch 选择 ComplexHoloNet 或 ComplexUNet。
两阶段训练、验证、评估、导出统一从这里构建主网络，保证配置一致。
"""

from src.models.holonet import ComplexHoloNet
from src.models.complex_unet import ComplexUNet


def build_main_net(*,
                   arch: str = "holonet",
                   input_dim: int,
                   num_layers: int = 30,
                   num_filters_per_layer: int = 24,
                   interleave_rate: int = 1,
                   filter_width: int = 3,
                   bias_stddev: float = 0.01,
                   weight_var_scale: float = 0.25,
                   unet_depth: int = 3,
                   unet_base_filters: int = 24,
                   unet_attention: bool = False,
                   unet_out_bn: bool = False,
                   unet_stem_skip: bool = False,
                   unet_refine_blocks: int = 0,
                   unet_global_in: bool = False,
                   unet_tail_blocks: int = 0):
    """构建主网络。

    arch="unet" 时使用多尺度 ComplexUNet（depth 为下采样级数，
    base_filters 为最浅层通道数，可启用瓶颈自注意力）；
    否则使用原 ComplexHoloNet。
    """
    if arch == "unet":
        return ComplexUNet(
            input_dim=input_dim,
            output_dim=3,
            depth=unet_depth,
            base_filters=unet_base_filters,
            filter_width=filter_width,
            bias_stddev=bias_stddev,
            weight_var_scale=weight_var_scale,
            use_attention=unet_attention,
            out_bn=unet_out_bn,
            stem_skip=unet_stem_skip,
            refine_blocks=unet_refine_blocks,
            global_in=unet_global_in,
            tail_blocks=unet_tail_blocks,
        )
    return ComplexHoloNet(
        input_dim=input_dim,
        num_layers=num_layers,
        num_filters_per_layer=num_filters_per_layer,
        interleave_rate=interleave_rate,
        filter_width=filter_width,
        bias_stddev=bias_stddev,
        weight_var_scale=weight_var_scale,
    )
