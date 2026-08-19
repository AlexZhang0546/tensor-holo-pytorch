# File: main.py
"""
Tensor Holography PyTorch 程序入口。

CLI 参数实现参照原始 TensorFlow 项目 main_v2.py：
  - 单 argparse + 模式开关：--train-mode / --validate-mode-s1 / --validate-mode-s2 /
    --eval-mode / --export-mode；
  - 参数名与默认值与原始实现保持一致，例如 --epoch_to_start_ddpm_training、
    --train-depth-shift、--phs-max、--trt-res-h/w 等；
  - checkpoint 目录命名遵循 main_v2.py 的 checkpoint_base_path 约定。
"""

import argparse
import os
import sys
import torch
import numpy as np

# 为方便导入，将 src 目录加入路径
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# 导入各模块入口函数
import src.train.stage1 as stage1_module
import src.train.stage2 as stage2_module
import src.eval.validate as validate_module
from src.eval.evaluate import evaluate as run_evaluate
import src.eval.export_onnx as export_module

# ----------------------------------------------------------------------
# checkpoint 路径约定（与 main_v2.py 的 checkpoint_base_path 一致）
# ----------------------------------------------------------------------
def _ckpt_dir_stage1(cur_dir, model_name, pitch, num_layers, num_filters):
    return os.path.join(
        cur_dir, "model",
        "ckpt_%s_pitch_%d_layers_%d_filters_%d_stage1" % (
            model_name, int(pitch * 1000), num_layers, num_filters))


def _ckpt_dir_stage2(cur_dir, model_name, pitch, num_layers, num_filters,
                     depth_shift, bypass):
    name = "ckpt_%s_pitch_%d_layers_%d_filters_%d_ddpm_%d" % (
        model_name, int(pitch * 1000), num_layers, num_filters, int(depth_shift))
    if bypass:
        name += "_bypass"
    return os.path.join(cur_dir, "model", name)


def _default_ckpt_stage1(cur_dir, args):
    return os.path.join(
        _ckpt_dir_stage1(cur_dir, args.model_name, args.pitch,
                         args.num_layers, args.num_filters_per_layer),
        "stage1_latest.pth")


def _default_ckpt_stage2(cur_dir, args):
    return os.path.join(
        _ckpt_dir_stage2(cur_dir, args.model_name, args.pitch,
                         args.num_layers, args.num_filters_per_layer,
                         args.train_depth_shift, args.bypass_ddpm_network),
        "stage2_joint_latest.pth")


# ----------------------------------------------------------------------
# 原始风格（参照 main_v2.py）的单 parser
# ----------------------------------------------------------------------
def build_original_parser():
    parser = argparse.ArgumentParser(
        description="Tensor Holography PyTorch (arguments follow main_v2.py)")
    cur_dir = os.path.dirname(os.path.realpath(__file__))

    # ---- 模式开关（与 main_v2.py 一致）----
    parser.add_argument("--train-mode", action="store_true",
                        help="Run in training mode")
    parser.add_argument("--validate-mode-s1", action="store_true",
                        help="Run in validation mode for stage 1")
    parser.add_argument("--validate-mode-s2", action="store_true",
                        help="Run in validation mode for stage 2")
    parser.add_argument("--eval-mode", action="store_true",
                        help="Run in evaluation mode")
    parser.add_argument("--export-mode", action="store_true",
                        help="Export model for onnx")

    # ---- 训练阶段选择（PyTorch 移植将训练拆为 stage1/stage2 两个脚本）----
    parser.add_argument("--train-stage", choices=["stage1", "stage2"],
                        default="stage1",
                        help="Which training stage to run (port extension)")

    # ---- 数据集参数 ----
    parser.add_argument("--dataset-res", default=192, type=int,
                        help="dataset image resolution")
    parser.add_argument("--pitch", default=0.008, type=float,
                        help="pixel pitch in mm")

    # ---- 模型参数 ----
    parser.add_argument("--num-filters-per-layer", default=24, type=int,
                        help="Number of filters per layer")
    parser.add_argument("--num-layers", default=30, type=int,
                        help="Number layers")
    parser.add_argument("--model-name", default="full_loss", type=str,
                        help="Model name")
    parser.add_argument("--model-arch", default="holonet",
                        choices=["holonet", "unet"],
                        help="Main network architecture "
                             "(holonet: original, unet: multi-scale ComplexUNet)")
    parser.add_argument("--unet-depth", default=3, type=int,
                        help="ComplexUNet downsample levels")
    parser.add_argument("--unet-base-filters", default=24, type=int,
                        help="ComplexUNet base filters (shallowest level)")
    parser.add_argument("--unet-attention", action="store_true",
                        help="Enable bottleneck self-attention in ComplexUNet")
    parser.add_argument("--weight-fs", default=20.0, type=float,
                        help="Focal stack loss weight (stage2)")
    parser.add_argument("--weight-fs-tv", default=20.0, type=float,
                        help="Focal stack TV loss weight (stage2)")
    parser.add_argument("--weight-std", default=0.02, type=float,
                        help="DDPM phase std regularizer weight (stage2)")
    parser.add_argument("--weight-mean", default=0.03, type=float,
                        help="DDPM phase mean regularizer weight (stage2)")
    parser.add_argument("--weight-holo-joint", default=0.0, type=float,
                        help="HoloNet-to-GT fidelity anchor weight in joint loss "
                             "(0 = original behavior)")
    parser.add_argument("--weight-ssim", default=0.0, type=float,
                        help="Weight of (1 - SSIM_img) in the joint loss "
                             "(0 = original behavior)")
    parser.add_argument("--weight-holo", default=1.0, type=float,
                        help="Stage-1 complex-field (phase-aligned) loss weight "
                             "(0 = drop the holo fidelity term)")
    parser.add_argument("--holo-method", default="phase_aligned", type=str,
                        choices=["phase_aligned", "magnitude_phase", "complex_diff"],
                        help="Stage-1 holo fidelity loss method")
    parser.add_argument("--freeze-bn", action="store_true",
                        help="Freeze BatchNorm running stats during stage1 "
                             "fine-tuning (model stays in eval mode)")
    parser.add_argument("--weight-decay", default=0.0, type=float,
                        help="L2 weight decay for the stage1 optimizer")
    parser.add_argument("--deterministic-depths", action="store_true",
                        help="Use deterministic focal-stack depth sampling")
    parser.add_argument("--unet-out-bn", action="store_true",
                        help="Add BatchNorm to the UNet output layer")
    parser.add_argument("--unet-stem-skip", action="store_true",
                        help="Add full-res stem bypass to the UNet output "
                             "(zero-initialized; safe to resume from ckpt)")
    parser.add_argument("--unet-refine-blocks", default=0, type=int,
                        help="Full-res refine blocks before the UNet output "
                             "head (0 = disabled)")
    parser.add_argument("--unet-tail-blocks", default=0, type=int,
                        help="Full-res holonet-style tail layers after the "
                             "UNet decoder (0 = disabled)")
    parser.add_argument("--unet-global-in", action="store_true",
                        help="Concat raw normalized input at the UNet output "
                             "head (holonet-style global skip)")
    parser.add_argument("--aperture-radius", default=None, type=int,
                        help="Frequency-domain aperture radius (pixels); "
                             "None = min(res_h,res_w)//2")

    # ---- 训练参数 ----
    parser.add_argument("--num-epochs", default=4050, type=int,
                        help="Number of training epochs (stage 1)")
    parser.add_argument("--train-depth-shift", default=12.0, type=float,
                        help="Depth shift (in mm) from the predicted midpoint "
                             "hologram during stage-2 training")
    parser.add_argument("--epoch_to_start_ddpm_training", default=3000, type=int,
                        help="The epoch to start stage-2 training")
    parser.add_argument("--stage2-epochs", default=50, type=int,
                        help="Number of epochs for identity pretraining (port extension)")
    parser.add_argument("--joint-epochs", default=200, type=int,
                        help="Number of epochs for joint training (port extension)")
    parser.add_argument("--batch", default=2, type=int,
                        help="Training batch size (port extension)")
    parser.add_argument("--learning-rate", default=1e-4, type=float,
                        help="Learning rate (port extension)")
    parser.add_argument("--num-iter-per-test", default=1000, type=int,
                        help="Number of iterations per validation (port extension)")
    parser.add_argument("--restore", action="store_true",
                        help="Restore stage-1 checkpoint (port extension)")
    parser.add_argument("--restore-stage1", action="store_true",
                        help="Load stage-1 weights before stage-2 (port extension)")
    parser.add_argument("--restore-stage2", action="store_true",
                        help="Resume stage-2 training (port extension)")
    parser.add_argument("--stage1-ckpt", type=str, default=None,
                        help="Path to stage-1 checkpoint (default: derived from "
                             "model params)")
    parser.add_argument("--stage2-ckpt-dir", type=str, default=None,
                        help="Stage-2 checkpoint directory (port extension)")
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="Stage-1 checkpoint directory (port extension)")

    # ---- DDPM 相关参数 ----
    parser.add_argument("--active-max-ldi-layer", default=0, type=int,
                        help="Active max LDI layer")
    parser.add_argument("--activate-ddpm", action="store_true",
                        help="Load ddpm network together with hologram rendering "
                             "network; depth shift specified by --train-depth-shift")
    parser.add_argument("--bypass-ddpm-network", action="store_true",
                        help="Train/evaluate ddpm without using ddpm network "
                             "(typical for 0 mm offset)")
    parser.add_argument("--padding", default=0, type=int,
                        help="Padding to the hologram to accommodate "
                             "out-of-frame diffraction")

    # ---- 验证/推理 checkpoint 路径（port extension）----
    parser.add_argument("--ckpt-path", type=str, default=None,
                        help="Path to model checkpoint (default: derived from "
                             "model params)")
    parser.add_argument("--ddpm-ckpt-path", type=str, default=None,
                        help="Path to DDPM checkpoint")

    # ---- 评估参数（与 main_v2.py 一致）----
    parser.add_argument("--eval-res-h", default=1080, type=int,
                        help="Input image height in evaluation mode")
    parser.add_argument("--eval-res-w", default=1920, type=int,
                        help="Input image width in evaluation mode")
    parser.add_argument("--eval-rgb-path",
                        default=os.path.join(cur_dir, "data", "example_input",
                                             "couch_rgb.png"),
                        help="Input rgb image path in evaluation mode")
    parser.add_argument("--eval-depth-path",
                        default=os.path.join(cur_dir, "data", "example_input",
                                             "couch_depth.png"),
                        help="Input depth image path in evaluation mode")
    parser.add_argument("--eval-output-path",
                        default=os.path.join(cur_dir, "data", "example_input"),
                        help="Output directory for results")
    parser.add_argument("--eval-depth-shift", default=0.0, type=float,
                        help="Depth shift (in mm) from the predicted midpoint "
                             "hologram to the target hologram plane")
    parser.add_argument("--gaussian-sigma", default=0.0, type=float,
                        help="Sigma of Gaussian kernel used by AA-DPM")
    parser.add_argument("--gaussian-width", default=3, type=int,
                        help="Width of Gaussian kernel used by AA-DPM")
    parser.add_argument("--phs-max", default=2.0, type=float,
                        help="Maximum phase modulation of SLM in unit of pi")
    parser.add_argument("--use-maimone-dpm", action="store_true",
                        help="Use DPM of Maimone et al. 2017")
    parser.add_argument("--k", default=1.0, type=float,
                        help="k for generating Fourier-space mask used by BL-DPM")
    parser.add_argument("--use-bldpm", action="store_true",
                        help="Use BL-DPM of Sui et al. 2021")
    parser.add_argument("--adaptive-phs-shift", action="store_true",
                        help="Adaptive phase shift in DPM (port extension)")

    # ---- 导出参数（与 main_v2.py 的 export 参数一致）----
    parser.add_argument("--trt-res-h", default=1080, type=int,
                        help="Input image height in export mode")
    parser.add_argument("--trt-res-w", default=1920, type=int,
                        help="Input image width in export mode")
    parser.add_argument("--output", default="model.onnx", type=str,
                        help="ONNX output path (port extension)")
    parser.add_argument("--input-dim", default=4, type=int,
                        help="Model input channels (port extension)")

    # ---- 调试 ----
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the translated command and exit "
                             "without running")
    parser.add_argument("--ddpm-arch", default="real", choices=["real", "complex"],
                        help="DDPM architecture (real: paper amp/phase CNN; complex: complex CNN)")
    parser.add_argument("--ddpm-bn", default="tf", choices=["tf", "batch"],
                        help="DDPM BN semantics (tf: like main_v2.py; batch: PyTorch BN)")
    return parser


# ----------------------------------------------------------------------
# 原始风格参数的翻译（映射到各子模块的 argv / Namespace）
# ----------------------------------------------------------------------
def _build_stage1_argv(args):
    argv = [
        "--model-name=%s" % args.model_name,
        "--dataset-res=%d" % args.dataset_res,
        "--pitch=%s" % args.pitch,
        "--num-layers=%d" % args.num_layers,
        "--num-filters-per-layer=%d" % args.num_filters_per_layer,
        "--num-epochs=%d" % args.num_epochs,
        "--batch=%d" % args.batch,
        "--learning-rate=%s" % args.learning_rate,
        "--num-iter-per-test=%d" % args.num_iter_per_test,
        "--active-max-ldi-layer=%d" % args.active_max_ldi_layer,
    ]
    if args.restore:
        argv.append("--restore")
    if args.ckpt_dir:
        argv.append("--ckpt-dir=%s" % args.ckpt_dir)
    argv.extend([
        "--model-arch=%s" % args.model_arch,
        "--unet-depth=%d" % args.unet_depth,
        "--unet-base-filters=%d" % args.unet_base_filters,
    ])
    if args.unet_attention:
        argv.append("--unet-attention")
    argv.append("--weight-ssim=%s" % args.weight_ssim)
    argv.append("--weight-holo=%s" % args.weight_holo)
    argv.append("--holo-method=%s" % args.holo_method)
    if args.freeze_bn:
        argv.append("--freeze-bn")
    argv.append("--weight-decay=%s" % args.weight_decay)
    if args.deterministic_depths:
        argv.append("--deterministic-depths")
    if args.unet_out_bn:
        argv.append("--unet-out-bn")
    if args.unet_stem_skip:
        argv.append("--unet-stem-skip")
    if args.unet_global_in:
        argv.append("--unet-global-in")
    argv.append("--unet-refine-blocks=%d" % args.unet_refine_blocks)
    argv.append("--unet-tail-blocks=%d" % args.unet_tail_blocks)
    return argv


def _build_stage2_argv(args, cur_dir):
    stage1_ckpt = args.stage1_ckpt or _default_ckpt_stage1(cur_dir, args)
    argv = [
        "--model-name=%s" % args.model_name,
        "--weight-fs=%s" % args.weight_fs,
        "--weight-fs-tv=%s" % args.weight_fs_tv,
        "--weight-std=%s" % args.weight_std,
        "--weight-mean=%s" % args.weight_mean,
        "--weight-holo-joint=%s" % args.weight_holo_joint,
        "--weight-ssim=%s" % args.weight_ssim,
        "--dataset-res=%d" % args.dataset_res,
        "--pitch=%s" % args.pitch,
        "--num-layers=%d" % args.num_layers,
        "--num-filters-per-layer=%d" % args.num_filters_per_layer,
        "--batch=%d" % args.batch,
        "--learning-rate=%s" % args.learning_rate,
        "--num-iter-per-test=%d" % args.num_iter_per_test,
        "--depth-shift=%s" % args.train_depth_shift,
        "--epoch-to-start-ddpm=%d" % args.epoch_to_start_ddpm_training,
        "--stage2-epochs=%d" % args.stage2_epochs,
        "--joint-epochs=%d" % args.joint_epochs,
        "--padding=%d" % args.padding,
        "--stage1-ckpt=%s" % stage1_ckpt,
    ]
    if args.activate_ddpm:
        argv.append("--activate-ddpm")
    if args.bypass_ddpm_network:
        argv.append("--bypass-ddpm-network")
    if args.restore_stage1:
        argv.append("--restore-stage1")
    if args.restore_stage2:
        argv.append("--restore-stage2")
    if args.stage2_ckpt_dir:
        argv.append("--stage2-ckpt-dir=%s" % args.stage2_ckpt_dir)
    argv.extend([
        "--model-arch=%s" % args.model_arch,
        "--unet-depth=%d" % args.unet_depth,
        "--unet-base-filters=%d" % args.unet_base_filters,
    ])
    if args.unet_attention:
        argv.append("--unet-attention")
    if args.deterministic_depths:
        argv.append("--deterministic-depths")
    if args.unet_out_bn:
        argv.append("--unet-out-bn")
    if args.unet_stem_skip:
        argv.append("--unet-stem-skip")
    if args.unet_global_in:
        argv.append("--unet-global-in")
    argv.append("--unet-refine-blocks=%d" % args.unet_refine_blocks)
    argv.append("--unet-tail-blocks=%d" % args.unet_tail_blocks)
    argv.append("--ddpm-arch=%s" % args.ddpm_arch)
    argv.append("--ddpm-bn=%s" % args.ddpm_bn)
    if args.aperture_radius is not None:
        argv.append("--aperture-radius=%d" % args.aperture_radius)
    return argv


def _build_validate_argv(args, val_mode, cur_dir):
    if args.ckpt_path:
        ckpt = args.ckpt_path
    else:
        ckpt = (_default_ckpt_stage1(cur_dir, args) if val_mode == "stage1"
                else _default_ckpt_stage2(cur_dir, args))
    argv = [
        "--val-mode=%s" % val_mode,
        "--ckpt-path=%s" % ckpt,
        "--model-name=%s" % args.model_name,
        "--dataset-res=%d" % args.dataset_res,
        "--pitch=%s" % args.pitch,
        "--num-layers=%d" % args.num_layers,
        "--num-filters-per-layer=%d" % args.num_filters_per_layer,
        "--batch=%d" % args.batch,
        "--padding=%d" % args.padding,
        "--depth-shift=%s" % args.train_depth_shift,
    ]
    if args.activate_ddpm:
        argv.append("--activate-ddpm")
    if args.bypass_ddpm_network:
        argv.append("--bypass-ddpm-network")
    if args.ddpm_ckpt_path:
        argv.append("--ddpm-ckpt-path=%s" % args.ddpm_ckpt_path)
    argv.append("--ddpm-arch=%s" % args.ddpm_arch)
    argv.append("--ddpm-bn=%s" % args.ddpm_bn)
    if args.aperture_radius is not None:
        argv.append("--aperture-radius=%d" % args.aperture_radius)
    argv.extend([
        "--model-arch=%s" % args.model_arch,
        "--unet-depth=%d" % args.unet_depth,
        "--unet-base-filters=%d" % args.unet_base_filters,
    ])
    if args.unet_attention:
        argv.append("--unet-attention")
    if args.unet_stem_skip:
        argv.append("--unet-stem-skip")
    if args.unet_global_in:
        argv.append("--unet-global-in")
    argv.append("--unet-refine-blocks=%d" % args.unet_refine_blocks)
    argv.append("--unet-tail-blocks=%d" % args.unet_tail_blocks)
    return argv


def _build_eval_namespace(args, cur_dir):
    ckpt = args.ckpt_path or _default_ckpt_stage1(cur_dir, args)
    return argparse.Namespace(
        ckpt_path=ckpt,
        ddpm_ckpt_path=args.ddpm_ckpt_path,
        ddpm_arch=args.ddpm_arch,
        ddpm_bn=args.ddpm_bn,
        activate_ddpm=args.activate_ddpm,
        bypass_ddpm_network=args.bypass_ddpm_network,
        num_layers=args.num_layers,
        num_filters_per_layer=args.num_filters_per_layer,
        active_max_ldi_layer=args.active_max_ldi_layer,
        eval_res_h=args.eval_res_h,
        eval_res_w=args.eval_res_w,
        eval_rgb_path=args.eval_rgb_path,
        eval_depth_path=args.eval_depth_path,
        eval_output_path=args.eval_output_path,
        eval_depth_shift=args.eval_depth_shift,
        padding=args.padding,
        use_maimone_dpm=args.use_maimone_dpm,
        use_bldpm=args.use_bldpm,
        adaptive_phs_shift=args.adaptive_phs_shift,
        gaussian_sigma=args.gaussian_sigma,
        gaussian_width=args.gaussian_width,
        phs_max=args.phs_max,
        k=args.k,
        pitch=args.pitch,
        model_arch=args.model_arch,
        unet_depth=args.unet_depth,
        unet_base_filters=args.unet_base_filters,
        unet_attention=args.unet_attention,
        unet_stem_skip=args.unet_stem_skip,
        unet_refine_blocks=args.unet_refine_blocks,
        unet_global_in=args.unet_global_in,
    )


def _build_export_argv(args, cur_dir):
    if args.ckpt_path:
        ckpt = args.ckpt_path
    else:
        ckpt = (_default_ckpt_stage2(cur_dir, args) if args.activate_ddpm
                else _default_ckpt_stage1(cur_dir, args))
    argv = [
        "--ckpt-path=%s" % ckpt,
        "--output=%s" % args.output,
        "--res-h=%d" % args.trt_res_h,
        "--res-w=%d" % args.trt_res_w,
        "--pad=%d" % args.padding,
        "--input-dim=%d" % args.input_dim,
        "--num-layers=%d" % args.num_layers,
        "--num-filters-per-layer=%d" % args.num_filters_per_layer,
    ]
    if args.activate_ddpm:
        argv.append("--activate-ddpm")
    if args.ddpm_ckpt_path:
        argv.append("--ddpm-ckpt-path=%s" % args.ddpm_ckpt_path)
    argv.append("--ddpm-arch=%s" % args.ddpm_arch)
    argv.append("--ddpm-bn=%s" % args.ddpm_bn)
    argv.extend([
        "--model-arch=%s" % args.model_arch,
        "--unet-depth=%d" % args.unet_depth,
        "--unet-base-filters=%d" % args.unet_base_filters,
    ])
    if args.unet_attention:
        argv.append("--unet-attention")
    return argv


def _original_style_main():
    args = build_original_parser().parse_args()
    cur_dir = os.path.dirname(os.path.realpath(__file__))

    plan = None
    if args.train_mode:
        if args.train_stage == "stage1":
            plan = ("train_stage1", _build_stage1_argv(args))
        else:
            plan = ("train_stage2", _build_stage2_argv(args, cur_dir))
    elif args.validate_mode_s1:
        plan = ("validate", _build_validate_argv(args, "stage1", cur_dir))
    elif args.validate_mode_s2:
        plan = ("validate", _build_validate_argv(args, "stage2", cur_dir))
    elif args.eval_mode:
        plan = ("evaluate", _build_eval_namespace(args, cur_dir))
    elif args.export_mode:
        plan = ("export", _build_export_argv(args, cur_dir))

    if plan is None:
        build_original_parser().print_help()
        return

    if args.dry_run:
        mode, payload = plan
        print("mode:", mode)
        if isinstance(payload, argparse.Namespace):
            for k, v in vars(payload).items():
                print("  --%s=%r" % (k, v))
        else:
            print("argv:", " ".join(payload))
        return

    mode, payload = plan
    if mode == "evaluate":
        run_evaluate(payload)
        return
    sys.argv = [sys.argv[0]] + payload
    {"train_stage1": stage1_module, "train_stage2": stage2_module,
     "validate": validate_module, "export": export_module}[mode].main()


def main():
    # 原始风格（参照 main_v2.py）
    _original_style_main()


if __name__ == "__main__":
    main()
