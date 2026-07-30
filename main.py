// File: main.py
"""
程序总入口：解析命令行参数，根据模式分发到训练（stage1/stage2）、
验证、评估或导出 ONNX 等功能。
支持通过 --complex 标志启用复数神经网络（ComplexHoloNet/ComplexDDPMNet）。
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


def main():
    parser = argparse.ArgumentParser(description="Tensor Holography PyTorch")
    # 全局复数标志（不影响现有子命令的参数）
    parser.add_argument("--complex", action="store_true", default=False,
                        help="Use complex-valued networks (ComplexHoloNet, ComplexDDPMNet)")

    subparsers = parser.add_subparsers(dest="mode", required=True, help="Operating mode")

    # 训练阶段一
    parser_s1 = subparsers.add_parser("train_stage1")
    parser_s1.add_argument("--model-name", default="full_loss")
    parser_s1.add_argument("--dataset-res", type=int, default=192)
    parser_s1.add_argument("--pitch", type=float, default=0.008)
    parser_s1.add_argument("--num-layers", type=int, default=30)
    parser_s1.add_argument("--num-filters-per-layer", type=int, default=24)
    parser_s1.add_argument("--num-epochs", type=int, default=4050)
    parser_s1.add_argument("--batch", type=int, default=2)
    parser_s1.add_argument("--learning-rate", type=float, default=1e-4)
    parser_s1.add_argument("--restore", action="store_true")
    parser_s1.add_argument("--ckpt-dir", default=None)
    parser_s1.add_argument("--num-iter-per-test", type=int, default=1000)
    parser_s1.add_argument("--active-max-ldi-layer", type=int, default=0)

    # 训练阶段二
    parser_s2 = subparsers.add_parser("train_stage2")
    parser_s2.add_argument("--model-name", default="full_loss")
    parser_s2.add_argument("--dataset-res", type=int, default=192)
    parser_s2.add_argument("--pitch", type=float, default=0.008)
    parser_s2.add_argument("--num-layers", type=int, default=30)
    parser_s2.add_argument("--num-filters-per-layer", type=int, default=24)
    parser_s2.add_argument("--batch", type=int, default=2)
    parser_s2.add_argument("--learning-rate", type=float, default=1e-4)
    parser_s2.add_argument("--stage1-ckpt", type=str, required=True)
    parser_s2.add_argument("--activate-ddpm", action="store_true")
    parser_s2.add_argument("--bypass-ddpm-network", action="store_true")
    parser_s2.add_argument("--padding", type=int, default=0)
    parser_s2.add_argument("--depth-shift", type=float, default=12.0)
    parser_s2.add_argument("--stage2-ckpt-dir", default=None)
    parser_s2.add_argument("--restore-stage2", action="store_true")
    parser_s2.add_argument("--stage2-epochs", type=int, default=50)
    parser_s2.add_argument("--joint-epochs", type=int, default=200)
    parser_s2.add_argument("--restore-stage1", action="store_true")
    parser_s2.add_argument("--epoch-to-start-ddpm", type=int, default=3000)
    parser_s2.add_argument("--num-iter-per-test", type=int, default=500)

    # 验证
    parser_val = subparsers.add_parser("validate")
    parser_val.add_argument("--mode", type=str, required=True, choices=["stage1", "stage2"])
    parser_val.add_argument("--ckpt-path", type=str, required=True)
    parser_val.add_argument("--ddpm-ckpt-path", type=str, default=None)
    parser_val.add_argument("--model-name", default="full_loss")
    parser_val.add_argument("--dataset-res", type=int, default=192)
    parser_val.add_argument("--pitch", type=float, default=0.008)
    parser_val.add_argument("--num-layers", type=int, default=30)
    parser_val.add_argument("--num-filters-per-layer", type=int, default=24)
    parser_val.add_argument("--batch", type=int, default=2)
    parser_val.add_argument("--padding", type=int, default=0)
    parser_val.add_argument("--depth-shift", type=float, default=12.0)
    parser_val.add_argument("--activate-ddpm", action="store_true")
    parser_val.add_argument("--bypass-ddpm-network", action="store_true")

    # 单张评估
    parser_eval = subparsers.add_parser("evaluate")
    parser_eval.add_argument("--ckpt-path", type=str, required=True)
    parser_eval.add_argument("--ddpm-ckpt-path", type=str, default=None)
    parser_eval.add_argument("--activate-ddpm", action="store_true")
    parser_eval.add_argument("--bypass-ddpm-network", action="store_true")
    parser_eval.add_argument("--num-layers", type=int, default=30)
    parser_eval.add_argument("--num-filters-per-layer", type=int, default=24)
    parser_eval.add_argument("--active-max-ldi-layer", type=int, default=0)
    parser_eval.add_argument("--eval-res-h", type=int, default=1080)
    parser_eval.add_argument("--eval-res-w", type=int, default=1920)
    parser_eval.add_argument("--eval-rgb-path", type=str, required=True)
    parser_eval.add_argument("--eval-depth-path", type=str, required=True)
    parser_eval.add_argument("--eval-output-path", type=str, required=True)
    parser_eval.add_argument("--eval-depth-shift", type=float, default=0.0)
    parser_eval.add_argument("--padding", type=int, default=0)
    parser_eval.add_argument("--use-maimone-dpm", action="store_true")
    parser_eval.add_argument("--use-bldpm", action="store_true")
    parser_eval.add_argument("--adaptive-phs-shift", action="store_true")
    parser_eval.add_argument("--gaussian-sigma", type=float, default=0.0)
    parser_eval.add_argument("--gaussian-width", type=int, default=3)
    parser_eval.add_argument("--phs-max", type=float, default=2.0)
    parser_eval.add_argument("--k", type=float, default=1.0)
    parser_eval.add_argument("--pitch", type=float, default=0.008)

    # 导出 ONNX
    parser_export = subparsers.add_parser("export")
    parser_export.add_argument("--ckpt-path", type=str, required=True)
    parser_export.add_argument("--ddpm-ckpt-path", type=str, default=None)
    parser_export.add_argument("--output", type=str, default="model.onnx")
    parser_export.add_argument("--res-h", type=int, default=1080)
    parser_export.add_argument("--res-w", type=int, default=1920)
    parser_export.add_argument("--pad", type=int, default=0)
    parser_export.add_argument("--input-dim", type=int, default=4)
    parser_export.add_argument("--activate-ddpm", action="store_true")
    parser_export.add_argument("--num-layers", type=int, default=30)
    parser_export.add_argument("--num-filters-per-layer", type=int, default=24)

    args = parser.parse_args()

    # 根据全局 --complex 标志设置环境变量，供子模块内部判断使用复数模型
    if args.complex:
        os.environ["HOLONET_COMPLEX"] = "1"
    else:
        os.environ.pop("HOLONET_COMPLEX", None)  # 确保无残留

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_argv(args_dict, exclude_keys=('mode', 'complex')):
        """将参数字典正确转换为命令行参数列表，正确处理 store_true 布尔值。
        注意：complex 是全局标志，不传入子模块（通过环境变量传递）。
        """
        cmd = [sys.argv[0]]
        for k, v in args_dict.items():
            if k in exclude_keys:
                continue
            if isinstance(v, bool):
                if v:
                    cmd.append(f'--{k.replace("_", "-")}')
            else:
                if v is not None:
                    cmd.append(f'--{k.replace("_", "-")}={v}')
        return cmd

    if args.mode == "train_stage1":
        sys.argv = build_argv(vars(args))
        stage1_module.main()
    elif args.mode == "train_stage2":
        sys.argv = build_argv(vars(args))
        stage2_module.main()
    elif args.mode == "validate":
        sys.argv = build_argv(vars(args))
        validate_module.main()
    elif args.mode == "evaluate":
        run_evaluate(args)  # 直接传 args，evaluate 内部可读取环境变量
    elif args.mode == "export":
        sys.argv = build_argv(vars(args))
        export_module.main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()