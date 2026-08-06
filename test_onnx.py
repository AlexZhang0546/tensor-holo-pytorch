# -*- coding: utf-8 -*-
"""
ONNX 导出正确性测试：
  1. 用 export_onnx 导出 checkpoint 为 ONNX；
  2. 在相同输入上对比 PyTorch 模型与 ONNX Runtime 的 real_out / imag_out；
  3. 打印最大/平均绝对误差、相对误差与振幅 SSIM，并给出 PASS/FAIL。

用法（在项目根目录、holography 环境下）：
    python test_onnx.py \
        --ckpt model/ckpt_full_loss_pitch_8_layers_30_filters_24_stage1/stage1_latest.pth
    python test_onnx.py \
        --ckpt model/stage2_test_v2/stage2_joint_latest.pth --activate-ddpm
"""

import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.holonet import ComplexHoloNet
from src.models.ddpm_net import ComplexDDPMNet
from src.eval.export_onnx import export_onnx
from src.utils.metrics import compute_ssim


def build_models(ckpt_path, activate_ddpm, input_dim=4):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    holonet = ComplexHoloNet(
        input_dim=input_dim, num_layers=30, num_filters_per_layer=24
    ).eval()
    if "model_state_dict" in ckpt:
        holonet.load_state_dict(ckpt["model_state_dict"])
    elif "holonet_state_dict" in ckpt:
        holonet.load_state_dict(ckpt["holonet_state_dict"])
    else:
        holonet.load_state_dict(ckpt)

    ddpm_net = None
    if activate_ddpm:
        ddpm_net = ComplexDDPMNet(
            input_dim=3, output_dim=3, num_layers=8, num_filters_per_layer=8
        ).eval()
        if "ddpm_net_state_dict" in ckpt:
            ddpm_net.load_state_dict(ckpt["ddpm_net_state_dict"])
        else:
            raise ValueError("checkpoint 中未找到 ddpm_net_state_dict")
    return holonet, ddpm_net


def make_inputs(res, batch, input_dim=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, input_dim, res, res, generator=g) * 0.25 + 0.5


def compare(tag, torch_model, onnx_path, x, tol=2e-4):
    import onnxruntime as ort

    with torch.no_grad():
        tr, ti = torch_model(x)
        tr, ti = tr.numpy(), ti.numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out = sess.run(None, {in_name: x.numpy()})
    orr, oi = out[0], out[1]

    assert tr.shape == orr.shape and ti.shape == oi.shape, \
        f"shape mismatch: torch {tr.shape} vs onnx {orr.shape}"

    dr = float(np.abs(tr - orr).max())
    di = float(np.abs(ti - oi).max())
    mr = float(np.abs(tr - orr).mean())
    mi = float(np.abs(ti - oi).mean())
    scale = max(float(np.abs(tr).max()), float(np.abs(ti).max()), 1e-6)
    rel = max(dr, di) / scale

    amp_t = torch.from_numpy(np.sqrt(tr ** 2 + ti ** 2)).unsqueeze(0)
    amp_o = torch.from_numpy(np.sqrt(orr ** 2 + oi ** 2)).unsqueeze(0)
    ssim = float(compute_ssim(
        amp_t, amp_o, data_range=max(float(amp_t.max()), 1e-6)).item())

    ok = max(dr, di) <= tol and ssim > 0.999
    print(f"[{tag}] shape={orr.shape} "
          f"max_abs_diff real={dr:.3e} imag={di:.3e} | "
          f"mean_abs_diff real={mr:.3e} imag={mi:.3e} | "
          f"rel_error={rel:.3e} | amp_ssim={ssim:.6f} | "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="ONNX export correctness test")
    ap.add_argument("--ckpt", required=True, help="path to .pth checkpoint")
    ap.add_argument("--activate-ddpm", action="store_true")
    ap.add_argument("--res", type=int, default=384)
    ap.add_argument("--output", default=None, help="onnx output path")
    ap.add_argument("--tol", type=float, default=2e-4)
    args = ap.parse_args()

    holonet, ddpm_net = build_models(args.ckpt, args.activate_ddpm)
    tag = "stage1" if not args.activate_ddpm else "stage2+ddpm"
    onnx_path = args.output or f"/tmp/{tag}_test.onnx"
    if os.path.exists(onnx_path):
        os.remove(onnx_path)

    print(f"exporting {tag} -> {onnx_path} (res={args.res}) ...")
    export_onnx(
        holonet, ddpm_net, onnx_path,
        res_h=args.res, res_w=args.res, pad=0, input_dim=4, device="cpu",
    )
    print("export ok")

    results = []
    for batch in (1, 2):
        x = make_inputs(args.res, batch)
        results.append(compare(
            f"{tag} batch={batch}", holonet, onnx_path, x, tol=args.tol))
    if not all(results):
        print("RESULT: FAIL")
        sys.exit(1)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
