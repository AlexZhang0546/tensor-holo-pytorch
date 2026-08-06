# -*- coding: utf-8 -*-
"""部署分辨率（1080x1920）ONNX 导出与 torch/ONNX 数值对比。"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.holonet import ComplexHoloNet
from src.models.ddpm_net import ComplexDDPMNet
from src.eval.export_onnx import export_onnx


def main():
    ckpt_path = sys.argv[1]
    res_h, res_w = 1080, 1920
    ckpt = torch.load(ckpt_path, map_location="cpu")
    holonet = ComplexHoloNet(input_dim=4, num_layers=30, num_filters_per_layer=24).eval()
    holonet.load_state_dict(ckpt["holonet_state_dict"])
    ddpm = ComplexDDPMNet(input_dim=3, output_dim=3, num_layers=8, num_filters_per_layer=8).eval()
    ddpm.load_state_dict(ckpt["ddpm_net_state_dict"])

    onnx_path = "/tmp/deploy_s2.onnx"
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
    t0 = time.time()
    export_onnx(holonet, ddpm, onnx_path, res_h=res_h, res_w=res_w,
                pad=0, input_dim=4, device="cpu")
    print("export time %.1fs" % (time.time() - t0))

    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, 4, res_h, res_w, generator=g) * 0.25 + 0.5

    t0 = time.time()
    with torch.no_grad():
        out = ddpm(holonet(x))
        t_r, t_i = out.real.numpy(), out.imag.numpy()
    print("torch forward time %.1fs" % (time.time() - t0))

    import onnxruntime as ort
    t0 = time.time()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    o_r, o_i = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})
    print("onnx forward time %.1fs" % (time.time() - t0))

    print("shapes: torch %s / onnx %s" % (t_r.shape, o_r.shape))
    print("finite onnx:", np.isfinite(o_r).all() and np.isfinite(o_i).all())
    dr = float(np.abs(t_r - o_r).max())
    di = float(np.abs(t_i - o_i).max())
    amp_t = np.hypot(t_r, t_i)
    amp_o = np.hypot(o_r, o_i)
    print("max_abs_diff real=%.4e imag=%.4e" % (dr, di))
    print("amp: torch max=%.4f mean=%.4f | onnx max=%.4f mean=%.4f | amp_maxdiff=%.4e"
          % (amp_t.max(), amp_t.mean(), amp_o.max(), amp_o.mean(),
             np.abs(amp_t - amp_o).max()))
    print("RESULT: %s" % ("PASS" if max(dr, di) <= 2e-4 else "FAIL"))


if __name__ == "__main__":
    main()
