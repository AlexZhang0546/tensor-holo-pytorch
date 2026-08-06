# -*- coding: utf-8 -*-
"""诊断 stage2+DDPM 的 ONNX 与 PyTorch 输出差异。"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.holonet import ComplexHoloNet
from src.models.ddpm_net import ComplexDDPMNet
from src.eval.export_onnx import export_onnx


def main():
    ckpt_path = sys.argv[1]
    res = int(sys.argv[2]) if len(sys.argv) > 2 else 192
    ckpt = torch.load(ckpt_path, map_location="cpu")

    holonet = ComplexHoloNet(input_dim=4, num_layers=30, num_filters_per_layer=24).eval()
    holonet.load_state_dict(ckpt["holonet_state_dict"])
    ddpm = ComplexDDPMNet(input_dim=3, output_dim=3, num_layers=8, num_filters_per_layer=8).eval()
    ddpm.load_state_dict(ckpt["ddpm_net_state_dict"])
    print("torch models eval:", holonet.training, ddpm.training)

    onnx_path = "/tmp/diag_s2.onnx"
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
    export_onnx(holonet, ddpm, onnx_path, res_h=res, res_w=res, pad=0,
                input_dim=4, device="cpu")
    print("export done")

    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, 4, res, res, generator=g) * 0.25 + 0.5

    with torch.no_grad():
        holo = holonet(x)
        tr, ti = holo.real.numpy(), holo.imag.numpy()
        ddpm_out = ddpm(holo)
        dr_t, di_t = ddpm_out.real.numpy(), ddpm_out.imag.numpy()

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out = sess.run(None, {in_name: x.numpy()})
    orr, oi = out[0], out[1]

    def report(tag, a, b):
        d = np.abs(a - b)
        idx = np.unravel_index(np.argmax(d), d.shape)
        print(f"{tag}: max={d.max():.4e} mean={d.mean():.4e} "
              f"argmax={idx} torch={a[idx]:.4f} onnx={b[idx]:.4f}")

    # 主网络输出（不含 DDPM）对比
    report("holonet real", tr, orr)
    report("holonet imag", ti, oi)
    print("holonet output mag: torch max=%.4f mean=%.4f | onnx max=%.4f mean=%.4f"
          % (np.sqrt(tr**2 + ti**2).max(), np.sqrt(tr**2 + ti**2).mean(),
             np.sqrt(orr**2 + oi**2).max(), np.sqrt(orr**2 + oi**2).mean()))

    # 单独导出 DDPM（输入=主网络输出）对比
    onnx_d = "/tmp/diag_ddpm.onnx"
    if os.path.exists(onnx_d):
        os.remove(onnx_d)

    class WrapDDPM(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
        def forward(self, x):
            y = self.net(x)
            return y.real, y.imag

    w = WrapDDPM(ddpm)
    torch.onnx.export(w, holo, onnx_d, input_names=["input"],
                      output_names=["real_out", "imag_out"],
                      dynamic_axes={"input": {0: "batch"},
                                    "real_out": {0: "batch"},
                                    "imag_out": {0: "batch"}},
                      opset_version=18)
    sess2 = ort.InferenceSession(onnx_d, providers=["CPUExecutionProvider"])
    out2 = sess2.run(None, {sess2.get_inputs()[0].name: holo.numpy()})
    orr2, oi2 = out2[0], out2[1]
    report("ddpm real", dr_t, orr2)
    report("ddpm imag", di_t, oi2)
    amp_t = np.sqrt(dr_t ** 2 + di_t ** 2)
    amp_o = np.sqrt(orr2 ** 2 + oi2 ** 2)
    print("ddpm amp: torch max=%.4f mean=%.4f | onnx max=%.4f mean=%.4f"
          % (amp_t.max(), amp_t.mean(), amp_o.max(), amp_o.mean()))
    print("ddpm amp max diff=%.4e" % np.abs(amp_t - amp_o).max())


if __name__ == "__main__":
    main()
