# -*- coding: utf-8 -*-
"""定位 stage2+DDPM ONNX 差异：逐像素分析 + 图结构 dump。"""

import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.holonet import ComplexHoloNet
from src.models.ddpm_net import ComplexDDPMNet
from src.eval.export_onnx import export_onnx


def main():
    ckpt_path, res = sys.argv[1], int(sys.argv[2])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    holonet = ComplexHoloNet(input_dim=4, num_layers=30, num_filters_per_layer=24).eval()
    holonet.load_state_dict(ckpt["holonet_state_dict"])
    ddpm = ComplexDDPMNet(input_dim=3, output_dim=3, num_layers=8, num_filters_per_layer=8).eval()
    ddpm.load_state_dict(ckpt["ddpm_net_state_dict"])

    onnx_path = "/tmp/diag_s2b.onnx"
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
    export_onnx(holonet, ddpm, onnx_path, res_h=res, res_w=res, pad=0,
                input_dim=4, device="cpu")

    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, 4, res, res, generator=g) * 0.25 + 0.5
    with torch.no_grad():
        holo = holonet(x)
        full = ddpm(holo)
        t_r, t_i = full.real.numpy(), full.imag.numpy()

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})
    o_r, o_i = out[0], out[1]

    d = np.sqrt((t_r - o_r) ** 2 + (t_i - o_i) ** 2)
    flat = np.argsort(d.ravel())[-8:][::-1]
    print("top diff pixels (b,c,h,w): diff | torch(real,imag,amp) | onnx(real,imag,amp)")
    for idx in flat:
        b, c, h, w = (int(v) for v in np.unravel_index(int(idx), d.shape))
        tr, ti = float(t_r[b, c, h, w]), float(t_i[b, c, h, w])
        orr, oi = float(o_r[b, c, h, w]), float(o_i[b, c, h, w])
        hr, hi = float(holo[b, c, h, w].real), float(holo[b, c, h, w].imag)
        print("  %s: diff=%.4f torch=(%.4f,%.4f,amp=%.4f) onnx=(%.4f,%.4f,amp=%.4f) holo=(%.4f,%.4f,amp=%.4f)"
              % ((b, c, h, w), float(d[b, c, h, w]), tr, ti, np.hypot(tr, ti), orr, oi,
                 np.hypot(orr, oi), hr, hi, np.hypot(hr, hi)))

    amp_t, amp_o = np.hypot(t_r, t_i), np.hypot(o_r, o_i)
    print("amp: torch max=%.4f mean=%.4f | onnx max=%.4f mean=%.4f | maxdiff=%.4e"
          % (amp_t.max(), amp_t.mean(), amp_o.max(), amp_o.mean(),
             np.abs(amp_t - amp_o).max()))
    print("frac pixels amp diff>0.05: %.6f" % (np.abs(amp_t - amp_o) > 0.05).mean())

    import onnx
    m = onnx.load(onnx_path)
    ops = Counter(n.op_type for n in m.graph.node)
    print("op histogram:", dict(ops))
    print("---- last 60 nodes ----")
    for n in m.graph.node[-60:]:
        print("%-22s in=%s out=%s" % (n.op_type, list(n.input)[:4], list(n.output)[:2]))


if __name__ == "__main__":
    main()
