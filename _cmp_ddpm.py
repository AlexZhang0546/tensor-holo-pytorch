# -*- coding: utf-8 -*-
"""Compare the converted real_ddpm_net against the official TF DDPM on identical input."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.real_ddpm_net import build_ddpm_net
from src.optics.complex_utils import compl_val

d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "official_ddpm_probe.npz"))
amp_in = torch.from_numpy(d["amp_in"])          # (B,3,H,W)
phs_in = torch.from_numpy(d["phs_in"])
amp_out_ref = d["amp_out"]
phs_out_ref = d["phs_out"]

field = compl_val(amp_in, (phs_in - 0.5) * 2.0 * np.pi)
net = build_ddpm_net({"input_dim": 3, "output_dim": 3, "num_layers": 8,
                      "num_filters_per_layer": 8, "interleave_rate": 1,
                      "filter_width": 3, "bias_stddev": 0.01,
                      "weight_var_scale": 0.25}, arch="real", bn_mode="tf")
ck = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "official_ddpm_net.pth"), map_location="cpu")
net.load_state_dict(ck["model_state_dict"])
net.eval()
with torch.no_grad():
    amp_out, phs_out = net.forward_amp_phase(field)
amp_out = amp_out.numpy()
phs_out = phs_out.numpy()
print("torch amp_out mean/std:", amp_out.mean(), amp_out.std())
print("torch phs_out mean/std:", phs_out.mean(), phs_out.std())
print("amp diff:", np.abs(amp_out - amp_out_ref).mean(), np.abs(amp_out - amp_out_ref).max())
print("phs diff:", np.abs(phs_out - phs_out_ref).mean(), np.abs(phs_out - phs_out_ref).max())
