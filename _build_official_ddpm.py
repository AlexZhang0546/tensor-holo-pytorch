# -*- coding: utf-8 -*-
"""Build real_ddpm_net state dict from the official TF weights."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models.real_ddpm_net import build_ddpm_net

d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "official_ddpm.npz"))
net = build_ddpm_net({"input_dim": 3, "output_dim": 3, "num_layers": 8,
                      "num_filters_per_layer": 8, "interleave_rate": 1,
                      "filter_width": 3, "bias_stddev": 0.01,
                      "weight_var_scale": 0.25}, arch="real", bn_mode="tf")
sd = net.state_dict()
for i in range(8):
    sd["convs.%d.weight" % i] = torch.from_numpy(d["conv%d_w" % i])
    sd["convs.%d.bias" % i] = torch.from_numpy(d["conv%d_b" % i])
    sd["bns.%d.gamma" % i] = torch.from_numpy(d["bn%d_gamma" % i])
    sd["bns.%d.beta" % i] = torch.from_numpy(d["bn%d_beta" % i])
torch.save({"model_state_dict": sd}, "official_ddpm_net.pth")
print("converted official_ddpm_net.pth")
