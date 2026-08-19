# -*- coding: utf-8 -*-
import sys, torch
sys.path.insert(0, "/root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch")
from src.models.real_ddpm_net import RealAmpPhaseDDPMNet, build_ddpm_net
net = RealAmpPhaseDDPMNet(input_dim=3, output_dim=3, num_layers=8, num_filters_per_layer=8, bn_mode='tf').cuda()
x = torch.randn(1, 3, 384, 384, dtype=torch.complex64, device='cuda')
z = net(x)
a, p = net.forward_amp_phase(x)
print('fwd ok', z.shape, float(a.min()), float(a.max()), float(p.min()), float(p.max()))
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
