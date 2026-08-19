# -*- coding: utf-8 -*-
"""Propagation round-trip debug + field-level DPM reconstruction checks."""
import sys, os, torch, numpy as np
sys.path.insert(0, "/root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch")
from src.data.dataset import create_dataloader
from src.optics.propagation import propagator_factory
from src.optics.complex_utils import compl_val, compl_exp
from src.utils.metrics import compute_ssim

BASE = "/root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch"
device = "cuda"
wav = np.array([0.000450, 0.000520, 0.000638])
prop = propagator_factory(input_shape=(384, 384), pitch=0.008, wavelengths=wav,
                          method="as", double_pad=True).to(device)
torch.manual_seed(0)
loader = create_dataloader(os.path.join(BASE, "data/validate_384_v2/validate_04.tfrecord"),
                           {"res_h": 384, "res_w": 384, "sample_count": 100},
                           ["amp_4", "phs_4", "img_0", "depth_0"], active_max_ldi_layer=0,
                           batch_size=1, shuffle=False, num_workers=0, drop_last=False)
batch = next(iter(loader))
amp_gt = batch["amp_4"].to(device); phs_gt = batch["phs_4"].to(device)
holo = compl_val(amp_gt, (phs_gt - 0.5) * 2.0 * np.pi)
dz = 12.0
wl = torch.tensor(wav, device=device).view(1, -1, 1, 1)
with torch.no_grad():
    # identity
    h0 = prop(holo, 0.0)
    print("dz=0  field amp SSIM:", compute_ssim(h0.abs(), holo.abs(), data_range=1.0).item(),
          " max abs diff:", (h0-holo).abs().max().item())
    # round trip no comp
    h_rt = prop(prop(holo, dz), -dz)
    print("rt nocomp field amp SSIM:", compute_ssim(h_rt.abs(), holo.abs(), data_range=1.0).item())
    # round trip with comp
    h_shift = prop(holo, dz) * compl_exp(-2*np.pi*dz/wl).to(torch.complex64)
    h_back = prop(h_shift, -dz) * compl_exp(2*np.pi*dz/wl).to(torch.complex64)
    print("rt comp   field amp SSIM:", compute_ssim(h_back.abs(), holo.abs(), data_range=1.0).item(),
          " max abs diff:", (h_back-holo).abs().max().item())
    # forward-only: does propagation change the amplitude a lot?
    print("|prop(x,12)| vs |x| amp SSIM:", compute_ssim(prop(holo, dz).abs(), holo.abs(), data_range=1.0).item())
    # check real/imag diff scale
    d = (h_back - holo).abs()
    print("rt comp diff: mean %.6f max %.6f  |holo| mean %.6f" % (d.mean().item(), d.max().item(), holo.abs().mean().item()))
    # a single channel slice diff pattern
    ch = 0
    print("diff chan0 mean", d[0,ch].mean().item(), "max", d[0,ch].max().item())
    # phase of the residual
    print("phase(holo) std", holo.angle()[0,ch].std().item(), "phase(h_back) std", h_back.angle()[0,ch].std().item())
    print("phase diff mean", (h_back.angle()-holo.angle())[0,ch].mean().item())