# -*- coding: utf-8 -*-
"""Check whether filter_phs_only reconstruction == baseband of ideal DPM field."""
import sys, os, torch, numpy as np
sys.path.insert(0, "/root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch")
from src.data.dataset import create_dataloader
from src.optics.propagation import propagator_factory
from src.optics.complex_utils import compl_val, compl_exp, fft2d, ifft2d, fftshift2d, ifftshift2d
from src.optics.dpm import aadpm
from src.optics.aperture import filter_phs_only, circ_filter
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
    h_shift = prop(holo, dz) * compl_exp(-2*np.pi*dz/wl).to(torch.complex64)
    phs_only, amax = aadpm(h_shift, propagator=prop, depth_shift=0.0, adaptive_phs_shift=False,
                           batch=1, num_channels=3, res_h=384, res_w=384, sigma=0.0, kernel_width=3,
                           phs_max=None, amp_max=None, clamp=True, normalize=False, wavelength=wav)
    # ideal DPM complex field
    amp = h_shift.abs(); an = amp / amax
    phs_z = h_shift.angle() - h_shift.angle().mean(dim=[2,3], keepdim=True)
    ideal = an * torch.exp(1j*phs_z)   # (B,3,384,384)
    # baseband of ideal (mask r=192, no back-prop)
    mask = circ_filter(1, 3, 384, 384, 192, device=device, dtype=torch.complex64)
    bb_ideal = ifft2d(ifftshift2d(fftshift2d(fft2d(ideal)) * mask))
    # baseband of phase-only pattern
    cpx = compl_val(amax.expand_as(phs_only), phs_only)
    bb_pat = ifft2d(ifftshift2d(fftshift2d(fft2d(cpx)) * mask))
    # what filter_phs_only gives (SLM plane, before back-prop)
    amp_f, phs_f = filter_phs_only(phs_only, unnormalize_input=False, normalize_output=False,
                                   propagator=prop, depth_shift=-dz, batch=1, num_channels=3,
                                   res_h=384, res_w=384, radius=192, phs_max=None,
                                   amp_max=amax, wavelength=wav)
    print("SSIM(bb_pat amp, bb_ideal amp):", compute_ssim(bb_pat.abs(), bb_ideal.abs(), data_range=1.0).item())
    print("SSIM(bb_ideal amp * amax, amp_gt):", compute_ssim((bb_ideal.abs()*amax).squeeze(0), amp_gt, data_range=1.0).item())
    print("SSIM(bb_pat amp, amp_gt):", compute_ssim(bb_pat.abs(), amp_gt, data_range=1.0).item())
    print("SSIM(amp_f, amp_gt):", compute_ssim(amp_f, amp_gt, data_range=1.0).item())
    print("SSIM(amp_f, bb_ideal*amax):", compute_ssim(amp_f, (bb_ideal.abs()*amax).squeeze(0), data_range=1.0).item())
    # low-pass of GT directly
    bb_gt = ifft2d(ifftshift2d(fftshift2d(fft2d(holo)) * mask))
    print("SSIM(lowpass GT amp, amp_gt):", compute_ssim(bb_gt.abs(), amp_gt, data_range=1.0).item())
    print("SSIM(amp_f, lowpass GT amp):", compute_ssim(amp_f, bb_gt.abs(), data_range=1.0).item())
    # also: phase-only baseband without the amp_max factor? bb_ideal is an (normalized). with amax factor:
    print("ranges: amp_gt %.3f-%.3f | bb_ideal*amax %.3f-%.3f | amp_f %.3f-%.3f | bb_pat %.3f-%.3f" % (
        amp_gt.min().item(), amp_gt.max().item(),
        (bb_ideal.abs()*amax).min().item(), (bb_ideal.abs()*amax).max().item(),
        amp_f.min().item(), amp_f.max().item(),
        bb_pat.abs().min().item(), bb_pat.abs().max().item()))