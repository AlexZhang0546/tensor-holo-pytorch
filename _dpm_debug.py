# -*- coding: utf-8 -*-
"""DPM debug: find where the encode/decode loses fidelity.
1) sub-pixel math check: (exp(j low)+exp(j high))/2 vs amp/amp_max*exp(j phs')
2) full aadpm -> filter_phs_only round trip at field level (amp SSIM), several radii
3) reconstruction before back-propagation (SLM-plane amplitude vs shifted GT amp)
4) no-circ-filter variant (radius = full) to isolate the aperture effect
"""
import sys, os, torch, numpy as np
sys.path.insert(0, "/root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch")
from src.data.dataset import create_dataloader
from src.optics.propagation import propagator_factory
from src.optics.complex_utils import compl_val, compl_exp, fft2d, ifft2d, fftshift2d, ifftshift2d
from src.optics.dpm import aadpm, _depth_to_space_nchw
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
h_shift = prop(holo, dz) * compl_exp(-2*np.pi*dz/wl).to(torch.complex64)
print("shapes: holo", tuple(holo.shape), "h_shift", tuple(h_shift.shape))
print("amp_gt range", amp_gt.min().item(), amp_gt.max().item())

with torch.no_grad():
    # ---- 1) sub-pixel math ----
    amp = h_shift.abs(); amp_max = amp.amax(dim=(2,3), keepdim=True) + 1e-6
    an = amp / amp_max
    phs = h_shift.angle()
    phs_z = phs - phs.mean(dim=[2,3], keepdim=True)
    off = torch.acos(torch.clamp(an, -1+1e-7, 1-1e-7))
    low = phs_z - off; high = phs_z + off
    recon = 0.5*(torch.exp(1j*low) + torch.exp(1j*high))
    target = an * torch.exp(1j*phs_z)
    err = (recon - target).abs().mean().item()
    print("subpixel-pair recon abs err vs target:", err, " (should be ~0)")
    # field-level amp ssim of pair reconstruction vs original shifted field amp
    print("pair-recon amp SSIM vs shifted amp:", compute_ssim(recon.abs(), amp, data_range=1.0).item())

    # ---- 2) full aadpm -> filter round trip with different radii ----
    phs_only, amax = aadpm(h_shift, propagator=prop, depth_shift=0.0, adaptive_phs_shift=False,
                           batch=1, num_channels=3, res_h=384, res_w=384, sigma=0.0, kernel_width=3,
                           phs_max=None, amp_max=None, clamp=True, normalize=False, wavelength=wav)
    print("phs_only shape", tuple(phs_only.shape), "amp_max shape", tuple(amax.shape))
    for radius in [None, 96, 128, 160, 192, 384//2, 300]:
        amp_f, phs_f = filter_phs_only(phs_only, unnormalize_input=False, normalize_output=False,
                                       propagator=prop, depth_shift=-dz, batch=1, num_channels=3,
                                       res_h=384, res_w=384, radius=radius, phs_max=None,
                                       amp_max=amax, wavelength=wav)
        ssim_amp_gt = compute_ssim(amp_f, amp_gt, data_range=1.0).item()
        # also compare reconstructed SLM-plane field (before back prop) with shifted holo amp
        # reconstruct without back-propagation: radius only
        cpx = compl_val(amax.expand_as(phs_only), phs_only)
        cf = fftshift2d(fft2d(cpx))
        mask = circ_filter(1, 3, 384, 384, 192 if radius is None else radius, device=device, dtype=torch.complex64)
        cpx_f = ifft2d(ifftshift2d(cf * mask))
        ssim_slm = compute_ssim(cpx_f.abs(), h_shift.abs(), data_range=1.0).item()
        print("radius %-4s  amp_f SSIM_vs_GT %.4f  slm-plane amp SSIM_vs_shift %.4f" % (
            str(radius), ssim_amp_gt, ssim_slm))

    # ---- 3) no filter at all: pure phase-only replay (no aperture), same res ----
    cpx = compl_val(amax.expand_as(phs_only), phs_only)
    cf = fftshift2d(fft2d(cpx))
    mask = torch.ones_like(cf)
    cpx_f = ifft2d(ifftshift2d(cf * mask))
    print("no-filter amp SSIM vs shifted:", compute_ssim(cpx_f.abs(), h_shift.abs(), data_range=1.0).item())
    # what if we use phase pattern directly (no FFT at all)?
    print("direct phase-only amp SSIM vs shifted:", compute_ssim(cpx.abs(), h_shift.abs(), data_range=1.0).item())

    # ---- 4) check amplitude of reconstruction scale ----
    amp_f, _ = filter_phs_only(phs_only, unnormalize_input=False, normalize_output=False,
                               propagator=prop, depth_shift=-dz, batch=1, num_channels=3,
                               res_h=384, res_w=384, radius=None, phs_max=None,
                               amp_max=amax, wavelength=wav)
    print("amp_f range", amp_f.min().item(), amp_f.max().item(), " amp_gt range", amp_gt.min().item(), amp_gt.max().item())
    print("amp_f mean", amp_f.mean().item(), " amp_gt mean", amp_gt.mean().item())
    print("amp_f / amp_max ratio", (amp_f / (amax+1e-9)).mean().item())