# -*- coding: utf-8 -*-
"""GT upper-bound diagnostic for the stage2 (DDPM/DPM) pipeline.

Variants measured on validate_384 (100 samples):
  sanity   : focal-stack SSIM of GT field vs itself            (expect ~1.0)
  roundtrip: shift +12mm (w/ phase comp) then -12mm, no DPM    (expect ~1.0)
  dpm_gt   : GT field -> shift -> AA-DPM encode -> reconstruct -> focal SSIM
             (perfect-holo, identity-DDPM ceiling)
  amp_dpm  : reconstructed amp vs GT amp (post-encoding SSIM_amp)
  model    : stage1 UNet(d2t16) holo -> shift -> DPM -> reconstruct (identity DDPM)
Also prints whether the 'model' pipeline reproduces the ~0.65 joint-start SSIM.
"""
import sys, os, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, "/root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch")
from src.models.factory import build_main_net
from src.data.dataset import create_dataloader
from src.optics.propagation import propagator_factory
from src.optics.complex_utils import compl_val, compl_exp
from src.optics.dpm import aadpm
from src.optics.aperture import filter_phs_only
from src.losses.focal_stack import compute_focal_stack_loss
from src.utils.metrics import compute_ssim, compute_psnr

BASE = "/root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch"
device = "cuda"
wav = np.array([0.000450, 0.000520, 0.000638])
hp = {"wavelengths": wav, "pitch": 0.008, "res_h": 384, "res_w": 384,
      "depth_base": -3, "depth_scale": 6, "double_pad": True}
tp = {"batch": 1, "num_top_depth_for_img_loss": 15, "num_random_depth_for_img_loss": 5,
      "depth_dependent_weight_scale": 0.35, "num_hist_bins": 200, "depth_shift": 12.0,
      "padding": 0, "deterministic_depths": False}
prop = propagator_factory(input_shape=(384, 384), pitch=0.008, wavelengths=wav,
                          method="as", double_pad=True).to(device)
torch.manual_seed(0)
loader = create_dataloader(os.path.join(BASE, "data/validate_384_v2/validate_04.tfrecord"),
                           {"res_h": 384, "res_w": 384, "sample_count": 100},
                           ["amp_4", "phs_4", "img_0", "depth_0"], active_max_ldi_layer=0,
                           batch_size=1, shuffle=False, num_workers=0, drop_last=False)

net = None
CKPT = "model/stage1_unet_d2t16/stage1_latest.pth"
if os.path.exists(os.path.join(BASE, CKPT)):
    c = torch.load(os.path.join(BASE, CKPT), map_location="cpu")
    net = build_main_net(arch="unet", input_dim=4, num_layers=30, num_filters_per_layer=24,
                         unet_depth=2, unet_base_filters=24, unet_attention=False,
                         unet_out_bn=False, unet_stem_skip=False,
                         unet_refine_blocks=0, unet_global_in=False,
                         unet_tail_blocks=16).to(device).eval()
    r = net.load_state_dict(c["model_state_dict"], strict=False)
    if r.missing_keys or r.unexpected_keys:
        print("load warn  missing:", r.missing_keys[:6], " unexpected:", r.unexpected_keys[:6])
else:
    print("no stage1 ckpt at", CKPT)

def fs_ssim(holo_out, holo_gt, x):
    fs, tv, ssim_i, psnr_i = compute_focal_stack_loss(holo_out, holo_gt, x, prop, hp, tp, F.l1_loss, 0)
    return ssim_i.item(), psnr_i.item()

S = {"sanity": [], "roundtrip": [], "dpm_gt": [], "amp_dpm_gt": [], "model": [], "amp_model": []}
dz = 12.0
wl = torch.tensor(wav, device=device).view(1, -1, 1, 1)
with torch.no_grad():
    for k, batch in enumerate(loader):
        x = batch["rgbd"].to(device); amp_gt = batch["amp_4"].to(device); phs_gt = batch["phs_4"].to(device)
        holo_gt = compl_val(amp_gt, (phs_gt - 0.5) * 2.0 * np.pi)
        # sanity
        s, p = fs_ssim(holo_gt, holo_gt, x); S["sanity"].append((s, p))
        # roundtrip shift+backshift
        h_shift = prop(holo_gt, dz) * compl_exp(-2*np.pi*dz/wl).to(torch.complex64)
        h_back = prop(h_shift, -dz) * compl_exp(2*np.pi*dz/wl).to(torch.complex64)
        s, p = fs_ssim(h_back, holo_gt, x); S["roundtrip"].append((s, p))
        # DPM encode/reconstruct of GT
        phs_only, amp_max = aadpm(h_shift, propagator=prop, depth_shift=0.0,
                                  adaptive_phs_shift=False, batch=1, num_channels=3,
                                  res_h=384, res_w=384, sigma=0.0, kernel_width=3,
                                  phs_max=None, amp_max=None, clamp=True, normalize=False,
                                  wavelength=wav)
        amp_f, phs_f = filter_phs_only(phs_only, unnormalize_input=False, normalize_output=False,
                                       propagator=prop, depth_shift=-dz, batch=1, num_channels=3,
                                       res_h=384, res_w=384, radius=None, phs_max=None,
                                       amp_max=amp_max, wavelength=wav)
        holo_dpm = compl_val(amp_f, phs_f)
        s, p = fs_ssim(holo_dpm, holo_gt, x); S["dpm_gt"].append((s, p))
        S["amp_dpm_gt"].append(compute_ssim(amp_f, amp_gt, data_range=1.0).item())
        # model pipeline (identity DDPM)
        if net is not None:
            holo_mid = net(x)
            h_shift_m = prop(holo_mid, dz) * compl_exp(-2*np.pi*dz/wl).to(torch.complex64)
            phs_only_m, amp_max_m = aadpm(h_shift_m, propagator=prop, depth_shift=0.0,
                                          adaptive_phs_shift=False, batch=1, num_channels=3,
                                          res_h=384, res_w=384, sigma=0.0, kernel_width=3,
                                          phs_max=None, amp_max=None, clamp=True, normalize=False,
                                          wavelength=wav)
            amp_fm, phs_fm = filter_phs_only(phs_only_m, unnormalize_input=False, normalize_output=False,
                                             propagator=prop, depth_shift=-dz, batch=1, num_channels=3,
                                             res_h=384, res_w=384, radius=None, phs_max=None,
                                             amp_max=amp_max_m, wavelength=wav)
            holo_dpm_m = compl_val(amp_fm, phs_fm)
            s, p = fs_ssim(holo_dpm_m, holo_gt, x); S["model"].append((s, p))
            S["amp_model"].append(compute_ssim(amp_fm, amp_gt, data_range=1.0).item())
        if k % 10 == 0:
            print("sample %3d  sanity %.4f  rt %.4f  dpm_gt %.4f  amp_gt %.4f  model %.4f  amp_model %.4f" % (
                k, S["sanity"][-1][0], S["roundtrip"][-1][0], S["dpm_gt"][-1][0],
                S["amp_dpm_gt"][-1], S["model"][-1][0] if S["model"] else -1,
                S["amp_model"][-1] if S["amp_model"] else -1), flush=True)
        if k >= 49:
            break

print("\n=== SUMMARY (first 50 val samples) ===")
for key, arr in S.items():
    if arr:
        a = np.array([v[0] if isinstance(v, tuple) else v for v in arr])
        print("%-12s mean %.4f  min %.4f  max %.4f" % (key, a.mean(), a.min(), a.max()))