# -*- coding: utf-8 -*-
"""DPM roundtrip ceiling: how good is aadpm+filter on a near-perfect field?
1) encode GT field (properly shifted) -> SSIM
2) optimize free complex field (3000 iters) -> periodically encode -> SSIM of ENCODED version
3) random smooth field roundtrip sanity check
"""
import sys, os, time, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, "/root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch")
from src.data.dataset import create_dataloader
from src.optics.propagation import propagator_factory
from src.optics.complex_utils import compl_val, compl_exp
from src.optics.dpm import aadpm
from src.optics.aperture import filter_phs_only
from src.losses.focal_stack import compute_focal_stack_loss
from src.losses.ddpm_loss import compute_ddpm_phase_loss
from src.utils.metrics import compute_ssim

BASE = "/root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch"
device = "cuda"
wav = np.array([0.000450, 0.000520, 0.000638])
RES = 384
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 3e-3
print("RES", RES, "ITERS", ITERS, "LR", LR, flush=True)

hp = {"wavelengths": wav, "pitch": 0.008, "res_h": RES, "res_w": RES,
      "depth_base": -3, "depth_scale": 6, "double_pad": True}
tp = {"batch": 1, "num_top_depth_for_img_loss": 15, "num_random_depth_for_img_loss": 5,
      "depth_dependent_weight_scale": 0.35, "num_hist_bins": 200,
      "depth_shift": 12.0, "padding": 0, "deterministic_depths": True}
prop = propagator_factory(input_shape=(RES, RES), pitch=0.008, wavelengths=wav,
                          method="as", double_pad=True).to(device)
depth_shift = 12.0
wavelengths_t = torch.tensor(wav, device=device).view(1, -1, 1, 1)

loader = create_dataloader(os.path.join(BASE, "data/validate_384_v2/validate_04.tfrecord"),
                           {"res_h": RES, "res_w": RES, "sample_count": 100},
                           ["amp_4", "phs_4", "img_0", "depth_0"], active_max_ldi_layer=0,
                           batch_size=1, shuffle=False, num_workers=0, drop_last=False)
batch = next(iter(loader))
rgbd = batch["rgbd"].to(device)
amp_gt = batch["amp_4"].to(device)
phs_gt = batch["phs_4"].to(device)
holo_gt = compl_val(amp_gt, (phs_gt - 0.5) * 2.0 * np.pi)

def encode_field(holo):
    holo_shifted = prop(holo, depth_shift) * compl_exp(-2 * np.pi * depth_shift / wavelengths_t)
    phs_only, amp_max = aadpm(holo_shifted, propagator=prop, depth_shift=0.0,
                              adaptive_phs_shift=False, batch=1, num_channels=3,
                              res_h=RES, res_w=RES, sigma=0.0, kernel_width=3,
                              phs_max=None, amp_max=None, clamp=True, normalize=False,
                              wavelength=wav)
    amp_f, phs_f = filter_phs_only(phs_only, unnormalize_input=False, normalize_output=False,
                                   propagator=prop, depth_shift=-depth_shift, batch=1,
                                   num_channels=3, res_h=RES, res_w=RES, radius=None,
                                   phs_max=None, amp_max=amp_max, wavelength=wav)
    return compl_val(amp_f, phs_f)

def report(tag, holo):
    with torch.no_grad():
        _, _, ssim_img, _ = compute_focal_stack_loss(holo, holo_gt, rgbd, prop, hp, tp, F.l1_loss, 0)
        sa = compute_ssim(holo.abs()[:, :, :RES, :RES], amp_gt, data_range=1.0).item()
    print("[%s] SSIM_img %.4f SSIM_amp %.4f" % (tag, ssim_img.item(), sa), flush=True)
    return ssim_img.item()

# 1) GT field properly shifted then encoded
holo_enc_gt = encode_field(holo_gt)
report("GT-encoded", holo_enc_gt)

# 3) DPM roundtrip on a random smooth field
torch.manual_seed(0)
r = torch.randn(1, 3, RES, RES, device=device)
s = torch.randn(1, 3, RES, RES, device=device)
r = F.avg_pool2d(r, 9, stride=1, padding=4)
s = F.avg_pool2d(s, 9, stride=1, padding=4)
rt = compl_val(r.abs() + 0.5, s / s.std() * 1.5)
rt_enc = encode_field(rt)
corr = (rt.conj() * rt_enc).real.mean() / (rt.abs().square().mean().sqrt() * rt_enc.abs().square().mean().sqrt() + 1e-9)
print("[DPM-roundtrip] complex correlation %.4f | rel-amp err %.4f" % (corr.item(),
      ((rt.abs() - rt_enc.abs()).abs().mean() / (rt.abs().mean() + 1e-9)).item()), flush=True)

# 2) free field optimization, periodically encode current field
field = torch.nn.Parameter((holo_gt.clone() + 0.1 * torch.randn_like(holo_gt)).detach())
opt = torch.optim.Adam([field], lr=LR, betas=(0.9, 0.99), eps=1e-8)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERS, eta_min=LR * 0.05)
t0 = time.time()
for it in range(ITERS):
    opt.zero_grad()
    holo = field
    fs, tv, ssim_img, _ = compute_focal_stack_loss(holo, holo_gt, rgbd, prop, hp, tp, F.l1_loss, 0)
    loss = 20.0 * fs + 20.0 * tv + 100.0 * (1.0 - ssim_img)
    loss.backward()
    opt.step()
    sched.step()
    if it % 500 == 0 or it == ITERS - 1:
        sa = compute_ssim(field.detach().abs()[:, :, :RES, :RES], amp_gt, data_range=1.0).item()
        enc = encode_field(field.detach())
        with torch.no_grad():
            _, _, ssim_enc, _ = compute_focal_stack_loss(enc, holo_gt, rgbd, prop, hp, tp, F.l1_loss, 0)
        print("[free] it %5d SSIM_img %.4f SSIM_amp %.4f | ENCODED SSIM_img %.4f (%.0fs)" % (
            it, ssim_img.item(), sa, ssim_enc.item(), time.time() - t0), flush=True)
print("DONE", flush=True)
