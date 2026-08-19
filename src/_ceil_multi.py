# -*- coding: utf-8 -*-
"""Per-image DPM ceiling via direct field optimization through aadpm+filter.
Args: ITERS LR WSSIM IMGS GLOBAL_AMPMax
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
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 3e-3
WSSIM = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
IMGS = [int(s) for s in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0]
GLOBAL_AM = int(sys.argv[5]) if len(sys.argv) > 5 else 0
print("ITERS", ITERS, "LR", LR, "WSSIM", WSSIM, "IMGS", IMGS, "GLOBAL_AM", GLOBAL_AM, flush=True)

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
samples = [b for i, b in enumerate(loader) if i in IMGS]
for k, batch in enumerate(samples):
    img_idx = IMGS[k]
    rgbd = batch["rgbd"].to(device)
    amp_gt = batch["amp_4"].to(device)
    phs_gt = batch["phs_4"].to(device)
    holo_gt = compl_val(amp_gt, (phs_gt - 0.5) * 2.0 * np.pi)

    amp0 = torch.clamp(amp_gt, max=1.41).detach()
    phs0 = (torch.angle(holo_gt) / (2.0 * np.pi) + 0.5).detach()
    amp_p = torch.nn.Parameter(amp0.clone())
    phs_p = torch.nn.Parameter(phs0.clone())
    opt = torch.optim.Adam([amp_p, phs_p], lr=LR, betas=(0.9, 0.99), eps=1e-8)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERS, eta_min=LR * 0.05)
    t0 = time.time()
    best = (0.0, 0.0)
    for it in range(ITERS):
        opt.zero_grad()
        amp = torch.sigmoid(amp_p) * 1.4142
        phs = torch.sigmoid(phs_p)
        holo_altered = compl_val(amp, (phs - 0.5) * 2.0 * np.pi)
        if GLOBAL_AM:
            amax = holo_altered.abs().max().detach()
        else:
            amax = None
        phs_only, amp_max = aadpm(holo_altered, propagator=prop, depth_shift=0.0,
                                  adaptive_phs_shift=False, batch=1, num_channels=3,
                                  res_h=RES, res_w=RES, sigma=0.0, kernel_width=3,
                                  phs_max=None, amp_max=amax, clamp=True, normalize=False,
                                  wavelength=wav)
        amp_f, phs_f = filter_phs_only(phs_only, unnormalize_input=False, normalize_output=False,
                                       propagator=prop, depth_shift=-depth_shift, batch=1,
                                       num_channels=3, res_h=RES, res_w=RES, radius=None,
                                       phs_max=None, amp_max=amp_max, wavelength=wav)
        holo_out = compl_val(amp_f, phs_f)
        fs, tv, ssim_img, _ = compute_focal_stack_loss(holo_out, holo_gt, rgbd, prop, hp, tp, F.l1_loss, 0)
        std_l, mean_l = compute_ddpm_phase_loss(phs, pad=0, res_h=RES, res_w=RES)
        loss = 20.0 * fs + 20.0 * tv + 0.02 * std_l + 0.03 * mean_l
        if WSSIM > 0:
            loss = loss + WSSIM * (1.0 - ssim_img)
        loss.backward()
        opt.step()
        sched.step()
        if it % 500 == 0 or it == ITERS - 1:
            sa = compute_ssim(amp_f[:, :, :RES, :RES], amp_gt, data_range=1.0).item()
            best = max(best, (ssim_img.item(), sa))
            print("[img%d] it %5d SSIM_img %.4f SSIM_amp %.4f (%.0fs)" % (
                img_idx, it, ssim_img.item(), sa, time.time() - t0), flush=True)
    print("[img%d] BEST SSIM_img %.4f SSIM_amp %.4f" % (img_idx, best[0], best[1]), flush=True)
print("DONE", flush=True)
