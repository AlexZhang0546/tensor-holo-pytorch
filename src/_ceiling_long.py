# -*- coding: utf-8 -*-
"""Long phase-only ceiling: direct phs_only optimization, cosine LR, no sigmoid.
Also reports aadpm(GT)-filtered reference quality.
Args: RES ITERS LR0 IMG_IDX
"""
import sys, os, time, math, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, "/root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch")
from src.models.factory import build_main_net
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
RES = int(sys.argv[1]) if len(sys.argv) > 1 else 384
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 15000
LR0 = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-3
IMG = int(sys.argv[4]) if len(sys.argv) > 4 else 0
WSSIM = float(sys.argv[5]) if len(sys.argv) > 5 else 100.0
print("RES", RES, "ITERS", ITERS, "LR0", LR0, "IMG", IMG, "WSSIM", WSSIM, flush=True)

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
for i, b in enumerate(loader):
    if i == IMG:
        batch = b
        break
rgbd = batch["rgbd"].to(device)
amp_gt = batch["amp_4"].to(device)
phs_gt = batch["phs_4"].to(device)
holo_gt = compl_val(amp_gt, (phs_gt - 0.5) * 2.0 * np.pi)

holonet = build_main_net(arch="unet", input_dim=4, unet_depth=2, unet_base_filters=24,
                         unet_tail_blocks=16, filter_width=3).to(device)
ck = torch.load(os.path.join(BASE, "model/stage1_unet_d2t16/stage1_latest.pth"), map_location="cpu")
holonet.load_state_dict(ck.get("model_state_dict", ck), strict=False)
holonet.eval()
with torch.no_grad():
    holo_mid = holonet(rgbd)
    holo_shifted = prop(holo_mid, depth_shift) * compl_exp(-2 * np.pi * depth_shift / wavelengths_t)
    _, _, ssim_pre, _ = compute_focal_stack_loss(holo_mid, holo_gt, rgbd, prop, hp, tp, F.l1_loss, 0)
    sa_pre = compute_ssim(holo_mid.abs(), amp_gt, data_range=1.0).item()
    print("[A] img %d stage1 pre-encoding: SSIM_img %.4f SSIM_amp %.4f" % (IMG, ssim_pre.item(), sa_pre), flush=True)

    # reference: encode GT field directly
    phs_gt_only, amp_max_gt = aadpm(holo_gt, propagator=prop, depth_shift=0.0,
                                    adaptive_phs_shift=False, batch=1, num_channels=3,
                                    res_h=RES, res_w=RES, sigma=0.0, kernel_width=3,
                                    phs_max=None, amp_max=None, clamp=True, normalize=False,
                                    wavelength=wav)
    amp_f_gt, phs_f_gt = filter_phs_only(phs_gt_only, unnormalize_input=False, normalize_output=False,
                                         propagator=prop, depth_shift=-depth_shift, batch=1,
                                         num_channels=3, res_h=RES, res_w=RES, radius=None,
                                         phs_max=None, amp_max=amp_max_gt, wavelength=wav)
    holo_out_gt = compl_val(amp_f_gt, phs_f_gt)
    _, _, ssim_gt_enc, _ = compute_focal_stack_loss(holo_out_gt, holo_gt, rgbd, prop, hp, tp, F.l1_loss, 0)
    sa_gt_enc = compute_ssim(amp_f_gt[:, :, :RES, :RES], amp_gt, data_range=1.0).item()
    print("[REF] GT directly encoded: SSIM_img %.4f SSIM_amp %.4f" % (ssim_gt_enc.item(), sa_gt_enc), flush=True)

    # init phs_only from stage1 aadpm
    phs_only0, amp_max0 = aadpm(holo_shifted, propagator=prop, depth_shift=0.0,
                                adaptive_phs_shift=False, batch=1, num_channels=3,
                                res_h=RES, res_w=RES, sigma=0.0, kernel_width=3,
                                phs_max=None, amp_max=None, clamp=True, normalize=False,
                                wavelength=wav)

phs_p = torch.nn.Parameter(phs_only0.clone())
opt = torch.optim.Adam([phs_p], lr=LR0, betas=(0.9, 0.99), eps=1e-8)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERS, eta_min=LR0 * 0.05)
t0 = time.time()
best = 0.0
for it in range(ITERS):
    opt.zero_grad()
    phs = phs_p
    amp_f, phs_f = filter_phs_only(phs, unnormalize_input=False, normalize_output=False,
                                   propagator=prop, depth_shift=-depth_shift, batch=1,
                                   num_channels=3, res_h=RES, res_w=RES, radius=None,
                                   phs_max=None, amp_max=None, wavelength=wav)
    holo_out = compl_val(amp_f, phs_f)
    fs, tv, ssim_img, _ = compute_focal_stack_loss(holo_out, holo_gt, rgbd, prop, hp, tp, F.l1_loss, 0)
    std_l, mean_l = compute_ddpm_phase_loss(phs / (2.0 * np.pi) + 0.5, pad=0, res_h=RES, res_w=RES)
    loss = 20.0 * fs + 20.0 * tv + 0.02 * std_l + 0.03 * mean_l
    if WSSIM > 0:
        loss = loss + WSSIM * (1.0 - ssim_img)
    loss.backward()
    opt.step()
    sched.step()
    if it % 500 == 0 or it == ITERS - 1:
        with torch.no_grad():
            sa = compute_ssim(amp_f[:, :, :RES, :RES], amp_gt, data_range=1.0).item()
        if ssim_img.item() > best:
            best = ssim_img.item()
        print("[C-long] it %5d lr %.1e loss %.6f FS %.5f | SSIM_img %.4f SSIM_amp %.4f | best %.4f (%.0fs)" % (
            it, sched.get_last_lr()[0], loss.item(), fs.item(), ssim_img.item(), sa, best, time.time() - t0), flush=True)
print("DONE", flush=True)
