# -*- coding: utf-8 -*-
"""Single-sample stage2 quick verification for RealAmpPhaseDDPMNet.

Phases:
  1. identity: fit DDPM to map holo_shifted -> holo_shifted (amp+phs L1, paper loss)
  2. joint:    fit DDPM + holonet with the full stage2 loss (fs+tv+std+mean)

Args: IDENT_ITERS JOINT_ITERS LR
Also prints reference ceilings: bypass-DDPM (identity) and GT-input DPM.
"""
import sys, os, time, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, "/root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch")
from src.models.factory import build_main_net
from src.models.real_ddpm_net import build_ddpm_net
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

holonet = build_main_net(arch="unet", input_dim=4, unet_depth=2, unet_base_filters=24,
                         unet_tail_blocks=16, filter_width=3).to(device)
ck = torch.load(os.path.join(BASE, "model/stage1_unet_d2t16/stage1_latest.pth"), map_location="cpu")
r = holonet.load_state_dict(ck.get("model_state_dict", ck), strict=False)
print("holonet load warn missing:%d unexpected:%d" % (len(r.missing_keys), len(r.unexpected_keys)), flush=True)

ddpm = build_ddpm_net({"input_dim": 3, "output_dim": 3, "num_layers": 8,
                       "num_filters_per_layer": 8, "interleave_rate": 1,
                       "filter_width": 3, "bias_stddev": 0.01,
                       "weight_var_scale": 0.25}, arch="real", bn_mode="tf").to(device)
print("ddpm params:", sum(p.numel() for p in ddpm.parameters()), flush=True)

ident_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 300
joint_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 300
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4

holonet.eval()
with torch.no_grad():
    holo_mid = holonet(rgbd)
    holo_shifted = prop(holo_mid, depth_shift) * compl_exp(-2 * np.pi * depth_shift / wavelengths_t)


def run_dpm(field, shift_in):
    phs_only, amp_max = aadpm(field, propagator=prop, depth_shift=0.0,
                              adaptive_phs_shift=False, batch=1, num_channels=3,
                              res_h=RES, res_w=RES, sigma=0.0, kernel_width=3,
                              phs_max=None, amp_max=None, clamp=True, normalize=False,
                              wavelength=wav)
    amp_f, phs_f = filter_phs_only(phs_only, unnormalize_input=False, normalize_output=False,
                                   propagator=prop, depth_shift=-shift_in, batch=1,
                                   num_channels=3, res_h=RES, res_w=RES, radius=None,
                                   phs_max=None, amp_max=amp_max, wavelength=wav)
    holo_out = compl_val(amp_f, phs_f)
    _, _, ssim_img, _ = compute_focal_stack_loss(holo_out, holo_gt, rgbd, prop,
                                                 hp, tp, F.l1_loss, 0)
    amp_c = amp_f[:, :, :RES, :RES]
    ssim_amp = compute_ssim(amp_c, amp_gt, data_range=1.0)
    return ssim_img.item(), ssim_amp.item()


with torch.no_grad():
    # ceiling 1: GT field through shift + DPM (paper pipeline upper bound)
    holo_gt_shift = prop(holo_gt, depth_shift) * compl_exp(-2 * np.pi * depth_shift / wavelengths_t)
    si_gt, sa_gt = run_dpm(holo_gt_shift, depth_shift)
    print("[ref] GT shifted -> DPM        : SSIM_img %.4f SSIM_amp %.4f" % (si_gt, sa_gt), flush=True)
    # ceiling 2: no DDPM (identity bypass)
    si_id, sa_id = run_dpm(holo_shifted, depth_shift)
    print("[ref] UNet shifted -> DPM (bypass): SSIM_img %.4f SSIM_amp %.4f" % (si_id, sa_id), flush=True)
    # ceiling 3: random init ddpm
    amp_a, phs_a = ddpm.forward_amp_phase(holo_shifted)
    holo_a = compl_val(amp_a, (phs_a - 0.5) * 2.0 * np.pi)
    si_d0, sa_d0 = run_dpm(holo_a, depth_shift)
    print("[ref] UNet shifted -> rand DDPM : SSIM_img %.4f SSIM_amp %.4f" % (si_d0, sa_d0), flush=True)

# ---------- identity ----------
ddpm.train()
opt_id = torch.optim.Adam(ddpm.parameters(), lr=LR, betas=(0.9, 0.99), eps=1e-8)
t0 = time.time()
for it in range(ident_iters):
    opt_id.zero_grad()
    amp_a, phs_a = ddpm.forward_amp_phase(holo_shifted)
    amp_t = holo_shifted.abs()
    phs_t = holo_shifted.angle() / (2.0 * np.pi) + 0.5
    loss = F.l1_loss(amp_a, amp_t) + F.l1_loss(phs_a, phs_t)
    loss.backward()
    opt_id.step()
    if it % 50 == 0 or it == ident_iters - 1:
        with torch.no_grad():
            amp_a, phs_a = ddpm.forward_amp_phase(holo_shifted)
            sa = compute_ssim(amp_a, amp_t, data_range=1.0).item()
        print("[identity] it %4d loss %.6f SSIM_amp %.4f (%.0fs)" % (it, loss.item(), sa, time.time() - t0), flush=True)

torch.save({"ddpm_net_state_dict": ddpm.state_dict()}, os.path.join(BASE, "model/ddpm_overfit_identity.pth"))
print("saved model/ddpm_overfit_identity.pth", flush=True)

with torch.no_grad():
    amp_a, phs_a = ddpm.forward_amp_phase(holo_shifted)
    holo_a = compl_val(amp_a, (phs_a - 0.5) * 2.0 * np.pi)
    si_di, sa_di = run_dpm(holo_a, depth_shift)
    print("[ref] after identity -> DPM    : SSIM_img %.4f SSIM_amp %.4f" % (si_di, sa_di), flush=True)

# ---------- joint ----------
if joint_iters > 0:
    ddpm.train()
    holonet.train()
    opt_j = torch.optim.Adam(list(ddpm.parameters()) + list(holonet.parameters()),
                             lr=LR, betas=(0.9, 0.99), eps=1e-8)
    t0 = time.time()
    for it in range(joint_iters):
        opt_j.zero_grad()
        with torch.no_grad():
            holo_mid = holonet(rgbd)
        holo_shifted = prop(holo_mid, depth_shift) * compl_exp(-2 * np.pi * depth_shift / wavelengths_t)
        holo_altered = ddpm(holo_shifted)
        phs_for_reg = torch.angle(holo_altered) / (2.0 * np.pi) + 0.5
        phs_only, amp_max = aadpm(holo_altered, propagator=prop, depth_shift=0.0,
                                  adaptive_phs_shift=False, batch=1, num_channels=3,
                                  res_h=RES, res_w=RES, sigma=0.0, kernel_width=3,
                                  phs_max=None, amp_max=None, clamp=True, normalize=False,
                                  wavelength=wav)
        amp_f, phs_f = filter_phs_only(phs_only, unnormalize_input=False, normalize_output=False,
                                       propagator=prop, depth_shift=-depth_shift, batch=1,
                                       num_channels=3, res_h=RES, res_w=RES, radius=None,
                                       phs_max=None, amp_max=amp_max, wavelength=wav)
        holo_out = compl_val(amp_f, phs_f)
        fs, tv, ssim_img, _ = compute_focal_stack_loss(holo_out, holo_gt, rgbd, prop,
                                                       hp, tp, F.l1_loss, 0)
        std_l, mean_l = compute_ddpm_phase_loss(phs_for_reg, pad=0, res_h=RES, res_w=RES)
        loss = 20.0 * fs + 20.0 * tv + 0.02 * std_l + 0.03 * mean_l
        loss.backward()
        opt_j.step()
        if it % 25 == 0 or it == joint_iters - 1:
            with torch.no_grad():
                holo_altered_e = ddpm(holo_shifted)
                phs_e = torch.angle(holo_altered_e) / (2.0 * np.pi) + 0.5
                st, mn = compute_ddpm_phase_loss(phs_e, pad=0, res_h=RES, res_w=RES)
                amp_c = amp_f[:, :, :RES, :RES]
                sa = compute_ssim(amp_c, amp_gt, data_range=1.0).item()
            print("[joint] it %4d loss %.6f FS %.5f TV %.5f mean %.5f std %.5f | SSIM_img %.4f SSIM_amp %.4f (%.0fs)" % (
                it, loss.item(), fs.item(), tv.item(), mn.item(), st.item(), ssim_img.item(), sa, time.time() - t0), flush=True)
    torch.save({"ddpm_net_state_dict": ddpm.state_dict(),
                "holonet_state_dict": holonet.state_dict()},
               os.path.join(BASE, "model/ddpm_overfit_joint.pth"))
    print("saved model/ddpm_overfit_joint.pth", flush=True)
