# -*- coding: utf-8 -*-
"""Long single-image network joint training (UNet + RealDDPM), paper recipe + SSIM loss.
Args: IMG RADIUS NF IDENT JOINT LR WSSIM
"""
import sys, os, time, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, "/root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch")
from src.models.factory import build_main_net
from src.models.real_ddpm_net import RealAmpPhaseDDPMNet
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
IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 0
RADIUS = int(sys.argv[2]) if len(sys.argv) > 2 else 192
NF = int(sys.argv[3]) if len(sys.argv) > 3 else 32
IDENT = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
JOINT = int(sys.argv[5]) if len(sys.argv) > 5 else 20000
LR = float(sys.argv[6]) if len(sys.argv) > 6 else 3e-4
WSSIM = float(sys.argv[7]) if len(sys.argv) > 7 else 50.0
print("IMG", IMG, "RADIUS", RADIUS, "NF", NF, "IDENT", IDENT, "JOINT", JOINT, "LR", LR, "WSSIM", WSSIM, flush=True)

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

ddpm = RealAmpPhaseDDPMNet(input_dim=3, output_dim=3, num_layers=8,
                           num_filters_per_layer=NF, interleave_rate=1,
                           filter_width=3, bias_stddev=0.01,
                           weight_var_scale=0.25, bn_mode='tf').to(device)
print("ddpm params:", sum(p.numel() for p in ddpm.parameters()), flush=True)

def pipeline(holo_mid):
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
                                   num_channels=3, res_h=RES, res_w=RES, radius=RADIUS,
                                   phs_max=None, amp_max=amp_max, wavelength=wav)
    holo_out = compl_val(amp_f, phs_f)
    return holo_out, phs_for_reg, amp_f

def joint_loss(holo_mid):
    holo_out, phs_for_reg, amp_f = pipeline(holo_mid)
    fs, tv, ssim_img, _ = compute_focal_stack_loss(holo_out, holo_gt, rgbd, prop, hp, tp, F.l1_loss, 0)
    std_l, mean_l = compute_ddpm_phase_loss(phs_for_reg, pad=0, res_h=RES, res_w=RES)
    loss = 20.0 * fs + 20.0 * tv + 0.02 * std_l + 0.03 * mean_l
    if WSSIM > 0:
        loss = loss + WSSIM * (1.0 - ssim_img)
    return loss, fs, ssim_img, std_l, mean_l, amp_f

# identity
holonet.eval()
ddpm.train()
opt_id = torch.optim.Adam(ddpm.parameters(), lr=LR, betas=(0.9, 0.99), eps=1e-8)
with torch.no_grad():
    holo_mid0 = holonet(rgbd)
    holo_shifted0 = prop(holo_mid0, depth_shift) * compl_exp(-2 * np.pi * depth_shift / wavelengths_t)
amp_t = holo_shifted0.abs().detach()
phs_t = (torch.angle(holo_shifted0) / (2.0 * np.pi) + 0.5).detach()
t0 = time.time()
for it in range(IDENT):
    opt_id.zero_grad()
    amp_a, phs_a = ddpm.forward_amp_phase(holo_shifted0)
    loss = F.l1_loss(amp_a, amp_t) + F.l1_loss(phs_a, phs_t)
    loss.backward()
    opt_id.step()
    if it % 500 == 0 or it == IDENT - 1:
        with torch.no_grad():
            amp_a, phs_a = ddpm.forward_amp_phase(holo_shifted0)
            sa = compute_ssim(amp_a, amp_t, data_range=1.0).item()
        print("[identity] it %5d loss %.6f SSIM_amp %.4f (%.0fs)" % (it, loss.item(), sa, time.time() - t0), flush=True)
torch.save({"ddpm_net_state_dict": ddpm.state_dict()}, os.path.join(BASE, "model/netlong_id_img%d_r%d.pth" % (IMG, RADIUS)))

# joint
ddpm.train()
holonet.train()
params = list(ddpm.parameters()) + list(holonet.parameters())
opt_j = torch.optim.Adam(params, lr=LR, betas=(0.9, 0.99), eps=1e-8)
t0 = time.time()
best = (0.0, 0.0)
for it in range(JOINT):
    opt_j.zero_grad()
    holo_mid = holonet(rgbd)
    loss, fs, ssim_img, std_l, mean_l, amp_f = joint_loss(holo_mid)
    loss.backward()
    opt_j.step()
    if it % 500 == 0 or it == JOINT - 1:
        with torch.no_grad():
            _, _, ssim_e, std_e, mean_e, amp_fe = joint_loss(holo_mid)
            sa = compute_ssim(amp_fe[:, :, :RES, :RES], amp_gt, data_range=1.0).item()
        best = max(best, (ssim_img.item(), sa))
        print("[joint] it %5d loss %.6f FS %.5f mean %.5f std %.5f | SSIM_img %.4f SSIM_amp %.4f (%.0fs)" % (
            it, loss.item(), fs.item(), mean_l.item(), std_l.item(), ssim_img.item(), sa, time.time() - t0), flush=True)
torch.save({"ddpm_net_state_dict": ddpm.state_dict(),
            "holonet_state_dict": holonet.state_dict()},
           os.path.join(BASE, "model/netlong_joint_img%d_r%d.pth" % (IMG, RADIUS)))
print("BEST SSIM_img %.4f SSIM_amp %.4f" % best, flush=True)
print("DONE", flush=True)
