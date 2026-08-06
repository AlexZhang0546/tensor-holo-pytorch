import sys
import torch
import numpy as np
import torch.nn.functional as F

sys.path.insert(0, ".")
torch.manual_seed(0)
dev = torch.device("cuda")

from src.data.dataset import THDataset
from src.models.holonet import ComplexHoloNet
from src.models.ddpm_net import ComplexDDPMNet
from src.train.stage2 import build_propagator_padded
from src.optics.dpm import aadpm
from src.optics.aperture import filter_phs_only
from src.optics.complex_utils import compl_exp
from src.utils.metrics import compute_ssim

s1 = "model/ckpt_full_loss_pitch_8_layers_30_filters_24_stage1/stage1_latest.pth"
idd = "model/stage2_test_pad0_shift12/stage2_identity_epoch_0002.pth"

holo = ComplexHoloNet(input_dim=4, num_layers=30, num_filters_per_layer=24).to(dev).eval()
ck = torch.load(s1, map_location="cpu")
holo.load_state_dict(ck["model_state_dict"])
print("holonet loaded from", s1, "epoch", ck.get("epoch"))

ddpm = ComplexDDPMNet(input_dim=3, output_dim=3, num_layers=8,
                      num_filters_per_layer=8).to(dev).eval()
ck2 = torch.load(idd, map_location="cpu")
ddpm.load_state_dict(ck2["ddpm_net_state_dict"])
print("ddpm loaded from", idd, "epoch", ck2.get("epoch"))

ds = THDataset(
    "data/test_384_v2/test_04.tfrecord",
    {"res_h": 384, "res_w": 384, "sample_count": 100},
    ["amp_4", "phs_4", "img_0", "depth_0"],
    active_max_ldi_layer=0, load_to_memory=True)
batch = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
b = next(iter(batch))
rgbd = b["rgbd"].to(dev)
amp_gt = b["amp_4"].to(dev)
phs_gt = b["phs_4"].to(dev)

pad = 0
depth_shift = 12.0
wav = np.array([0.000450, 0.000520, 0.000638])
hologram_params = {"wavelengths": wav, "pitch": 0.008, "res_h": 384, "res_w": 384}
prop = build_propagator_padded(hologram_params, pad).to(dev)
wt = torch.tensor(wav, device=dev, dtype=torch.float32).view(1, -1, 1, 1)

with torch.no_grad():
    holo_mid = holo(rgbd)
    holo_shifted = prop(holo_mid, depth_shift) * compl_exp(
        -2 * np.pi * depth_shift / wt)
    holo_altered = ddpm(holo_shifted)
    phs_only, amp_max = aadpm(
        holo_altered, propagator=prop, depth_shift=0.0,
        adaptive_phs_shift=False, batch=2, num_channels=3,
        res_h=384, res_w=384, sigma=0.0, kernel_width=3,
        phs_max=None, amp_max=None, clamp=True, normalize=False,
        wavelength=wav.tolist())
    amp_final, phs_final = filter_phs_only(
        phs_only, unnormalize_input=False, normalize_output=False,
        propagator=prop, depth_shift=-depth_shift, batch=2, num_channels=3,
        res_h=384, res_w=384, radius=None, phs_max=None, amp_max=amp_max,
        wavelength=wav.tolist())

for name, t in [("amp_gt", amp_gt), ("holo_mid amp", holo_mid.abs()),
                ("holo_shifted amp", holo_shifted.abs()),
                ("holo_altered amp", holo_altered.abs()),
                ("amp_final", amp_final)]:
    print("%-18s mean %.4f std %.4f min %.4f max %.4f" % (
        name, t.mean().item(), t.std().item(), t.min().item(), t.max().item()))

for c in range(3):
    a = amp_final[0, c].flatten().float()
    g = amp_gt[0, c].flatten().float()
    corr = torch.corrcoef(torch.stack([a, g]))[0, 1].item()
    print("ch%d corr(amp_final, amp_gt) = %.4f" % (c, corr))

ssim = compute_ssim(amp_final, amp_gt, data_range=1.0)
print("SSIM(amp_final, amp_gt) = %.4f" % ssim.item())
print("phs_only range: %.3f .. %.3f" % (phs_only.min().item(),
                                         phs_only.max().item()))
print("amp_max shape:", tuple(amp_max.shape), "vals:",
      [round(v, 4) for v in amp_max.flatten().tolist()])

# 对比：不经过 DDPM（bypass）时的 amp_final
with torch.no_grad():
    phs_only2, amp_max2 = aadpm(
        holo_shifted, propagator=prop, depth_shift=0.0,
        adaptive_phs_shift=False, batch=2, num_channels=3,
        res_h=384, res_w=384, sigma=0.0, kernel_width=3,
        phs_max=None, amp_max=None, clamp=True, normalize=False,
        wavelength=wav.tolist())
    amp_final2, _ = filter_phs_only(
        phs_only2, unnormalize_input=False, normalize_output=False,
        propagator=prop, depth_shift=-depth_shift, batch=2, num_channels=3,
        res_h=384, res_w=384, radius=None, phs_max=None, amp_max=amp_max2,
        wavelength=wav.tolist())
for c in range(3):
    a = amp_final2[0, c].flatten().float()
    g = amp_gt[0, c].flatten().float()
    corr = torch.corrcoef(torch.stack([a, g]))[0, 1].item()
    print("bypass ch%d corr = %.4f" % (c, corr))
print("bypass SSIM = %.4f" % compute_ssim(amp_final2, amp_gt,
                                          data_range=1.0).item())
