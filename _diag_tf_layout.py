import sys
import os
import torch
import numpy as np

sys.path.insert(0, ".")
torch.manual_seed(0)
dev = torch.device("cuda")

from src.data.dataset import THDataset
from src.train.stage2 import build_propagator_padded
from src.optics.aperture import filter_phs_only
from src.optics.complex_utils import compl_val, compl_exp
from src.utils.metrics import compute_ssim

DUMP = "_gt_dump"
wav = np.array([0.000450, 0.000520, 0.000638])
hp = {"wavelengths": wav, "pitch": 0.008, "res_h": 384, "res_w": 384}
prop = build_propagator_padded(hp, 0).to(dev)
wt = torch.tensor(wav, device=dev).view(1, -1, 1, 1)

ds = THDataset(
    "data/test_384_v2/test_04.tfrecord",
    {"res_h": 384, "res_w": 384, "sample_count": 100},
    ["amp_4", "phs_4", "img_0", "depth_0"], 0, True)
b = next(iter(torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)))
amp_gt = b["amp_4"].to(dev)
phs_gt = b["phs_4"].to(dev)
holo_gt = compl_val(amp_gt, (phs_gt - 0.5) * 2.0 * np.pi)


def depth_to_space_tf(x, r):
    """TF NCHW depth_to_space: out[b,c,h*r+t,w*r+s] = in[b, c + C*s + C*r*t, h, w]"""
    B, Ck, H, W = x.shape
    C = Ck // (r * r)
    y = x.view(B, r, r, C, H, W)          # (t, s, c)
    y = y.permute(0, 3, 4, 1, 5, 2).contiguous()  # (B, C, H, t, W, s)
    return y.reshape(B, C, H * r, W * r)


def aadpm_tf_layout(cpx):
    amp = cpx.abs()
    phs = cpx.angle()
    amp_max = amp.amax(dim=(2, 3), keepdim=True) + 1e-6
    amp = amp / amp_max
    amp = torch.clamp(amp, max=1.0 - 1e-6)
    phs_zero_mean = phs - phs.mean(dim=[2, 3], keepdim=True)
    phs_offset = torch.acos(torch.clamp(amp, min=-1.0 + 1e-7, max=1.0 - 1e-7))
    phs_low = phs_zero_mean - phs_offset
    phs_high = phs_zero_mean + phs_offset
    p11 = phs_low[:, :, 0::2, 0::2]
    p12 = phs_high[:, :, 0::2, 1::2]
    p21 = phs_high[:, :, 1::2, 0::2]
    p22 = phs_low[:, :, 1::2, 1::2]
    stacked = torch.cat([p11, p12, p21, p22], dim=1)
    return depth_to_space_tf(stacked, 2), amp_max


with torch.no_grad():
    holo_shift = prop(holo_gt, 12.0) * compl_exp(-2 * np.pi * 12.0 / wt)
    phs_tf, amp_max = aadpm_tf_layout(holo_shift)

    orig_phs = np.load(os.path.join(DUMP, "orig_phs_only.npy"))
    port_np = phs_tf.detach().cpu().numpy()[0]
    d = np.abs(port_np - orig_phs)
    print("phs_only vs orig: max diff %.6f, >0.1 rad: %.4f%%" % (
        d.max(), 100 * (d > 0.1).mean()))

    amp_f, _ = filter_phs_only(
        phs_tf, unnormalize_input=False, normalize_output=False,
        propagator=prop, depth_shift=-12.0, batch=1, num_channels=3,
        res_h=384, res_w=384, radius=None, phs_max=None, amp_max=amp_max,
        wavelength=wav.tolist())
    s = compute_ssim(amp_f, amp_gt, data_range=1.0).item()
    print("SSIM with TF layout: %.4f (orig target 0.3728)" % s)
