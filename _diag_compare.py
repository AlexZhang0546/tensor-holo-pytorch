import sys
import os
import torch
import numpy as np

sys.path.insert(0, ".")
torch.manual_seed(0)
dev = torch.device("cuda")

from src.data.dataset import THDataset
from src.train.stage2 import build_propagator_padded
from src.optics.dpm import aadpm
from src.optics.aperture import filter_phs_only
from src.optics.complex_utils import compl_val, compl_exp

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

with torch.no_grad():
    holo_shift = prop(holo_gt, 12.0) * compl_exp(-2 * np.pi * 12.0 / wt)
    phs_only, amp_max = aadpm(
        holo_shift, propagator=prop, depth_shift=0.0,
        adaptive_phs_shift=False, batch=1, num_channels=3,
        res_h=384, res_w=384, sigma=0.0, kernel_width=3,
        phs_max=None, amp_max=None, clamp=True, normalize=False,
        wavelength=wav.tolist())
    amp_out, phs_out = filter_phs_only(
        phs_only, unnormalize_input=False, normalize_output=False,
        propagator=prop, depth_shift=-12.0, batch=1, num_channels=3,
        res_h=384, res_w=384, radius=None, phs_max=None, amp_max=amp_max,
        wavelength=wav.tolist())


def load(name):
    return np.load(os.path.join(DUMP, name + ".npy"))


def compare(name, port_val, orig_val, tol=1e-3):
    port_np = port_val.detach().cpu().numpy()[0]
    diff = np.abs(port_np - orig_val)
    print("%s: max abs diff %.6f (port %.4f..%.4f, orig %.4f..%.4f)" % (
        name, diff.max(), port_np.min(), port_np.max(),
        orig_val.min(), orig_val.max()))
    return diff


compare("holo_shift real", holo_shift.real, load("orig_holo_shift").real)
compare("holo_shift imag", holo_shift.imag, load("orig_holo_shift").imag)
compare("phs_only", phs_only, load("orig_phs_only"))
compare("amp_out", amp_out, load("orig_amp_out"))
compare("phs_out", phs_out, load("orig_phs_out"))
print("port amp_max:", [round(v, 4) for v in amp_max.flatten().tolist()])
print("orig amp_max: 1.3270")
