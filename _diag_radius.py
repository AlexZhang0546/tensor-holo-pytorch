import sys
import torch
import numpy as np

sys.path.insert(0, ".")
torch.manual_seed(0)
dev = torch.device("cuda")

from src.data.dataset import THDataset
from src.optics.dpm import aadpm
from src.optics.aperture import filter_phs_only, circ_filter
from src.optics.complex_utils import compl_val, fft2d, ifft2d, fftshift2d, ifftshift2d
from src.utils.metrics import compute_ssim

ds = THDataset(
    "data/test_384_v2/test_04.tfrecord",
    {"res_h": 384, "res_w": 384, "sample_count": 100},
    ["amp_4", "phs_4", "img_0", "depth_0"], 0, True)
b = next(iter(torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)))
amp_gt = b["amp_4"].to(dev)
phs_gt = b["phs_4"].to(dev)
wav = np.array([0.000450, 0.000520, 0.000638])

holo_gt = compl_val(amp_gt, (phs_gt - 0.5) * 2.0 * np.pi)

# 1) 纯滤波往返（无 DPM）：cpx -> fft -> mask -> ifft，半径扫描
print("== filter-only roundtrip (no DPM) ==")
for radius in [2, 4, 8, 16, 32, 64, 96, 128, 160, 192]:
    cpx_fft = fftshift2d(fft2d(holo_gt))
    mask = circ_filter(2, 3, 384, 384, radius, device=dev, dtype=holo_gt.dtype)
    cpx_out = ifft2d(ifftshift2d(cpx_fft * mask))
    s = compute_ssim(cpx_out.abs(), amp_gt, data_range=1.0).item()
    print("radius %3d: SSIM %.4f" % (radius, s))

# 2) DPM 往返（aadpm + filter），半径扫描
print("== DPM roundtrip (aadpm + filter) ==")
phs_only, amp_max = aadpm(
    holo_gt, propagator=None, depth_shift=0.0, adaptive_phs_shift=False,
    batch=2, num_channels=3, res_h=384, res_w=384, sigma=0.0,
    kernel_width=3, phs_max=None, amp_max=None, clamp=True,
    normalize=False, wavelength=wav.tolist())
print("amp_max:", [round(v, 3) for v in amp_max.flatten().tolist()])
for radius in [1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192]:
    amp_f, _ = filter_phs_only(
        phs_only, unnormalize_input=False, normalize_output=False,
        propagator=None, depth_shift=0.0, batch=2, num_channels=3,
        res_h=384, res_w=384, radius=radius, phs_max=None, amp_max=amp_max,
        wavelength=wav.tolist())
    corr = torch.corrcoef(torch.stack([amp_f[0, 1].flatten().float(),
                                       amp_gt[0, 1].flatten().float()]))[0, 1].item()
    print("radius %3d: SSIM %.4f corr %.4f amp_f mean %.3f" % (
        radius, compute_ssim(amp_f, amp_gt, data_range=1.0).item(), corr,
        amp_f.mean().item()))
