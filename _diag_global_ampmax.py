import sys
import torch
import numpy as np

sys.path.insert(0, ".")
torch.manual_seed(0)
dev = torch.device("cuda")

from src.data.dataset import THDataset
from src.optics.dpm import aadpm
from src.optics.aperture import filter_phs_only
from src.optics.complex_utils import compl_val
from src.utils.metrics import compute_ssim

ds = THDataset(
    "data/test_384_v2/test_04.tfrecord",
    {"res_h": 384, "res_w": 384, "sample_count": 100},
    ["amp_4", "phs_4", "img_0", "depth_0"], 0, True)
b = next(iter(torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)))
amp_gt = b["amp_4"].to(dev)
phs_gt = b["phs_4"].to(dev)
wav = np.array([0.000450, 0.000520, 0.000638])
holo_gt = compl_val(amp_gt, (phs_gt - 0.5) * 2.0 * np.pi)

# 编码（不传 amp_max，内部默认逐通道）
phs_only, amp_max_per = aadpm(
    holo_gt, propagator=None, depth_shift=0.0, adaptive_phs_shift=False,
    batch=1, num_channels=3, res_h=384, res_w=384, sigma=0.0,
    kernel_width=3, phs_max=None, amp_max=None, clamp=True,
    normalize=False, wavelength=wav.tolist())

# 全局 amp_max（与原始一致）
amp_global = holo_gt.abs().max() + 1e-6
print("amp_max per-channel:", [round(v, 4) for v in amp_max_per.flatten().tolist()])
print("amp_max global:", round(amp_global.item(), 4))

for label, am in [("per-channel", amp_max_per), ("global", amp_global)]:
    for radius in [192, 64, 32]:
        amp_f, _ = filter_phs_only(
            phs_only, unnormalize_input=False, normalize_output=False,
            propagator=None, depth_shift=0.0, batch=1, num_channels=3,
            res_h=384, res_w=384, radius=radius, phs_max=None, amp_max=am,
            wavelength=wav.tolist())
        corr = torch.corrcoef(torch.stack([amp_f[0, 1].flatten().float(),
                                           amp_gt[0, 1].flatten().float()]))[0, 1].item()
        print("%s radius %3d: SSIM %.4f corr %.4f amp_f mean %.3f max %.2f" % (
            label, radius, compute_ssim(amp_f, amp_gt, data_range=1.0).item(),
            corr, amp_f.mean().item(), amp_f.max().item()))
