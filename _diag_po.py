import sys
import torch
import numpy as np

sys.path.insert(0, ".")
torch.manual_seed(0)
dev = torch.device("cuda")

from src.data.dataset import THDataset
from src.train.stage2 import build_propagator_padded
from src.optics.dpm import aadpm
from src.optics.complex_utils import compl_val, compl_exp
from src.utils.metrics import compute_ssim

wav = np.array([0.000450, 0.000520, 0.000638])
hp = {"wavelengths": wav, "pitch": 0.008, "res_h": 384, "res_w": 384}
prop = build_propagator_padded(hp, 0).to(dev)
wt = torch.tensor(wav, device=dev).view(1, -1, 1, 1)

ds = THDataset(
    "data/test_384_v2/test_04.tfrecord",
    {"res_h": 384, "res_w": 384, "sample_count": 100},
    ["amp_4", "phs_4", "img_0", "depth_0"], 0, True)
b = next(iter(torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)))
amp_gt = b["amp_4"].to(dev)
phs_gt = b["phs_4"].to(dev)

holo_gt = compl_val(amp_gt, (phs_gt - 0.5) * 2.0 * np.pi)

with torch.no_grad():
    holo_gt_shift = prop(holo_gt, 12.0) * compl_exp(-2 * np.pi * 12.0 / wt)
    phs_only, amp_max = aadpm(
        holo_gt_shift, propagator=prop, depth_shift=0.0,
        adaptive_phs_shift=False, batch=2, num_channels=3,
        res_h=384, res_w=384, sigma=0.0, kernel_width=3,
        phs_max=None, amp_max=None, clamp=True, normalize=False,
        wavelength=wav.tolist())

    # 1) 纯相位重建：amplitude=1
    holo_po = compl_exp(phs_only)
    # 2) 2x2 局部平均重建（空间域 box 滤波恢复振幅），再乘 amp_max
    real = holo_gt_shift.real.unfold(2, 2, 2).unfold(3, 2, 2).mean(dim=(-1, -2))
    imag = holo_gt_shift.imag.unfold(2, 2, 2).unfold(3, 2, 2).mean(dim=(-1, -2))
    amp_2x2 = torch.complex(real, imag).abs()  # (B,C,192,192) 下采样振幅
    phs_2x2 = torch.complex(real, imag).angle()
    # 用 2x2 平均重建后的场（192x192），双线性上采样到 384
    up = torch.nn.functional.interpolate
    holo_avg = torch.complex(up(real, scale_factor=2, mode="bilinear",
                                align_corners=False),
                             up(imag, scale_factor=2, mode="bilinear",
                                align_corners=False))

    for focus in [-3.0, 0.0, 3.0]:
        img_gt = prop(holo_gt, -focus).abs()
        s_po = compute_ssim(img_gt, prop(holo_po, -focus).abs(),
                            data_range=1.0).item()
        s_avg = compute_ssim(img_gt, prop(holo_avg, -focus).abs(),
                             data_range=1.0).item()
        print("focus %+4.1fmm  phase-only ssim %.4f | 2x2-avg ssim %.4f" % (
            focus, s_po, s_avg))
