# -*- coding: utf-8 -*-
"""Pure low-pass resolution loss: SSIM_img of low-passed GT field at various radii.
Shows how much quality the Fourier aperture alone removes (no DPM).
"""
import sys, os, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, "/root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch")
from src.data.dataset import create_dataloader
from src.optics.propagation import propagator_factory
from src.optics.complex_utils import compl_val, fft2d, ifft2d, fftshift2d, ifftshift2d
from src.optics.aperture import circ_filter
from src.losses.focal_stack import compute_focal_stack_loss

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

loader = create_dataloader(os.path.join(BASE, "data/validate_384_v2/validate_04.tfrecord"),
                           {"res_h": RES, "res_w": RES, "sample_count": 100},
                           ["amp_4", "phs_4", "img_0", "depth_0"], active_max_ldi_layer=0,
                           batch_size=1, shuffle=False, num_workers=0, drop_last=False)

def lp(field, radius):
    mask = circ_filter(1, 3, RES, RES, radius, device=device, dtype=torch.complex64)
    return ifft2d(ifftshift2d(fftshift2d(fft2d(field)) * mask))

with torch.no_grad():
    for k, b in enumerate(loader):
        if k > 4:
            break
        rgbd = b["rgbd"].to(device)
        amp_gt = b["amp_4"].to(device)
        phs_gt = b["phs_4"].to(device)
        holo_gt = compl_val(amp_gt, (phs_gt - 0.5) * 2.0 * np.pi)
        row = []
        for r in [192, 160, 128, 96]:
            _, _, ssim_img, _ = compute_focal_stack_loss(lp(holo_gt, r), holo_gt, rgbd, prop, hp, tp, F.l1_loss, 0)
            row.append("%.4f" % ssim_img.item())
        print("img%d  lowpass GT SSIM_img  r192=%s r160=%s r128=%s r96=%s" % (k, *row), flush=True)
print("DONE", flush=True)
