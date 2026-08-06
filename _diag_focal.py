import sys
import torch
import numpy as np

sys.path.insert(0, ".")
torch.manual_seed(0)
dev = torch.device("cuda")

from src.data.dataset import THDataset
from src.models.holonet import ComplexHoloNet
from src.train.stage2 import build_propagator_padded
from src.optics.dpm import aadpm
from src.optics.aperture import filter_phs_only
from src.optics.complex_utils import compl_val, compl_exp
from src.utils.metrics import compute_ssim

wav = np.array([0.000450, 0.000520, 0.000638])
hp = {"wavelengths": wav, "pitch": 0.008, "res_h": 384, "res_w": 384}
prop = build_propagator_padded(hp, 0).to(dev)
wt = torch.tensor(wav, device=dev).view(1, -1, 1, 1)

holo = ComplexHoloNet(input_dim=4, num_layers=30, num_filters_per_layer=24).to(dev).eval()
holo.load_state_dict(torch.load(
    "model/ckpt_full_loss_pitch_8_layers_30_filters_24_stage1/stage1_latest.pth",
    map_location="cpu")["model_state_dict"])

ds = THDataset(
    "data/test_384_v2/test_04.tfrecord",
    {"res_h": 384, "res_w": 384, "sample_count": 100},
    ["amp_4", "phs_4", "img_0", "depth_0"], 0, True)
b = next(iter(torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)))
rgbd = b["rgbd"].to(dev)
amp_gt = b["amp_4"].to(dev)
phs_gt = b["phs_4"].to(dev)

holo_gt = compl_val(amp_gt, (phs_gt - 0.5) * 2.0 * np.pi)


def dpm_reconstruct(cpx_in):
    phs_only, amp_max = aadpm(
        cpx_in, propagator=prop, depth_shift=0.0, adaptive_phs_shift=False,
        batch=2, num_channels=3, res_h=384, res_w=384, sigma=0.0,
        kernel_width=3, phs_max=None, amp_max=None, clamp=True,
        normalize=False, wavelength=wav.tolist())
    amp_f, phs_f = filter_phs_only(
        phs_only, unnormalize_input=False, normalize_output=False,
        propagator=prop, depth_shift=-12.0, batch=2, num_channels=3,
        res_h=384, res_w=384, radius=None, phs_max=None, amp_max=amp_max,
        wavelength=wav.tolist())
    return compl_val(amp_f, phs_f), amp_max


with torch.no_grad():
    holo_mid = holo(rgbd)
    holo_shift = prop(holo_mid, 12.0) * compl_exp(-2 * np.pi * 12.0 / wt)
    # GT 场也做同样的 shift 编码（对照）
    holo_gt_shift = prop(holo_gt, 12.0) * compl_exp(-2 * np.pi * 12.0 / wt)

    holo_out_gt, am_gt = dpm_reconstruct(holo_gt_shift)
    holo_out_model, am_md = dpm_reconstruct(holo_shift)

    for focus in [-3.0, -1.5, 0.0, 1.5, 3.0]:
        img_gt = prop(holo_gt, -focus).abs()
        img_gt_out = prop(holo_out_gt, -focus).abs()
        img_md_out = prop(holo_out_model, -focus).abs()
        s_gt = compute_ssim(img_gt, img_gt_out, data_range=1.0).item()
        s_md = compute_ssim(img_gt, img_md_out, data_range=1.0).item()
        # 模型直接传播（无 DPM）对照
        s_direct = compute_ssim(img_gt, prop(holo_mid, -focus).abs(),
                                data_range=1.0).item()
        print("focus %+5.1fmm  GT->DPM ssim %.4f | model->DPM ssim %.4f | "
              "model direct ssim %.4f" % (focus, s_gt, s_md, s_direct))
