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

wav = np.array([0.000450, 0.000520, 0.000638])
hp = {"wavelengths": wav, "pitch": 0.008, "res_h": 384, "res_w": 384}
prop = build_propagator_padded(hp, 0).to(dev)
wt = torch.tensor(wav, device=dev).view(1, -1, 1, 1)


def encode_filter(cpx, depth_shift_in, depth_shift_out):
    if depth_shift_in != 0.0:
        cpx = prop(cpx, depth_shift_in) * compl_exp(
            -2 * np.pi * depth_shift_in / wt)
    phs, amax = aadpm(
        cpx, propagator=prop, depth_shift=0.0, adaptive_phs_shift=False,
        batch=2, num_channels=3, res_h=384, res_w=384, sigma=0.0,
        kernel_width=3, phs_max=None, amp_max=None, clamp=True,
        normalize=False, wavelength=wav.tolist())
    amp_f, _ = filter_phs_only(
        phs, unnormalize_input=False, normalize_output=False,
        propagator=prop, depth_shift=depth_shift_out, batch=2, num_channels=3,
        res_h=384, res_w=384, radius=None, phs_max=None, amp_max=amax,
        wavelength=wav.tolist())
    return amp_f, amax


with torch.no_grad():
    holo_gt = compl_val(amp_gt, (phs_gt - 0.5) * 2.0 * np.pi)

    # 1) GT 中点场，直接编码+滤波（无位移）
    af1, am1 = encode_filter(holo_gt, 0.0, 0.0)
    print("GT mid->mid:          SSIM %.4f  corr %.4f  amp_f mean %.3f max %.2f" % (
        compute_ssim(af1, amp_gt, data_range=1.0).item(),
        torch.corrcoef(torch.stack([af1[0, 1].flatten().float(),
                                    amp_gt[0, 1].flatten().float()]))[0, 1].item(),
        af1.mean().item(), af1.max().item()))

    # 2) GT 场，+12 位移后编码，再 -12 传播回（完整 stage2 GT 回路）
    af2, am2 = encode_filter(holo_gt, 12.0, -12.0)
    print("GT +12->-12:          SSIM %.4f  corr %.4f  amp_f mean %.3f max %.2f" % (
        compute_ssim(af2, amp_gt, data_range=1.0).item(),
        torch.corrcoef(torch.stack([af2[0, 1].flatten().float(),
                                    amp_gt[0, 1].flatten().float()]))[0, 1].item(),
        af2.mean().item(), af2.max().item()))

    # 3) 模型输出 holo_mid，直接编码+滤波（无位移）
    holo_mid = holo(rgbd)
    af3, am3 = encode_filter(holo_mid, 0.0, 0.0)
    print("model mid->mid:       SSIM %.4f  corr %.4f  amp_f mean %.3f max %.2f" % (
        compute_ssim(af3, amp_gt, data_range=1.0).item(),
        torch.corrcoef(torch.stack([af3[0, 1].flatten().float(),
                                    amp_gt[0, 1].flatten().float()]))[0, 1].item(),
        af3.mean().item(), af3.max().item()))

    # 4) 模型输出 +12 编码，-12 传回（stage2 实际路径，bypass ddpm）
    af4, am4 = encode_filter(holo_mid, 12.0, -12.0)
    print("model +12->-12:       SSIM %.4f  corr %.4f  amp_f mean %.3f max %.2f" % (
        compute_ssim(af4, amp_gt, data_range=1.0).item(),
        torch.corrcoef(torch.stack([af4[0, 1].flatten().float(),
                                    amp_gt[0, 1].flatten().float()]))[0, 1].item(),
        af4.mean().item(), af4.max().item()))

    # 5) 模型输出 +12 编码，不传回（在位移平面比较）
    af5, am5 = encode_filter(holo_mid, 12.0, 0.0)
    hs = prop(holo_mid, 12.0) * compl_exp(-2 * np.pi * 12.0 / wt)
    hs_norm = hs.abs() / am5
    print("model +12 (at plane): SSIM %.4f  corr %.4f  amp_f mean %.3f max %.2f" % (
        compute_ssim(af5, hs_norm, data_range=1.0).item(),
        torch.corrcoef(torch.stack([af5[0, 1].flatten().float(),
                                    hs_norm[0, 1].flatten().float()]))[0, 1].item(),
        af5.mean().item(), af5.max().item()))
