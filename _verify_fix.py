import sys
import torch
import numpy as np

sys.path.insert(0, ".")
torch.manual_seed(0)
dev = torch.device("cuda")

from src.data.dataset import THDataset
from src.models.holonet import ComplexHoloNet
from src.models.ddpm_net import ComplexDDPMNet
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
ddpm = ComplexDDPMNet(input_dim=3, output_dim=3, num_layers=8,
                      num_filters_per_layer=8).to(dev).eval()
ddpm.load_state_dict(torch.load(
    "model/stage2_test_pad0_shift12/stage2_identity_epoch_0002.pth",
    map_location="cpu")["ddpm_net_state_dict"])

ds = THDataset(
    "data/test_384_v2/test_04.tfrecord",
    {"res_h": 384, "res_w": 384, "sample_count": 100},
    ["amp_4", "phs_4", "img_0", "depth_0"], 0, True)
b = next(iter(torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)))
rgbd = b["rgbd"].to(dev)
amp_gt = b["amp_4"].to(dev)
phs_gt = b["phs_4"].to(dev)
holo_gt = compl_val(amp_gt, (phs_gt - 0.5) * 2.0 * np.pi)


def reconstruct(cpx_in, depth_shift_back):
    phs_only, amp_max = aadpm(
        cpx_in, propagator=prop, depth_shift=0.0, adaptive_phs_shift=False,
        batch=2, num_channels=3, res_h=384, res_w=384, sigma=0.0,
        kernel_width=3, phs_max=None, amp_max=None, clamp=True,
        normalize=False, wavelength=wav.tolist())
    amp_f, phs_f = filter_phs_only(
        phs_only, unnormalize_input=False, normalize_output=False,
        propagator=prop, depth_shift=depth_shift_back, batch=2,
        num_channels=3, res_h=384, res_w=384, radius=None, phs_max=None,
        amp_max=amp_max, wavelength=wav.tolist())
    return compl_val(amp_f, phs_f), amp_max


with torch.no_grad():
    # 1) GT 场完整往返
    holo_gt_shift = prop(holo_gt, 12.0) * compl_exp(-2 * np.pi * 12.0 / wt)
    holo_out_gt, am_gt = reconstruct(holo_gt_shift, -12.0)
    print("GT roundtrip: SSIM_amp %.4f | focal -3/0/+3: %.4f %.4f %.4f" % (
        compute_ssim(holo_out_gt.abs(), amp_gt, data_range=1.0).item(),
        compute_ssim(prop(holo_gt, 3.0).abs(),
                     prop(holo_out_gt, 3.0).abs(), data_range=1.0).item(),
        compute_ssim(prop(holo_gt, 0.0).abs(),
                     prop(holo_out_gt, 0.0).abs(), data_range=1.0).item(),
        compute_ssim(prop(holo_gt, -3.0).abs(),
                     prop(holo_out_gt, -3.0).abs(), data_range=1.0).item()))

    # 2) stage1 模型输出（bypass DDPM）往返
    holo_mid = holo(rgbd)
    holo_shift = prop(holo_mid, 12.0) * compl_exp(-2 * np.pi * 12.0 / wt)
    holo_out_md, am_md = reconstruct(holo_shift, -12.0)
    print("model (bypass): SSIM_amp %.4f  amp_max %.2f" % (
        compute_ssim(holo_out_md.abs(), amp_gt, data_range=1.0).item(),
        am_md.max().item()))

    # 3) 完整 stage2（identity DDPM）
    holo_altered = ddpm(holo_shift)
    holo_out_ddpm, am_ddpm = reconstruct(holo_altered, -12.0)
    print("model + ddpm:   SSIM_amp %.4f  amp_max %.2f  ddpm_out max %.2f" % (
        compute_ssim(holo_out_ddpm.abs(), amp_gt, data_range=1.0).item(),
        am_ddpm.max().item(), holo_altered.abs().max().item()))

    # 4) ddpm + 振幅限幅到输入 max（缓解尖峰）
    max_in = holo_shift.abs().amax(dim=(2, 3), keepdim=True)
    scale = torch.clamp(max_in / (holo_altered.abs() + 1e-6), max=1.0)
    holo_out_cap, am_cap = reconstruct(holo_altered * scale, -12.0)
    print("model + ddpm(cap): SSIM_amp %.4f  amp_max %.2f" % (
        compute_ssim(holo_out_cap.abs(), amp_gt, data_range=1.0).item(),
        am_cap.max().item()))
