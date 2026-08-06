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
from src.optics.complex_utils import compl_exp
from src.utils.metrics import compute_ssim

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

wav = np.array([0.000450, 0.000520, 0.000638])
hp = {"wavelengths": wav, "pitch": 0.008, "res_h": 384, "res_w": 384}
prop = build_propagator_padded(hp, 0).to(dev)
wt = torch.tensor(wav, device=dev).view(1, -1, 1, 1)


def pipeline(holo_altered, amp_max_override=None):
    phs_only, amp_max = aadpm(
        holo_altered, propagator=prop, depth_shift=0.0,
        adaptive_phs_shift=False, batch=2, num_channels=3,
        res_h=384, res_w=384, sigma=0.0, kernel_width=3,
        phs_max=None, amp_max=amp_max_override, clamp=True, normalize=False,
        wavelength=wav.tolist())
    amp_final, _ = filter_phs_only(
        phs_only, unnormalize_input=False, normalize_output=False,
        propagator=prop, depth_shift=-12.0, batch=2, num_channels=3,
        res_h=384, res_w=384, radius=None, phs_max=None, amp_max=amp_max,
        wavelength=wav.tolist())
    return amp_final, amp_max


with torch.no_grad():
    hs = prop(holo(rgbd), 12.0) * compl_exp(-2 * np.pi * 12.0 / wt)
    ha = ddpm(hs)

    # 原始路径
    af0, am0 = pipeline(ha)
    print("raw:        SSIM %.4f  amp_max %.2f" % (
        compute_ssim(af0, amp_gt, data_range=1.0).item(), am0.max().item()))

    # 方案A：DDPM 输出振幅限幅到输入每通道 max
    max_in = hs.abs().amax(dim=(2, 3), keepdim=True)
    scale = torch.clamp(max_in / (ha.abs() + 1e-6), max=1.0)
    ha_cap = ha * scale
    af1, am1 = pipeline(ha_cap)
    print("cap@max_in: SSIM %.4f  amp_max %.2f" % (
        compute_ssim(af1, amp_gt, data_range=1.0).item(), am1.max().item()))

    # 方案A'：限幅到固定 √2（与原始 tanh 上界一致）
    bound = float(np.sqrt(2.0))
    scale2 = torch.clamp(bound / (ha.abs() + 1e-6), max=1.0)
    ha_cap2 = ha * scale2
    af2, am2 = pipeline(ha_cap2)
    print("cap@sqrt2:  SSIM %.4f  amp_max %.2f" % (
        compute_ssim(af2, amp_gt, data_range=1.0).item(), am2.max().item()))

    # 方案B：不用 max，用 99.9 分位数作为 amp_max（对尖峰鲁棒）
    phs_only, _ = aadpm(
        ha, propagator=prop, depth_shift=0.0, adaptive_phs_shift=False,
        batch=2, num_channels=3, res_h=384, res_w=384, sigma=0.0,
        kernel_width=3, phs_max=None, amp_max=None, clamp=True,
        normalize=False, wavelength=wav.tolist())
    am_robust = torch.zeros(2, 3, 1, 1, device=dev)
    for i in range(2):
        for c in range(3):
            a = ha[i, c].abs().flatten().float()
            am_robust[i, c, 0, 0] = torch.quantile(a, 0.999) + 1e-6
    af3, _ = filter_phs_only(
        phs_only, unnormalize_input=False, normalize_output=False,
        propagator=prop, depth_shift=-12.0, batch=2, num_channels=3,
        res_h=384, res_w=384, radius=None, phs_max=None, amp_max=am_robust,
        wavelength=wav.tolist())
    print("p999 amp_max: SSIM %.4f  amp_max %.2f" % (
        compute_ssim(af3, amp_gt, data_range=1.0).item(),
        am_robust.max().item()))

    # 基线：DDPM 输出与 GT 直接比较（限幅后）
    print("baseline SSIM(holo_shifted amp, gt) = %.4f" % compute_ssim(
        hs.abs(), amp_gt, data_range=1.0).item())
