import sys
import torch
import numpy as np
import torch.nn.functional as F

sys.path.insert(0, ".")
torch.manual_seed(0)
dev = torch.device("cuda")

from src.data.dataset import THDataset
from src.models.holonet import ComplexHoloNet
from src.models.ddpm_net import ComplexDDPMNet
from src.train.stage2 import _run_stage2_forward, build_propagator_padded
from src.optics.complex_utils import compl_exp


def make_models():
    holo = ComplexHoloNet(input_dim=4, num_layers=30,
                          num_filters_per_layer=24).to(dev)
    holo.load_state_dict(torch.load(
        "model/ckpt_full_loss_pitch_8_layers_30_filters_24_stage1/stage1_latest.pth",
        map_location="cpu")["model_state_dict"])
    ddpm = ComplexDDPMNet(input_dim=3, output_dim=3, num_layers=8,
                          num_filters_per_layer=8).to(dev)
    ddpm.load_state_dict(torch.load(
        "model/stage2_test_pad0_shift12/stage2_identity_epoch_0002.pth",
        map_location="cpu")["ddpm_net_state_dict"])
    return holo, ddpm


wav = np.array([0.000450, 0.000520, 0.000638])
hologram_params = {"wavelengths": wav, "pitch": 0.008, "res_h": 384,
                   "res_w": 384, "depth_base": -3.0, "depth_scale": 6.0}
training_params = {"num_top_depth_for_img_loss": 15,
                   "num_random_depth_for_img_loss": 5,
                   "depth_dependent_weight_scale": 0.35, "num_hist_bins": 200,
                   "padding": 0, "depth_shift": 12.0}
loss_params = {"loss_type": "l1", "weight_fs": 20.0, "weight_fs_tv": 20.0,
               "weight_std": 0.02, "weight_mean": 0.03,
               "phs_max": [2 * np.pi] * 3}
prop = build_propagator_padded(hologram_params, 0).to(dev)

ds = THDataset(
    "data/test_384_v2/test_04.tfrecord",
    {"res_h": 384, "res_w": 384, "sample_count": 100},
    ["amp_4", "phs_4", "img_0", "depth_0"], 0, True)
loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True)
batches = [next(iter(loader)) for _ in range(3)]


def run_config(lr, steps, label):
    holo, ddpm = make_models()
    holo.train()
    ddpm.train()
    opt = torch.optim.Adam(list(holo.parameters()) + list(ddpm.parameters()),
                           lr=lr, betas=(0.9, 0.99), eps=1e-8)
    losses = []
    for step in range(steps):
        b = batches[step % 3]
        rgbd = b["rgbd"].to(dev)
        amp_gt = b["amp_4"].to(dev)
        phs_gt = b["phs_4"].to(dev)
        out = _run_stage2_forward(rgbd, amp_gt, phs_gt, holo, ddpm, prop,
                                  12.0, 0, hologram_params, training_params,
                                  loss_params, F.l1_loss, bypass_ddpm=False)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        losses.append(out["loss"].item())
    wins = [losses[i] for i in (0, steps // 4 - 1, steps // 2 - 1,
                                steps * 3 // 4 - 1, steps - 1)]
    print("%s (lr=%.0e, %d steps): losses at %s  final %.4f" % (
        label, lr, steps, ["%.3f" % v for v in wins], losses[-1]))


run_config(1e-4, 200, "sqrt2-cap lr1e-4")
run_config(3e-5, 200, "sqrt2-cap lr3e-5")

# 限幅饱和度（修正 dtype）
holo, ddpm = make_models()
holo.eval()
ddpm.eval()
b = next(iter(loader))
rgbd = b["rgbd"].to(dev)
with torch.no_grad():
    holo_mid = holo(rgbd)
    wt = torch.tensor(wav, device=dev, dtype=torch.float32).view(1, -1, 1, 1)
    holo_shift = prop(holo_mid, 12.0) * compl_exp(-2 * np.pi * 12.0 / wt)
    ha = ddpm(holo_shift)
    bound = np.sqrt(2.0)
    capped = (ha.abs() > bound).float().mean().item()
    print("\nDDPM sqrt2-cap saturation: %.4f%% pixels capped" % (100 * capped))
    print("ddpm out amp: mean %.3f max %.3f | input amp max %.3f" % (
        ha.abs().mean().item(), ha.abs().max().item(),
        holo_shift.abs().amax(dim=(2, 3)).max().item()))
