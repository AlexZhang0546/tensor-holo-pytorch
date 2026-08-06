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

holo = ComplexHoloNet(input_dim=4, num_layers=30, num_filters_per_layer=24).to(dev)
holo.load_state_dict(torch.load(
    "model/ckpt_full_loss_pitch_8_layers_30_filters_24_stage1/stage1_latest.pth",
    map_location="cpu")["model_state_dict"])
ddpm = ComplexDDPMNet(input_dim=3, output_dim=3, num_layers=8,
                      num_filters_per_layer=8).to(dev)
ddpm.load_state_dict(torch.load(
    "model/stage2_test_pad0_shift12/stage2_identity_epoch_0002.pth",
    map_location="cpu")["ddpm_net_state_dict"])

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

holo.train()
ddpm.train()
opt = torch.optim.Adam(list(holo.parameters()) + list(ddpm.parameters()),
                       lr=1e-4, betas=(0.9, 0.99), eps=1e-8)

for step, b in zip(range(30), loader):
    rgbd = b["rgbd"].to(dev)
    amp_gt = b["amp_4"].to(dev)
    phs_gt = b["phs_4"].to(dev)
    out = _run_stage2_forward(rgbd, amp_gt, phs_gt, holo, ddpm, prop, 12.0,
                              0, hologram_params, training_params, loss_params,
                              F.l1_loss, bypass_ddpm=False)
    opt.zero_grad()
    out["loss"].backward()
    gn_h = sum(p.grad.abs().sum().item() for p in holo.parameters()
               if p.grad is not None)
    gn_d = sum(p.grad.abs().sum().item() for p in ddpm.parameters()
               if p.grad is not None)
    opt.step()
    if step < 5 or step % 10 == 9:
        print("step %2d loss %.4f fs %.4f tv %.4f std %.4f | grad holonet %.4e ddpm %.4e"
              % (step, out["loss"].item(), out["fs_loss"].item(),
                 out["fs_tv"].item(), out["std_loss"].item(), gn_h, gn_d))

# 限幅饱和度统计（一次前向）
holo.eval()
ddpm.eval()
b = next(iter(loader))
rgbd = b["rgbd"].to(dev)
with torch.no_grad():
    holo_mid = holo(rgbd)
    holo_shift = prop(holo_mid, 12.0) * torch.exp(
        -2j * 2 * np.pi * 12.0 / torch.tensor(wav, device=dev).view(1, -1, 1, 1))
    ha = ddpm(holo_shift)
    max_in = holo_shift.abs().amax(dim=(2, 3), keepdim=True)
    capped = (ha.abs() > max_in).float().mean().item()
    print("\nDDPM cap saturation: %.2f%% pixels capped (|out|>max_in)" % (100 * capped))
    print("input amp: mean %.3f max %.3f | ddpm out amp: mean %.3f max %.3f"
          % (holo_shift.abs().mean().item(), max_in.max().item(),
             ha.abs().mean().item(), ha.abs().max().item()))
