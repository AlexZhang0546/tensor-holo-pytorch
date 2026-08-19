# -*- coding: utf-8 -*-
"""Single-sample stage-1 overfit for UNet: verify the model can quickly fit one
image with the stage-1 (pre-DPM) loss. Prints SSIM_amp/SSIM_img per step."""
import sys, os, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
BASE = "/root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch"
sys.path.insert(0, BASE)
from src.models.factory import build_main_net
from src.data.dataset import create_dataloader
from src.optics.propagation import propagator_factory
from src.train.stage1 import combine_loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img-idx', type=int, default=0)
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-ssim', type=float, default=50.0)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--freeze-bn', action='store_true')
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda')
    RES = 384
    wav = np.array([0.000450, 0.000520, 0.000638])
    hp = {"wavelengths": wav, "pitch": 0.008, "res_h": RES, "res_w": RES,
          "depth_base": -3, "depth_scale": 6, "double_pad": True}
    tp = {"batch": args.batch, "num_iter_per_test": 1000,
          "num_top_depth_for_img_loss": 15, "num_random_depth_for_img_loss": 5,
          "depth_dependent_weight_scale": 0.35, "num_hist_bins": 200,
          "depth_shift": 0.0, "deterministic_depths": False}
    lp = {"loss_type": "l1", "weight_holo": 1.0, "holo_method": "phase_aligned",
          "weight_fs": 20.0, "weight_fs_tv": 20.0, "weight_std": 0.02,
          "weight_mean": 0.03, "weight_ssim": args.weight_ssim}

    loader = create_dataloader(os.path.join(BASE, "data/validate_384_v2/validate_04.tfrecord"),
                               {"res_h": RES, "res_w": RES, "sample_count": 100},
                               ["amp_4", "phs_4", "img_0", "depth_0"], active_max_ldi_layer=0,
                               batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    samples = [b for b in loader]
    b = samples[args.img_idx % len(samples)]
    rgbd = b["rgbd"].to(device).repeat(args.batch, 1, 1, 1)
    target = b["target_complex"].to(device).repeat(args.batch, 1, 1, 1)
    print("single image", args.img_idx, "repeated batch", args.batch, flush=True)

    model = build_main_net(arch="unet", input_dim=4, unet_depth=2,
                           unet_base_filters=24, unet_tail_blocks=16).to(device)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck.get("model_state_dict", ck), strict=False)
        print("loaded", args.ckpt, flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print("params %.2fM" % (n_params/1e6), flush=True)
    if args.freeze_bn:
        model.eval()
        print("BN frozen (eval mode)", flush=True)
    else:
        model.train()

    prop = propagator_factory(input_shape=(RES, RES), pitch=0.008, wavelengths=wav,
                              method="as", double_pad=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8)
    loss_fn = F.l1_loss
    t0 = time.time()
    for it in range(args.steps):
        holo_out = model(rgbd)
        total, holo_l, fs_l, tv_l, s_amp, p_amp, s_img, p_img = combine_loss(
            holo_out, target, rgbd, prop, hp, tp, loss_fn, "l1", lp, pad=0,
            holo_method="phase_aligned")
        opt.zero_grad(); total.backward(); opt.step()
        if it % args.log_every == 0 or it == args.steps-1:
            print("[%4d] loss %.4f holo %.4f fs %.4f | SSIM_amp %.4f SSIM_img %.4f (%.0fs)" % (
                it, total.item(), holo_l.item(), fs_l.item(), s_amp.item(), s_img.item(),
                time.time()-t0), flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()