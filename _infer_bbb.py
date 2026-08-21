# -*- coding: utf-8 -*-
"""Single-image hologram inference from bbb photo using the final stage-2 checkpoint.
Pipeline mirrors _eval_paper.py stage2 path (depth shift -> real DDPM -> AA-DPM ->
aperture filter) so the result matches the final post-DPM metrics, and also saves
the pre-DPM CNN field for comparison.
"""
import os, sys, argparse, math
import numpy as np
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.factory import build_main_net
from src.models.real_ddpm_net import build_ddpm_net
from src.optics.propagation import propagator_factory
from src.optics.complex_utils import compl_val, compl_exp
from src.optics.dpm import aadpm
from src.optics.aperture import filter_phs_only


def complex_pad(x, pad):
    if pad == 0:
        return x
    return F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--rgb-path", required=True)
    ap.add_argument("--depth-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--res-h", type=int, default=1080)
    ap.add_argument("--res-w", type=int, default=1920)
    ap.add_argument("--depth-shift", type=float, default=12.0)
    ap.add_argument("--pitch", type=float, default=0.008)
    ap.add_argument("--unet-depth", type=int, default=2)
    ap.add_argument("--unet-base-filters", type=int, default=24)
    ap.add_argument("--unet-tail-blocks", type=int, default=16)
    ap.add_argument("--unet-out-bn", action="store_true")
    ap.add_argument("--unet-stem-skip", action="store_true")
    ap.add_argument("--unet-global-in", action="store_true")
    ap.add_argument("--unet-attention", action="store_true")
    ap.add_argument("--unet-refine-blocks", type=int, default=0)
    ap.add_argument("--aperture-radius", type=int, default=None)
    ap.add_argument("--ddpm-arch", default="real", choices=["real", "complex"])
    ap.add_argument("--ddpm-bn", default="tf", choices=["tf", "batch"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ---- build networks ----
    holonet = build_main_net(
        arch="unet", input_dim=4,
        num_layers=30, num_filters_per_layer=24,
        unet_depth=args.unet_depth, unet_base_filters=args.unet_base_filters,
        unet_attention=args.unet_attention, unet_out_bn=args.unet_out_bn,
        unet_stem_skip=args.unet_stem_skip, unet_refine_blocks=args.unet_refine_blocks,
        unet_global_in=args.unet_global_in, unet_tail_blocks=args.unet_tail_blocks,
    ).to(device)
    ddpm_net = build_ddpm_net(
        {"input_dim": 3, "output_dim": 3, "num_layers": 8,
         "num_filters_per_layer": 8, "interleave_rate": 1, "filter_width": 3,
         "bias_stddev": 0.01, "weight_var_scale": 0.25},
        arch=args.ddpm_arch, bn_mode=args.ddpm_bn,
    ).to(device)

    ck = torch.load(args.ckpt_path, map_location="cpu")
    if "model_state_dict" in ck:
        r = holonet.load_state_dict(ck["model_state_dict"], strict=False)
        print("holonet missing:", len(r.missing_keys), "unexpected:", len(r.unexpected_keys))
        if "ddpm_net_state_dict" in ck:
            r2 = ddpm_net.load_state_dict(ck["ddpm_net_state_dict"], strict=False)
            print("ddpm missing:", len(r2.missing_keys), "unexpected:", len(r2.unexpected_keys))
        else:
            print("WARNING: no ddpm_net_state_dict in ckpt")
    else:
        holonet.load_state_dict(ck)
    holonet.eval()
    ddpm_net.eval()

    # ---- load input image ----
    rgb = cv2.resize(cv2.imread(args.rgb_path, cv2.IMREAD_COLOR)[:, :, ::-1],
                     (args.res_w, args.res_h), interpolation=cv2.INTER_CUBIC)
    rgb = np.transpose(rgb, (2, 0, 1)) / 255.0
    depth = cv2.imread(args.depth_path, cv2.IMREAD_GRAYSCALE)
    depth = cv2.resize(depth, (args.res_w, args.res_h), interpolation=cv2.INTER_CUBIC)
    depth = depth.astype(np.float32)[None, :, :] / 255.0
    rgbd = torch.from_numpy(np.concatenate([rgb, depth], axis=0)).unsqueeze(0).float().to(device)
    print("input rgbd:", tuple(rgbd.shape))

    hologram_params = {
        "wavelengths": np.array([0.000450, 0.000520, 0.000638]),
        "pitch": args.pitch, "res_h": args.res_h, "res_w": args.res_w,
        "depth_base": -3, "depth_scale": 6, "double_pad": True,
    }
    propagator = propagator_factory(input_shape=(args.res_h, args.res_w), pitch=args.pitch,
                                    wavelengths=hologram_params["wavelengths"],
                                    method="as", double_pad=True).to(device)
    wl = torch.tensor(hologram_params["wavelengths"], device=device).view(1, -1, 1, 1)

    os.makedirs(args.out_dir, exist_ok=True)
    with torch.no_grad():
        # pre-DPM CNN field
        holo_mid = holonet(rgbd)
        amp_pre = holo_mid.abs()
        phs_pre = torch.angle(holo_mid) / (2.0 * np.pi) + 0.5
        print("pre-DPM amp range: %.3f-%.3f" % (amp_pre.min().item(), amp_pre.max().item()))

        # depth shift + DDPM
        holo_shifted = propagator(holo_mid, args.depth_shift) * \
            compl_exp(-2 * np.pi * args.depth_shift / wl)
        holo_altered = ddpm_net(holo_shifted)

        # AA-DPM (training/eval path: normalize=False -> radians)
        phs_only, amp_max = aadpm(
            holo_altered, propagator=propagator, depth_shift=0.0,
            adaptive_phs_shift=False, batch=1, num_channels=3,
            res_h=holo_altered.shape[2], res_w=holo_altered.shape[3],
            sigma=0.0, kernel_width=3, phs_max=None, amp_max=None,
            clamp=True, normalize=False, wavelength=hologram_params["wavelengths"])
        # aperture filter back to midpoint
        amp_final, phs_final = filter_phs_only(
            phs_only, unnormalize_input=False, normalize_output=False,
            propagator=propagator, depth_shift=-args.depth_shift,
            batch=1, num_channels=3,
            res_h=holo_altered.shape[2], res_w=holo_altered.shape[3],
            radius=args.aperture_radius, phs_max=None, amp_max=amp_max,
            wavelength=hologram_params["wavelengths"])
        amp_final = amp_final.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (H,W,3)

    # ---- save visuals ----
    def norm255(a, lo=None, hi=None):
        a = a.astype(np.float64)
        lo = a.min() if lo is None else lo
        hi = a.max() if hi is None else hi
        return np.clip((a - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255).astype(np.uint8)

    # pre-DPM amp/phase
    amp_pre_np = amp_pre.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    phs_pre_np = phs_pre.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    cv2.imwrite(os.path.join(args.out_dir, "pre_amp.png"), norm255(amp_pre_np))
    cv2.imwrite(os.path.join(args.out_dir, "pre_phs.png"), norm255(phs_pre_np))

    # phase-only hologram per channel: wrap to [0,2pi) then to [0,1]
    phs_np = phs_only.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # radians
    holo_gray = norm255(np.mod(phs_np, 2 * np.pi) / (2 * np.pi))
    cv2.imwrite(os.path.join(args.out_dir, "holo_b.png"), holo_gray[:, :, 0])
    cv2.imwrite(os.path.join(args.out_dir, "holo_g.png"), holo_gray[:, :, 1])
    cv2.imwrite(os.path.join(args.out_dir, "holo_r.png"), holo_gray[:, :, 2])
    # RGB composite (BGR for cv2)
    holo_rgb = np.stack([holo_gray[:, :, 2], holo_gray[:, :, 1], holo_gray[:, :, 0]], axis=2)
    cv2.imwrite(os.path.join(args.out_dir, "holo_rgb.png"), holo_rgb)
    cv2.imwrite(os.path.join(args.out_dir, "holo_phase.png"), holo_gray)

    # reconstruction (what you see after optical playback)
    recon_bgr = np.clip(amp_final * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.out_dir, "recon_rgb.png"), recon_bgr[:, :, ::-1])
    cv2.imwrite(os.path.join(args.out_dir, "recon_b.png"), norm255(amp_final[:, :, 0]))
    cv2.imwrite(os.path.join(args.out_dir, "recon_g.png"), norm255(amp_final[:, :, 1]))
    cv2.imwrite(os.path.join(args.out_dir, "recon_r.png"), norm255(amp_final[:, :, 2]))

    print("saved to", args.out_dir)
    print("DONE")


if __name__ == "__main__":
    main()
