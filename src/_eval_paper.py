# -*- coding: utf-8 -*-
"""Paper-metric evaluation on the official validate split (paper Table 2 protocol).
Pre-DPM metrics: SSIM_amp/PSNR_amp (amplitude map) + SSIM_img/PSNR_img (15+5 focal stack).
Post-DPM metrics (stage2): same metrics computed through the DDPM/DPM/filter pipeline.
"""
import os, sys, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.factory import build_main_net
from src.models.real_ddpm_net import build_ddpm_net
from src.data.dataset import create_dataloader
from src.optics.propagation import propagator_factory
from src.train.stage1 import combine_loss
from src.train.stage2 import _run_stage2_forward

def build_propagator(res_h, res_w, pitch, wavelengths, double_pad=True):
    return propagator_factory(input_shape=(res_h, res_w), pitch=pitch,
                              wavelengths=wavelengths, method='as', double_pad=double_pad)

def load_weights(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get('model_state_dict', ckpt)
    res = model.load_state_dict(sd, strict=False)
    if res.missing_keys:
        print("missing keys:", res.missing_keys[:8])
    if res.unexpected_keys:
        print("unexpected keys:", res.unexpected_keys[:8])
    print("loaded", ckpt_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-path', required=True)
    ap.add_argument('--ddpm-ckpt-path', default=None)
    ap.add_argument('--model-arch', default='unet', choices=['holonet','unet'])
    ap.add_argument('--dataset-res', type=int, default=384)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--split', default='validate', choices=['validate','test'])
    ap.add_argument('--unet-depth', type=int, default=2)
    ap.add_argument('--unet-base-filters', type=int, default=24)
    ap.add_argument('--unet-tail-blocks', type=int, default=16)
    ap.add_argument('--unet-refine-blocks', type=int, default=0)
    ap.add_argument('--unet-global-in', action='store_true')
    ap.add_argument('--unet-stem-skip', action='store_true')
    ap.add_argument('--unet-attention', action='store_true')
    ap.add_argument('--unet-out-bn', action='store_true')
    ap.add_argument('--num-layers', type=int, default=30)
    ap.add_argument('--num-filters-per-layer', type=int, default=24)
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--stage2', action='store_true')
    ap.add_argument('--depth-shift', type=float, default=12.0)
    ap.add_argument('--aperture-radius', type=int, default=None)
    ap.add_argument('--ddpm-arch', default='real', choices=['real','complex'])
    ap.add_argument('--ddpm-bn', default='tf', choices=['tf','batch'])
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print("device", device)

    hologram_params = {
        'wavelengths': np.array([0.000450, 0.000520, 0.000638]),
        'pitch': 0.008, 'res_h': args.dataset_res, 'res_w': args.dataset_res,
        'depth_base': -3, 'depth_scale': 6, 'double_pad': True,
    }
    training_params = {
        'batch': args.batch, 'num_iter_per_test': 1000,
        'num_top_depth_for_img_loss': 15, 'num_random_depth_for_img_loss': 5,
        'depth_dependent_weight_scale': 0.35, 'num_hist_bins': 200,
        'depth_shift': args.depth_shift,
        'deterministic_depths': args.deterministic,
        'aperture_radius': args.aperture_radius,
    }
    loss_params = {'loss_type': 'l1', 'weight_holo': 1.0,
                   'weight_fs': 20.0, 'weight_fs_tv': 20.0,
                   'weight_std': 0.02, 'weight_mean': 0.03,
                   'weight_ssim': 0.0, 'weight_holo_joint': 0.0,
                   'holo_method': 'phase_aligned'}

    model = build_main_net(arch=args.model_arch, input_dim=4,
        num_layers=args.num_layers, num_filters_per_layer=args.num_filters_per_layer,
        unet_depth=args.unet_depth, unet_base_filters=args.unet_base_filters,
        unet_attention=args.unet_attention, unet_out_bn=args.unet_out_bn,
        unet_stem_skip=args.unet_stem_skip, unet_refine_blocks=args.unet_refine_blocks,
        unet_global_in=args.unet_global_in, unet_tail_blocks=args.unet_tail_blocks).to(device)
    load_weights(model, args.ckpt_path, device)
    model.eval()

    ddpm_net = None
    if args.stage2:
        ddpm_net = build_ddpm_net({"input_dim": 3, "output_dim": 3, "num_layers": 8,
            "num_filters_per_layer": 8, "interleave_rate": 1, "filter_width": 3,
            "bias_stddev": 0.01, "weight_var_scale": 0.25},
            arch=args.ddpm_arch, bn_mode=args.ddpm_bn).to(device)
        if args.ddpm_ckpt_path:
            load_weights(ddpm_net, args.ddpm_ckpt_path, device)
        else:
            ck = torch.load(args.ckpt_path, map_location=device)
            ddpm_net.load_state_dict(ck['ddpm_net_state_dict'])
        ddpm_net.eval()

    cur_dir = os.getcwd()
    tfrecord = os.path.join(cur_dir, 'data', '%s_%d_v2' % (args.split, args.dataset_res),
                            '%s_04.tfrecord' % args.split)
    loader = create_dataloader(tfrecord_path=tfrecord,
        dataset_params={'res_h': args.dataset_res, 'res_w': args.dataset_res, 'sample_count': 100},
        labels=['amp_4','phs_4','img_0','depth_0'], active_max_ldi_layer=0,
        batch_size=args.batch, shuffle=False, num_workers=2, drop_last=False)

    propagator = build_propagator(args.dataset_res, args.dataset_res,
                                  hologram_params['pitch'], hologram_params['wavelengths']).to(device)
    propagator_pad = propagator  # pad=0
    loss_fn = F.l1_loss

    agg = {'ssim_amp': [], 'psnr_amp': [], 'ssim_img': [], 'psnr_img': []}
    if args.stage2:
        agg2 = {'ssim_amp': [], 'psnr_amp': [], 'ssim_img': [], 'psnr_img': [], 'mean': [], 'std': []}
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            rgbd = batch['rgbd'].to(device)
            amp_gt = batch['amp_4'].to(device)
            phs_gt = batch['phs_4'].to(device)
            target = batch['target_complex'].to(device)
            holo_out = model(rgbd)
            _, holo_l, fs_l, _, s_amp, p_amp, s_img, p_img = combine_loss(
                holo_out, target, rgbd, propagator, hologram_params, training_params,
                loss_fn, 'l1', loss_params, pad=0, holo_method='phase_aligned')
            agg['ssim_amp'].append(s_amp.item()); agg['psnr_amp'].append(p_amp.item())
            agg['ssim_img'].append(s_img.item()); agg['psnr_img'].append(p_img.item())
            if args.stage2:
                r = _run_stage2_forward(rgbd, amp_gt, phs_gt, model, ddpm_net,
                    propagator_pad, args.depth_shift, 0, hologram_params,
                    training_params, loss_params, loss_fn, bypass_ddpm=False)
                agg2['ssim_amp'].append(r['ssim_amp'].item()); agg2['psnr_amp'].append(r['psnr_amp'].item())
                agg2['ssim_img'].append(r['ssim_img'].item()); agg2['psnr_img'].append(r['psnr_img'].item())
                agg2['mean'].append(r['mean_loss'].item()); agg2['std'].append(r['std_loss'].item())
            if bi % 10 == 0:
                print("batch %d/%d (%.0fs)" % (bi, len(loader), time.time()-t0), flush=True)
    print()
    print("==== pre-DPM (paper Table 2 protocol) on %s split ====" % args.split)
    for k in ['ssim_amp','psnr_amp','ssim_img','psnr_img']:
        a = np.array(agg[k])
        print("%s: mean %.4f  std %.4f  max %.4f  min %.4f" % (k, a.mean(), a.std(), a.max(), a.min()))
    if args.stage2:
        print("==== post-DPM (stage2 full pipeline) ====")
        for k in ['ssim_amp','psnr_amp','ssim_img','psnr_img','mean','std']:
            a = np.array(agg2[k])
            print("%s: mean %.4f  std %.4f  max %.4f  min %.4f" % (k, a.mean(), a.std(), a.max(), a.min()))
    print("DONE")

if __name__ == '__main__':
    main()