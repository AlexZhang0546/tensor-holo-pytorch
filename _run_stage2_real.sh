#!/bin/bash
# Stage-2 retrain with paper-faithful REAL DDPM (RealAmpPhaseDDPMNet).
# Identity warm-up (stage1 UNet as teacher) then joint with direct post-DPM SSIM loss.
cd /root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch
PY=/root/autodl-tmp/miniconda3/envs/holography/bin/python
$PY main.py --train-mode --train-stage stage2 \
  --model-arch unet --unet-depth 2 --unet-base-filters 24 --unet-tail-blocks 16 \
  --dataset-res 384 --activate-ddpm --restore-stage1 \
  --stage1-ckpt model/stage1_unet_d2t16/stage1_latest.pth \
  --stage2-ckpt-dir model/stage2_real_d2t16 \
  --stage2-epochs 8 --joint-epochs 120 --train-depth-shift 12.0 \
  --weight-holo-joint 0 --weight-ssim 30 --num-iter-per-test 500 \
  --ddpm-arch real --ddpm-bn tf \
  --batch 2 --learning-rate 1e-4 \
  2>&1 | tee stage2_real_d2t16.log
