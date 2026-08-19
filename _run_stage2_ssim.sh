#!/bin/bash
# Stage-2 joint (long) for UNet d2t16 with direct post-DPM SSIM loss.
# Resumes the existing joint ckpt. No holonet anchor (official-style joint).
cd /root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch
PY=/root/autodl-tmp/miniconda3/envs/holography/bin/python
$PY main.py --train-mode --train-stage stage2 \
  --model-arch unet --unet-depth 2 --unet-base-filters 24 --unet-tail-blocks 16 \
  --dataset-res 384 --activate-ddpm --restore-stage1 --restore-stage2 \
  --stage1-ckpt model/stage1_unet_d2t16/stage1_latest.pth \
  --stage2-ckpt-dir model/stage2_d2t16 \
  --stage2-epochs 0 --joint-epochs 300 --train-depth-shift 12.0 \
  --weight-holo-joint 0 --weight-ssim 10 --num-iter-per-test 500 --ddpm-arch complex \
  --batch 2 --learning-rate 1e-4 \
  2>&1 | tee stage2_d2t16_joint_ssim.log
