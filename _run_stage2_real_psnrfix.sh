#!/bin/bash
# Balanced stage-2 fine-tune that fixes the PSNR regression caused by the
# oversized --weight-ssim 30 in the previous run.
#
# Starts from model/stage2_real_d2t16/stage2_joint_latest.pth and fine-tunes
# a few epochs with a much smaller SSIM weight, so the L1 focal-stack loss
# (which drives PSNR) is no longer overwhelmed. Result checkpoint is written
# to model/stage2_real_d2t16_psnrfix/.
set -e
cd /root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch
PY=/root/autodl-tmp/miniconda3/envs/holography/bin/python

NEWDIR=model/stage2_real_d2t16_psnrfix
mkdir -p "$NEWDIR"
cp model/stage2_real_d2t16/stage2_joint_latest.pth "$NEWDIR/stage2_joint_latest.pth"

$PY main.py --train-mode --train-stage stage2 \
  --model-arch unet --unet-depth 2 --unet-base-filters 24 --unet-tail-blocks 16 \
  --dataset-res 384 --activate-ddpm --restore-stage1 --restore-stage2 \
  --stage1-ckpt model/stage1_unet_d2t16/stage1_latest.pth \
  --stage2-ckpt-dir "$NEWDIR" \
  --stage2-epochs 0 --joint-epochs 122 --train-depth-shift 12.0 \
  --weight-holo-joint 0 --weight-ssim 3 --num-iter-per-test 500 \
  --ddpm-arch real --ddpm-bn tf \
  --batch 2 --learning-rate 3e-5 \
  2>&1 | tee stage2_real_d2t16_psnrfix.log
