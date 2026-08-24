#!/bin/bash
set -e
cd /root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch
PY=/root/autodl-tmp/miniconda3/envs/holography/bin/python

NEWDIR=model/stage2_real_d2t16_psnrfix
mkdir -p "$NEWDIR"
cp model/stage2_real_d2t16/stage2_joint_latest.pth "$NEWDIR/stage2_joint_latest.pth"

echo "===== fine-tune (weight_ssim=3, lr=3e-5, 2 epochs) ====="
$PY main.py --train-mode --train-stage stage2 \
  --model-arch unet --unet-depth 2 --unet-base-filters 24 --unet-tail-blocks 16 \
  --dataset-res 384 --activate-ddpm --restore-stage1 --restore-stage2 \
  --stage1-ckpt model/stage1_unet_d2t16/stage1_latest.pth \
  --stage2-ckpt-dir "$NEWDIR" \
  --stage2-epochs 0 --joint-epochs 122 --train-depth-shift 12.0 \
  --weight-holo-joint 0 --weight-ssim 3 --num-iter-per-test 500 \
  --ddpm-arch real --ddpm-bn tf \
  --batch 2 --learning-rate 3e-5 \
  2>&1 | tee _finetune_psnr.log

echo "===== eval fine-tuned ckpt ====="
$PY src/_eval_paper.py \
  --ckpt-path "$NEWDIR/stage2_joint_latest.pth" \
  --model-arch unet --unet-depth 2 --unet-base-filters 24 --unet-tail-blocks 16 \
  --ddpm-arch real --ddpm-bn tf --depth-shift 12.0 --dataset-res 384 --batch 2 \
  --split validate --stage2 2>&1 | tee _finetune_psnr_eval.log

echo "===== DONE ====="
