#!/bin/bash
cd /root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch
PY=/root/autodl-tmp/miniconda3/envs/holography/bin/python
echo "=== fresh-init overfit (img0, 2000 steps) ==="
$PY src/_overfit_s1.py --img-idx 0 --steps 2000 --batch 2 --lr 1e-4 --weight-ssim 50 2>&1 | tee overfit_s1_fresh.log
echo "=== warm-start overfit (d2t16 ckpt, frozen BN, 500 steps) ==="
$PY src/_overfit_s1.py --img-idx 0 --steps 500 --batch 2 --lr 3e-4 --weight-ssim 50 \
  --ckpt model/stage1_unet_d2t16/stage1_latest.pth --freeze-bn 2>&1 | tee overfit_s1_warm.log
echo "ALL_DONE"