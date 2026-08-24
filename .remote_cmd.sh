cd /root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch
/root/autodl-tmp/miniconda3/envs/holography/bin/python src/_eval_paper.py \
  --ckpt-path model/stage2_real_d2t16/stage2_joint_latest.pth \
  --model-arch unet --unet-depth 2 --unet-base-filters 24 --unet-tail-blocks 16 \
  --ddpm-arch real --ddpm-bn tf --depth-shift 12.0 --dataset-res 384 --batch 2 \
  --split validate --stage2
