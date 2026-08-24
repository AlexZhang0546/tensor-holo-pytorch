cd /root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch
echo "===== screen ====="
screen -ls
echo "===== tail _finetune_psnr.log ====="
tail -n 30 _finetune_psnr.log 2>/dev/null || true
echo "===== tail eval ====="
tail -n 20 _finetune_psnr_eval.log 2>/dev/null || true
