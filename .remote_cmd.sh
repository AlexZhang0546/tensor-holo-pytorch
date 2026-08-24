echo "===== screen -ls ====="
screen -ls
echo "===== holography python ====="
/root/autodl-tmp/miniconda3/envs/holography/bin/python -V
/root/autodl-tmp/miniconda3/envs/holography/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "===== zrxenv python ====="
/root/autodl-tmp/miniconda3/envs/zrxenv/bin/python -V
echo "===== zrxenv_tf2 python ====="
/root/autodl-tmp/miniconda3/envs/zrxenv_tf2/bin/python -V
