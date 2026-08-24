cd /root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch
/root/autodl-tmp/miniconda3/envs/holography/bin/python -c "import torch; ck=torch.load('model/stage2_real_d2t16/stage2_joint_latest.pth', map_location='cpu'); print('epoch', ck.get('epoch'), 'global_step', ck.get('global_step')); print('keys', list(ck.keys()))"
