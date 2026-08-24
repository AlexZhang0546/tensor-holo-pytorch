cd /root/autodl-tmp/ZhangRuixuan/tensor-holo
echo "===== tf_propagator / _propagate ====="
grep -n "def tf_propagator\|def _propagate\|double_pad\|tf_compl_exp\|tf_compl_val\|tf_fft2d\|tf_ifft2d" -A 55 optics.py
