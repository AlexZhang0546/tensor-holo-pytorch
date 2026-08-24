cd /root/autodl-tmp/ZhangRuixuan/tensor-holo
grep -n "def _build_graph\|data_format\|tf.nn.conv2d\|tf.compat.v1.layers.conv2d\|depth_to_space\|transpose" -A 70 main_v2.py | head -n 220
