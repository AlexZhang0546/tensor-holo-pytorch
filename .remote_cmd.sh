cd /root/autodl-tmp/ZhangRuixuan/tensor-holo
echo "===== ls ====="
ls -la
echo "===== eval_official_s2.py ====="
cat eval_official_s2.py
echo "===== model dirs ====="
find model -maxdepth 1 -type d | sort
echo "===== data dirs ====="
find data -maxdepth 1 -type d | sort
