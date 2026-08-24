cd /root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch/model
for d in */; do
  echo "=== $d"
  ls -lt "$d" | head -n 6
done
