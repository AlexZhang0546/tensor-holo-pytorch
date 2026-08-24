import paramiko

HOST = "connect.cqa1.seetacloud.com"
PORT = 17892
USER = "root"
PASSWORD = "/TZ66W6ssn/q"

BASE = "/root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch"

script = r"""
set -e
cd BASE || exit 1
pwd

rm -f 1.cpp 1.exe merged.txt newcode.txt paper_text.txt paper_v2.txt paper_v2.pdf \
      pre_dpm_official.py official_ddpm.npz official_ddpm_net.pth \
      official_ddpm_probe.npz test_384_v2.zip

find . -maxdepth 1 -type f -name '*.log*' -delete
find . -maxdepth 1 -type f -name '_*.py' -delete
find . -maxdepth 1 -type f -name '_*.sh' ! -name '_run_stage2_real_psnrfix.sh' -delete
find . -maxdepth 1 -type d -name '__pycache__' -exec rm -rf {} +
find . -maxdepth 1 -type d \( -name 'tmp_ckpt_stage1_384' -o -name 'tmp_ckpt_stage2' \) -exec rm -rf {} +
find . -maxdepth 1 -type d \( -name 'output' -o -name 'output1' -o -name 'output2' -o -name 'output_cli_test' \) -exec rm -rf {} +

find src -type f -name '_*.py' ! -name '_eval_paper.py' -delete
find src -type f -name '_*.sh' -delete
find src -type d -name '__pycache__' -exec rm -rf {} +

rm -rf tensor-holo

echo "---- remaining top level ----"
ls -la
echo "---- remaining src files ----"
find src -maxdepth 2 -type f | sort
""".replace("BASE", BASE)

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
_, out, err = client.exec_command("bash -s", timeout=120)
out.channel.send(script)
out.channel.shutdown_write()
print(out.read().decode("utf-8", "replace"))
print(err.read().decode("utf-8", "replace"))
client.close()
