import paramiko

HOST = "connect.cqa1.seetacloud.com"
PORT = 17892
USER = "root"
PASSWORD = "/TZ66W6ssn/q"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
cmd = "cd /root/autodl-tmp/ZhangRuixuan/tensor-holo-pytorch && ls -la"
_, out, err = client.exec_command(cmd)
print(out.read().decode("utf-8", "replace"))
print(err.read().decode("utf-8", "replace"))
client.close()
