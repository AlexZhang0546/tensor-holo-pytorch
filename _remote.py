import os
import sys
import paramiko

HOST = "connect.cqa1.seetacloud.com"
PORT = 17892
USER = "root"
PASSWORD = "/TZ66W6ssn/q"


def run(cmd, timeout=120):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                   timeout=30, banner_timeout=30, auth_timeout=30)
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout,
                                                get_pty=True)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    client.close()
    sys.stdout.write(out)
    if err:
        sys.stdout.write("\n--- STDERR ---\n" + err)
    sys.stdout.write("\n--- EXIT %d ---\n" % code)


def sftp_put(local, remote):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                   timeout=30, banner_timeout=30, auth_timeout=30)
    sftp = client.open_sftp()
    sftp.put(local, remote)
    sftp.close()
    client.close()
    print("uploaded %s -> %s" % (local, remote))


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--upload":
        sftp_put(sys.argv[2], sys.argv[3])
        sys.exit(0)
    if len(sys.argv) >= 3 and sys.argv[1] == "--file":
        with open(sys.argv[2], "r", encoding="utf-8") as f:
            cmd = f.read()
    else:
        cmd = " ".join(sys.argv[1:])
    run(cmd)
