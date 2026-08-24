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


if __name__ == "__main__":
    run(sys.argv[1], timeout=int(sys.argv[2]) if len(sys.argv) > 2 else 120)
