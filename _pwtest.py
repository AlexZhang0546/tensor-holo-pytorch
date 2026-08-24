import paramiko

cands = [
    "TZ66W6ssn/q",
    "/TZ66W6ssn/q",
    "TZ66W6ssn",
    "TZ66W6ssn/q/",
    "TZ66W6ssn/q ",
]
for pw in cands:
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect("connect.cqa1.seetacloud.com", port=17892, username="root",
                  password=pw, allow_agent=False, look_for_keys=False,
                  timeout=15, banner_timeout=15, auth_timeout=15)
        print("SUCCESS", repr(pw))
        c.close()
        break
    except Exception as e:
        print("FAIL", repr(pw), type(e).__name__, str(e)[:120])
