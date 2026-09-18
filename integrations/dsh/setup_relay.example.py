"""Provision a loopback-only forwarding account on a relay host.

This is a template. Supply the user, public key, and expected host identity
through environment variables; never commit those values to the repository.
Run it only after reviewing the relay's sshd policy.
"""

import os
import pwd
import subprocess
from pathlib import Path


USER = os.environ.get("DSV41_RELAY_USER", "dsv41-tunnel")
PUBLIC_KEY = os.environ.get("DSV41_PUBLIC_KEY")
RELAY_HOST = os.environ.get("DSV41_RELAY_HOST")

if not PUBLIC_KEY or not RELAY_HOST:
    raise SystemExit("Set DSV41_PUBLIC_KEY and DSV41_RELAY_HOST before provisioning")
if not PUBLIC_KEY.startswith("ssh-"):
    raise SystemExit("DSV41_PUBLIC_KEY must be an OpenSSH public key")

try:
    account = pwd.getpwnam(USER)
except KeyError:
    subprocess.run(
        ["useradd", "--create-home", "--user-group", "--shell", "/usr/sbin/nologin", USER],
        check=True,
    )
    account = pwd.getpwnam(USER)

config = subprocess.check_output(
    ["/usr/sbin/sshd", "-T", "-C", f"user={USER},host=localhost,addr=127.0.0.1"],
    text=True,
)
settings = dict(line.split(" ", 1) for line in config.splitlines() if " " in line)
if settings.get("gatewayports") == "yes":
    raise SystemExit("Refusing a relay that enables public GatewayPorts")
if settings.get("allowtcpforwarding") not in ("yes", "all", "remote"):
    raise SystemExit("Remote TCP forwarding is disabled by sshd")

directory = Path(account.pw_dir) / ".ssh"
directory.mkdir(mode=0o700, exist_ok=True)
os.chown(directory, account.pw_uid, account.pw_gid)
directory.chmod(0o700)
target = directory / "authorized_keys"
key_material = PUBLIC_KEY.split()[1]
line = (
    'restrict,port-forwarding,permitlisten="127.0.0.1:49241",'
    'permitopen="127.0.0.1:1",command="/usr/bin/false" '
    + PUBLIC_KEY
)
existing = target.read_text() if target.exists() else ""
if key_material not in existing:
    target.write_text(existing.rstrip() + ("\n" if existing.strip() else "") + line + "\n")
os.chown(target, account.pw_uid, account.pw_gid)
target.chmod(0o600)

host_key = Path("/etc/ssh/ssh_host_ed25519_key.pub").read_text().split()
print("RELAY_ACCOUNT_READY", USER, "listen=127.0.0.1:49241", "shell=disabled")
print("Record this host key out of band:", RELAY_HOST, host_key[0], host_key[1])
