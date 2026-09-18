#!/usr/bin/env python3
"""Start the loopback DSH adapter and its existing, restricted SSH tunnel."""
import fcntl
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = Path(os.environ.get('DSV41_CONNECTION_STATE', '/root/dsv41/dsh-integration/state'))
KEYS = Path('/mnt/workspace/.dsv41-connection')
RELAY_HOST = os.environ.get('DSV41_RELAY_HOST')


def healthy():
    try:
        with urllib.request.urlopen('http://127.0.0.1:48242/health', timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def launch(arguments, logfile):
    with (STATE / logfile).open('ab') as log:
        return subprocess.Popen(arguments, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=log, start_new_session=True)


def main():
    if not RELAY_HOST:
        raise RuntimeError('DSV41_RELAY_HOST must identify the relay; no host is hardcoded in this public template')
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (STATE / 'startup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name in ('dmit_ed25519', 'known_hosts'):
            if not (KEYS / name).is_file():
                raise RuntimeError(f'Missing existing connection credential: {KEYS / name}')
        if not healthy():
            candidates = [ROOT / 'encoding.py', Path('/root/dsv41/capability/encoding.py')]
            encoder = next((p for p in candidates if p.is_file()), None)
            if encoder is None:
                raise RuntimeError('Official V4.1 encoding.py is missing')
            child = launch([sys.executable, str(ROOT / 'dsh_api_adapter.py'), '--encoder', str(encoder)], 'adapter.log')
            for _ in range(20):
                if healthy():
                    break
                if child.poll() is not None:
                    raise RuntimeError('Adapter failed; inspect state/adapter.log')
                time.sleep(0.5)
            else:
                raise RuntimeError('Model or adapter is not ready; run this command after model startup')
        # remote-connect owns the authoritative flock; a duplicate invocation
        # exits without replacing or killing a live tunnel.
        statefile = STATE / 'remote-connect.json'
        live = False
        if statefile.exists():
            try:
                state = json.loads(statefile.read_text())
                cmd = Path(f"/proc/{int(state['pid'])}/cmdline").read_bytes()
                live = b'remote-connect.py' in cmd and b'48242' in cmd
            except (ValueError, KeyError, OSError):
                pass
        if not live:
            child = launch([sys.executable, str(ROOT / 'remote-connect.py'),
                            '--state-dir', str(STATE), '--identity-file', str(KEYS / 'dmit_ed25519'),
                            '--known-hosts', str(KEYS / 'known_hosts'), '--relay-host', RELAY_HOST,
                            '--backend-port', '48242'],
                           'remote-connect.launch.log')
            time.sleep(0.5)
            if child.poll() is not None:
                raise RuntimeError('Tunnel supervisor failed; inspect state/remote-connect.launch.log')
        print('DSH_ADAPTER_READY: Windows endpoint http://127.0.0.1:48241/v1')
        print('Remote tunnel reconnects automatically while this instance is running.')


if __name__ == '__main__':
    main()
