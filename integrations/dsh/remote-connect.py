#!/usr/bin/env python3
"""Maintain the ModelScope-to-relay reverse SSH tunnel.

This program is intended to run *on the ModelScope instance*.  It only
forwards the loopback model API.  It deliberately does not start, stop, or
inspect the model server or the benchmark process.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STATE_DIR = Path("/mnt/workspace/.dsv41-connection")
STOP = False
CHILD: subprocess.Popen[bytes] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log_line(path: Path, message: str) -> None:
    # This logger only records lifecycle messages.  It never records command
    # arguments, environment variables, or key material.
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()} {message}\n")


def write_state(path: Path, **values: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def signal_stop(signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def stop_child(process: subprocess.Popen[bytes] | None, timeout: float = 15) -> None:
    """Stop only the ssh process started by this program."""
    if process is None or process.poll() is not None:
        return
    try:
        # ssh is started in a separate process group so an SSH helper cannot
        # survive a clean shutdown of this supervisor.
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    result.add_argument("--identity-file", type=Path)
    result.add_argument("--known-hosts", type=Path)
    result.add_argument(
        "--relay-host",
        default=os.environ.get("DSV41_RELAY_HOST"),
        help="relay host; may also be supplied via DSV41_RELAY_HOST",
    )
    result.add_argument("--relay-user", default="dsv41-tunnel")
    result.add_argument("--relay-port", type=int, default=49241)
    result.add_argument("--backend-host", default="127.0.0.1")
    result.add_argument("--backend-port", type=int, default=48241,
                        help="Model API port on this instance (use 48242 for a future adapter).")
    result.add_argument("--ssh", default="ssh")
    return result


def main() -> int:
    global CHILD
    args = parser().parse_args()
    if not args.relay_host:
        print("remote-connect: --relay-host or DSV41_RELAY_HOST is required", file=sys.stderr)
        return 2
    state_dir: Path = args.state_dir.expanduser().resolve()
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    state_dir.chmod(0o700)
    identity = (args.identity_file or state_dir / "dmit_ed25519").expanduser().resolve()
    known_hosts = (args.known_hosts or state_dir / "known_hosts").expanduser().resolve()
    for required, label in ((identity, "identity file"), (known_hosts, "known_hosts file")):
        if not required.is_file():
            print(f"remote-connect: required {label} is missing: {required}", file=sys.stderr)
            return 2

    lock_path = state_dir / "remote-connect.lock"
    log_path = state_dir / "remote-connect.log"
    status_path = state_dir / "remote-connect.json"
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("remote-connect: another supervisor already owns this tunnel", file=sys.stderr)
        return 3

    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)
    started_at = utc_now()
    backoff = 1
    try:
        while not STOP:
            destination = f"{args.relay_user}@{args.relay_host}"
            reverse = f"127.0.0.1:{args.relay_port}:{args.backend_host}:{args.backend_port}"
            command = [
                args.ssh, "-N", "-T", "-i", str(identity),
                "-o", "BatchMode=yes",
                "-o", "IdentitiesOnly=yes",
                "-o", "StrictHostKeyChecking=yes",
                "-o", f"UserKnownHostsFile={known_hosts}",
                "-o", "ExitOnForwardFailure=yes",
                "-o", "ConnectTimeout=10",
                "-o", "ServerAliveInterval=15",
                "-o", "ServerAliveCountMax=3",
                "-o", "LogLevel=ERROR",
                "-R", reverse, destination,
            ]
            with log_path.open("ab", buffering=0) as logfile:
                try:
                    CHILD = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=logfile,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except OSError as error:
                    log_line(log_path, f"ssh could not start: {error.__class__.__name__}")
                    CHILD = None
                else:
                    write_state(
                        status_path,
                        pid=os.getpid(),
                        start_time=started_at,
                        status="connected-or-connecting",
                        ssh_pid=CHILD.pid,
                        ssh_start_time=utc_now(),
                        relay=f"{args.relay_host}:127.0.0.1:{args.relay_port}",
                        backend=f"{args.backend_host}:{args.backend_port}",
                    )
                    while CHILD.poll() is None and not STOP:
                        time.sleep(0.25)
                    if STOP:
                        stop_child(CHILD)
                    exit_code = CHILD.poll()
                    log_line(log_path, f"ssh exited with status {exit_code}")
                    CHILD = None

            if STOP:
                break
            write_state(
                status_path,
                pid=os.getpid(), start_time=started_at, status="reconnecting",
                retry_seconds=backoff,
                relay=f"{args.relay_host}:127.0.0.1:{args.relay_port}",
                backend=f"{args.backend_host}:{args.backend_port}",
            )
            # Interruptible exponential backoff; the cap avoids retry storms.
            deadline = time.monotonic() + backoff
            while not STOP and time.monotonic() < deadline:
                time.sleep(0.2)
            backoff = min(backoff * 2, 60)
    finally:
        stop_child(CHILD)
        CHILD = None
        write_state(
            status_path,
            pid=os.getpid(), start_time=started_at, status="stopped",
            stopped_at=utc_now(),
            relay=f"{args.relay_host}:127.0.0.1:{args.relay_port}",
            backend=f"{args.backend_host}:{args.backend_port}",
        )
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
