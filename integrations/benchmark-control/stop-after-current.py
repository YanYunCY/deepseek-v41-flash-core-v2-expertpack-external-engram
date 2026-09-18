#!/usr/bin/env python3
"""Stop the active GPQA question only after its saved result is verified.

This controller is for the already-running ``run_gpqa.py`` process.  It does
not edit or restart the runner, model server, dataset, configuration, or any
result file.  At installation it freezes the question index recorded by the
runner heartbeat.  It waits for that one result, then sends *one* SIGINT to
the exact process held by a Linux pidfd.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:  # The deployed controller requires Linux. Unit tests run on Windows.
    import fcntl
except ModuleNotFoundError:  # pragma: no cover - exercised by Windows imports
    fcntl = None  # type: ignore[assignment]

from stage_report import read_json, validate_evidence, validate_result


class ControlRefused(RuntimeError):
    """A condition that means the controller must never send a signal."""


TERMINAL_STATUSES = {
    "runner_exited_before_saved_current_result",
    "runner_exited_after_saved_current_result",
    "runner_identity_changed",
    "signal_sent_waiting_for_exit",
    "stopped_after_saved_current_result",
    "stop_wait_timeout_no_second_signal",
    "report_failed",
    "report_written",
    "upload_failed",
    "complete",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def set_status(path: Path, state: dict[str, Any], status: str, **extra: Any) -> None:
    state.update(extra)
    state["status"] = status
    state["updated_at"] = utc_now()
    atomic_write_json(path, state)
    print(f"[{state['updated_at']}] {status}: {state.get('detail', '')}", flush=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def proc_stat(pid: int, proc_root: Path = Path("/proc")) -> tuple[str, int]:
    """Return Linux process state and field-22 starttime, handling spaces in comm."""
    raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    close = raw.rfind(")")
    if close < 0:
        raise ControlRefused(f"cannot parse /proc/{pid}/stat")
    fields = raw[close + 2 :].split()
    if len(fields) <= 19:
        raise ControlRefused(f"short /proc/{pid}/stat")
    return fields[0], int(fields[19])


def runner_spec(root: Path) -> dict[str, Any]:
    try:
        spec = read_json(root / "process.json")
    except (OSError, json.JSONDecodeError) as error:
        raise ControlRefused(f"cannot read process.json: {error}") from error
    if not isinstance(spec, dict):
        raise ControlRefused("process.json is not an object")
    if spec.get("mode") != "full198":
        raise ControlRefused("process.json mode is not 'full198'")
    if type(spec.get("pid")) is not int or spec["pid"] <= 0:
        raise ControlRefused("process.json has no positive integer pid")
    if not isinstance(spec.get("argv"), list):
        raise ControlRefused("process.json has no argv list")
    return spec


def identify_runner(root: Path, pid: int, proc_root: Path = Path("/proc"), expected_argv: list[str] | None = None) -> dict[str, Any]:
    """Verify the live PID is precisely the expected full GPQA runner."""
    try:
        status, starttime = proc_stat(pid, proc_root)
        args = (proc_root / str(pid) / "cmdline").read_bytes().split(b"\0")
    except (OSError, ValueError, ControlRefused) as error:
        raise ControlRefused(f"runner PID {pid} is unavailable: {error}") from error
    if status == "Z":
        raise ControlRefused(f"runner PID {pid} is a zombie")
    args = [arg for arg in args if arg]
    script = os.fsencode(str((root / "run_gpqa.py").resolve()))
    output = os.fsencode(str((root / "results").resolve()))
    if script not in args:
        raise ControlRefused("live command line does not contain this run_gpqa.py")
    if expected_argv is not None:
        if not all(isinstance(value, str) for value in expected_argv):
            raise ControlRefused("process.json argv contains a non-string argument")
        recorded = [os.fsencode(value) for value in expected_argv]
        if script not in recorded or args[args.index(script):] != recorded[recorded.index(script):]:
            raise ControlRefused("live runner script and arguments differ from process.json argv")
    try:
        output_index = args.index(b"--output")
        actual_output = args[output_index + 1]
    except (ValueError, IndexError) as error:
        raise ControlRefused("live command line has no --output value") from error
    if actual_output != output:
        raise ControlRefused("live runner --output is not this benchmark results directory")
    return {
        "pid": pid,
        "starttime": starttime,
        "cmdline_sha256": hashlib.sha256(b"\0".join(args)).hexdigest(),
    }


def open_verified_pidfd(
    root: Path,
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
    pidfd_open: Callable[[int], int] | None = None,
    expected_argv: list[str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Hold a PID reference before identity checks, closing it on any refusal.

    Holding the pidfd first prevents PID reuse between the identity check and
    the later signal. The caller must use this fd, never ``os.kill(pid, ...)``.
    """
    if pidfd_open is None:
        if not hasattr(os, "pidfd_open"):
            raise ControlRefused("Linux os.pidfd_open is required")
        pidfd_open = os.pidfd_open
    try:
        fd = pidfd_open(pid)
    except OSError as error:
        raise ControlRefused(f"cannot open pidfd for runner PID {pid}: {error}") from error
    try:
        identity = identify_runner(root, pid, proc_root, expected_argv)
    except Exception:
        os.close(fd)
        raise
    return fd, identity


def assert_same_runner(root: Path, identity: dict[str, Any], proc_root: Path) -> None:
    current = identify_runner(root, identity["pid"], proc_root)
    if current["starttime"] != identity["starttime"]:
        raise ControlRefused("runner PID starttime changed")
    if current["cmdline_sha256"] != identity["cmdline_sha256"]:
        raise ControlRefused("runner command line changed")


def attempt_dir(case_dir: Path, attempt: Any) -> Path:
    """Map known runner attempt encodings to their one safe case subdirectory."""
    if type(attempt) is int and attempt >= 1:
        name = f"attempt-{attempt:02d}"
    elif isinstance(attempt, str) and attempt.startswith("attempt-") and attempt[8:].isdigit() and int(attempt[8:]) >= 1:
        name = attempt
    else:
        raise ControlRefused(f"unknown result attempt encoding: {attempt!r}")
    directory = case_dir / name
    if directory.parent != case_dir:
        raise ControlRefused("attempt directory escapes case directory")
    return directory


def read_manifest_and_sources(root: Path) -> dict[str, Any]:
    try:
        manifest = read_json(root / "results" / "manifest.json")
    except (OSError, json.JSONDecodeError) as error:
        raise ControlRefused(f"cannot read results/manifest.json: {error}") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("fingerprints"), dict):
        raise ControlRefused("results/manifest.json has no fingerprints object")
    fingerprints = manifest["fingerprints"]
    # These are the three source fingerprints used by the existing runner and
    # report. Do not assume undocumented manifest keys.
    for key, filename in (("dataset", "dataset.json"), ("config", "config.json"), ("harness", "run_gpqa.py")):
        expected = fingerprints.get(key)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ControlRefused(f"manifest fingerprint {key!r} is missing or invalid")
        if sha256(root / filename) != expected:
            raise ControlRefused(f"{filename} does not match manifest fingerprint")
    return fingerprints


def read_dataset(root: Path) -> list[dict[str, Any]]:
    try:
        dataset = read_json(root / "dataset.json")
    except (OSError, json.JSONDecodeError) as error:
        raise ControlRefused(f"cannot read dataset.json: {error}") from error
    if not isinstance(dataset, list):
        raise ControlRefused("dataset.json is not a list")
    for index, row in enumerate(dataset):
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ControlRefused(f"dataset item {index} has no string id")
    return dataset


def saved_result(
    root: Path,
    dataset: list[dict[str, Any]],
    fingerprints: dict[str, Any],
    index: int,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Check the frozen result and its two saved evidence files only."""
    if type(index) is not int or index < 0 or index >= len(dataset):
        raise ControlRefused(f"invalid frozen question index {index!r}")
    result_path = root / "results" / "cases" / f"{index:03d}" / "result.json"
    if not result_path.is_file():
        return False, f"waiting for case {index:03d} result.json", None
    try:
        result = read_json(result_path)
    except (OSError, json.JSONDecodeError) as error:
        return False, f"waiting for stable case {index:03d} result.json ({error})", None
    valid, reason = validate_result(result, index, dataset[index]["id"])
    if not valid:
        return False, f"waiting for valid case {index:03d} result.json ({reason})", None
    if result.get("fingerprints") != fingerprints:
        raise ControlRefused(f"case {index:03d} fingerprints do not match manifest")
    try:
        # validate_evidence resolves integer 1 as attempt-01 and verifies both
        # output.txt and request.json against result hashes.
        validate_evidence(result_path, result, fingerprints)
    except (OSError, ValueError) as error:
        return False, f"waiting for saved evidence for case {index:03d} ({error})", None
    return True, f"saved and verified case {index:03d}", result


def heartbeat_index(root: Path, dataset_size: int) -> tuple[int, str]:
    try:
        heartbeat = read_json(root / "results" / "heartbeat.json")
    except (OSError, json.JSONDecodeError) as error:
        raise ControlRefused(f"cannot read results/heartbeat.json: {error}") from error
    if not isinstance(heartbeat, dict) or heartbeat.get("state") not in {"starting", "generating"}:
        raise ControlRefused("heartbeat does not identify an active question")
    index = heartbeat.get("index")
    case = heartbeat.get("case")
    if index is None and isinstance(case, str) and len(case) == 3 and case.isdigit():
        index = int(case)
    elif case is not None and (not isinstance(case, str) or case != f"{index:03d}"):
        raise ControlRefused("heartbeat case and index disagree")
    if type(index) is not int or not 0 <= index < dataset_size:
        raise ControlRefused("heartbeat has no valid active question index")
    return index, str(heartbeat["state"])


def acquire_lock(path: Path):
    if fcntl is None:
        raise RuntimeError("this controller requires Linux fcntl locking")
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise ControlRefused(f"another controller holds {path.name}") from error
    return handle


def process_exited(fd: int, timeout_seconds: float) -> bool:
    return bool(select.select([fd], [], [], timeout_seconds)[0])


def generate_report(root: Path, target: int, state_path: Path, state: dict[str, Any], upload: bool) -> int:
    """Report only the frozen inclusive prefix; later interrupted work is excluded."""
    report = root / "stage_report.py"
    try:
        completed = subprocess.run(
            [sys.executable, str(report), "--benchmark-dir", str(root), "--target", str(target)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        set_status(state_path, state, "report_failed", detail="stage_report.py exceeded 120 seconds")
        return 1
    if completed.returncode:
        set_status(
            state_path,
            state,
            "report_failed",
            detail="stage_report.py failed; see saved stdout/stderr in stop-current.json",
            report_stdout=completed.stdout[-8000:],
            report_stderr=completed.stderr[-8000:],
        )
        return 1
    set_status(state_path, state, "report_written", detail=f"wrote a fixed-prefix report for cases 000..{target - 1:03d}")
    if not upload:
        set_status(state_path, state, "complete", detail="report complete; model server was left running", upload_skipped=True)
        return 0
    upload_script = root / "checkpoint_upload.py"
    if not upload_script.is_file():
        set_status(state_path, state, "upload_failed", detail="checkpoint_upload.py is absent; report remains local")
        return 1
    try:
        uploaded = subprocess.run(
            [sys.executable, str(upload_script)], cwd=root, capture_output=True, text=True, timeout=1200, check=False
        )
    except subprocess.TimeoutExpired:
        set_status(state_path, state, "upload_failed", detail="checkpoint_upload.py exceeded 1200 seconds")
        return 1
    if uploaded.returncode:
        set_status(
            state_path,
            state,
            "upload_failed",
            detail="checkpoint_upload.py failed; report remains local",
            upload_stdout=uploaded.stdout[-8000:],
            upload_stderr=uploaded.stderr[-8000:],
        )
        return 1
    set_status(state_path, state, "complete", detail="report and requested checkpoint upload complete")
    return 0


def run(root: Path, upload: bool) -> int:
    root = root.resolve()
    state_path = root / "stop-current.json"
    lock = acquire_lock(root / "stop-current.lock")
    fd: int | None = None
    state: dict[str, Any] = {}
    try:
        # A prior controller may already have signalled the runner, or may have
        # observed its natural exit.  Check that durable fact before touching
        # process.json or /proc, because either can be gone at this point.
        if state_path.exists():
            saved = read_json(state_path)
            if not isinstance(saved, dict):
                raise ControlRefused("existing stop-current.json is not an object")
            state = saved
            if state.get("status") in TERMINAL_STATUSES:
                print(f"controller will not repeat a terminal operation: {state.get('status')}", flush=True)
                return 0
            if state.get("signal_attempted") is True:
                print("controller will not repeat SIGINT: a prior controller already attempted it", flush=True)
                return 0
        if not hasattr(signal, "pidfd_send_signal"):
            raise ControlRefused("Linux signal.pidfd_send_signal is required")
        spec = runner_spec(root)
        fd, identity = open_verified_pidfd(root, spec["pid"], expected_argv=spec["argv"])
        fingerprints = read_manifest_and_sources(root)
        dataset = read_dataset(root)

        if state:
            frozen = state.get("frozen")
            if not isinstance(frozen, dict) or type(frozen.get("index")) is not int:
                raise ControlRefused("existing state has no valid frozen question")
            if state.get("runner") != identity:
                raise ControlRefused("live runner identity differs from frozen state")
            current_index = frozen["index"]
        else:
            current_index, heartbeat_state = heartbeat_index(root, len(dataset))
            state = {
                "schema_version": 1,
                "policy": "stop after the question active when this controller was installed",
                "controller_pid": os.getpid(),
                "started_at": utc_now(),
                "runner": identity,
                "frozen": {"index": current_index, "case": f"{current_index:03d}", "heartbeat_state": heartbeat_state},
                "report_target": current_index + 1,
                "signal_attempted": False,
            }
            set_status(state_path, state, "installed", detail=f"froze active case {current_index:03d}; waiting for its saved result")

        report_target = state.get("report_target")
        if type(report_target) is not int or report_target != current_index + 1:
            raise ControlRefused("existing state report target does not match frozen question")
        if not 0 <= current_index < len(dataset):
            raise ControlRefused("frozen question index is outside dataset")

        while True:
            complete, detail, _ = saved_result(root, dataset, fingerprints, current_index)
            if complete:
                if process_exited(fd, 0):
                    set_status(
                        state_path,
                        state,
                        "runner_exited_after_saved_current_result",
                        detail=f"runner naturally exited after {detail}; no signal was needed",
                    )
                    return generate_report(root, report_target, state_path, state, upload)
                # Repeat /proc identity and probe the held pidfd immediately
                # before the one real signal. The pidfd is the actual target.
                assert_same_runner(root, identity, Path("/proc"))
                try:
                    signal.pidfd_send_signal(fd, 0)
                except OSError as error:
                    raise ControlRefused(f"held runner pidfd is no longer live: {error}") from error
                set_status(
                    state_path,
                    state,
                    "signal_attempted",
                    detail=f"verified {detail}; sending one SIGINT through the held pidfd",
                    signal_attempted=True,
                    signal="SIGINT",
                    signal_attempted_at=utc_now(),
                )
                try:
                    signal.pidfd_send_signal(fd, signal.SIGINT)
                except OSError as error:
                    set_status(state_path, state, "signal_sent_waiting_for_exit", detail=f"SIGINT call returned {error}; no repeat signal")
                    return 1
                if not process_exited(fd, 120):
                    set_status(
                        state_path,
                        state,
                        "stop_wait_timeout_no_second_signal",
                        detail="runner did not exit within 120 seconds after SIGINT; no second signal was sent",
                    )
                    return 1
                set_status(
                    state_path,
                    state,
                    "stopped_after_saved_current_result",
                    detail=f"runner exited after saved case {current_index:03d}; model server was left running",
                )
                return generate_report(root, report_target, state_path, state, upload)
            # The runner can naturally exit between checks; this wait bounds
            # normal observation to 0.25 seconds without rehashing old cases.
            if process_exited(fd, 0.25):
                set_status(
                    state_path,
                    state,
                    "runner_exited_before_saved_current_result",
                    detail=f"runner exited while {detail}; no signal sent",
                )
                return 1
    except (ControlRefused, OSError, json.JSONDecodeError) as error:
        if state:
            set_status(state_path, state, "refused", detail=str(error))
        else:
            atomic_write_json(state_path, {"schema_version": 1, "status": "refused", "detail": str(error), "updated_at": utc_now()})
        print(f"controller refused to act: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        if fd is not None:
            os.close(fd)
        lock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="existing /root/dsv41/benchmark directory")
    parser.add_argument("--upload", action="store_true", help="run existing checkpoint_upload.py after the fixed-prefix report")
    args = parser.parse_args()
    return run(args.root, args.upload)


if __name__ == "__main__":
    raise SystemExit(main())
