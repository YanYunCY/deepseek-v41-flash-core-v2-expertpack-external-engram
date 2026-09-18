#!/usr/bin/env python3
"""Safely stop the existing GPQA runner after a complete result prefix.

The controller never changes run_gpqa.py, its configuration, its dataset, or a
result.  Once it observes complete atomically-written results 000..029, it
verifies the original runner's live /proc identity and sends one SIGINT only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from stage_report import read_json, validate_result

try:  # Linux is required when the controller is actually run.
    import fcntl
except ModuleNotFoundError:  # Allows the read-only unit tests to run on Windows.
    fcntl = None  # type: ignore[assignment]


class IdentityError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def proc_stat(pid: int, proc_root: Path = Path("/proc")) -> tuple[str, int]:
    """Return Linux process state and starttime (field 22), robust to spaces in comm."""
    raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    closing = raw.rfind(")")
    if closing < 0:
        raise IdentityError(f"cannot parse /proc/{pid}/stat")
    fields_after_comm = raw[closing + 2 :].split()
    if len(fields_after_comm) <= 19:
        raise IdentityError(f"short /proc/{pid}/stat")
    return fields_after_comm[0], int(fields_after_comm[19])


def command_line(pid: int, proc_root: Path = Path("/proc")) -> bytes:
    return (proc_root / str(pid) / "cmdline").read_bytes()


def load_runner_spec(benchmark_dir: Path) -> dict[str, Any]:
    process_path = benchmark_dir / "process.json"
    try:
        document = read_json(process_path)
    except (OSError, json.JSONDecodeError) as error:
        raise IdentityError(f"cannot read {process_path}: {error}") from error
    if not isinstance(document, dict):
        raise IdentityError("process.json is not an object")
    pid = document.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        raise IdentityError("process.json has no positive integer pid")
    if document.get("mode") != "full198":
        raise IdentityError(f"process.json mode is {document.get('mode')!r}, expected 'full198'")
    argv = document.get("argv")
    if argv is not None and not isinstance(argv, list):
        raise IdentityError("process.json argv is not a list")
    return {"pid": pid, "document": document}


def live_identity(benchmark_dir: Path, spec: dict[str, Any], proc_root: Path = Path("/proc")) -> dict[str, Any]:
    """Prove the PID is the intended live runner before it may receive SIGINT."""
    pid = spec["pid"]
    try:
        state, starttime = proc_stat(pid, proc_root)
        cmdline = command_line(pid, proc_root)
    except (OSError, ValueError, IdentityError) as error:
        raise IdentityError(f"runner PID {pid} is unavailable: {error}") from error
    if state == "Z":
        raise IdentityError(f"runner PID {pid} is already a zombie")
    expected_output = os.fsencode(str((benchmark_dir / "results").resolve()))
    if b"run_gpqa.py" not in cmdline:
        raise IdentityError("live command line does not contain run_gpqa.py")
    if expected_output not in cmdline:
        raise IdentityError(f"live command line does not contain expected output {expected_output.decode()}")
    argv = spec["document"].get("argv")
    if isinstance(argv, list):
        recorded = "\0".join(str(value) for value in argv).encode("utf-8", "surrogateescape")
        if b"run_gpqa.py" not in recorded or expected_output not in recorded:
            raise IdentityError("process.json argv does not identify the expected runner/output")
    return {
        "pid": pid,
        "starttime": starttime,
        "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
    }


def has_same_starttime_or_exited(pid: int, starttime: int, proc_root: Path = Path("/proc")) -> str:
    """Return ``live``, ``exited``, or ``replaced`` without trusting PID reuse."""
    try:
        state, actual_starttime = proc_stat(pid, proc_root)
    except (OSError, ValueError, IdentityError):
        return "exited"
    if state == "Z":
        return "exited"
    return "live" if actual_starttime == starttime else "replaced"


def results_complete(benchmark_dir: Path, target: int) -> tuple[bool, str]:
    """Require every result in the desired prefix, in order, to be fully valid."""
    try:
        dataset = read_json(benchmark_dir / "dataset.json")
    except (OSError, json.JSONDecodeError) as error:
        return False, f"cannot read dataset.json: {error}"
    if not isinstance(dataset, list) or len(dataset) < target:
        return False, f"dataset has fewer than {target} items"
    for index in range(target):
        item = dataset[index]
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            return False, f"dataset {index:03d} has no string id"
        result_path = benchmark_dir / "results" / "cases" / f"{index:03d}" / "result.json"
        if not result_path.is_file():
            return False, f"waiting for {index:03d}/result.json"
        try:
            result = read_json(result_path)
        except (OSError, json.JSONDecodeError) as error:
            return False, f"waiting for stable {index:03d}/result.json ({error})"
        valid, reason = validate_result(result, index, item["id"])
        if not valid:
            return False, f"waiting for valid {index:03d}/result.json ({reason})"
    return True, f"complete 000..{target - 1:03d}"


def update_control(path: Path, control: dict[str, Any], **changes: Any) -> None:
    control.update(changes)
    control["updated_at"] = utc_now()
    atomic_write_json(path, control)


def acquire_lock(path: Path):
    if fcntl is None:
        raise RuntimeError("stop_after.py requires Linux fcntl locking")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another stop controller holds {path}")
    return handle


def finish_report_and_upload(
    benchmark_dir: Path,
    target: int,
    control_path: Path,
    control: dict[str, Any],
    upload: bool,
    upload_timeout: float,
) -> int:
    report = benchmark_dir / "stage_report.py"
    upload_script = benchmark_dir / "checkpoint_upload.py"
    log_path = benchmark_dir / "stop-after-upload.log"
    complete, detail = results_complete(benchmark_dir, target)
    if not complete:
        update_control(
            control_path,
            control,
            status="report_refused_incomplete_target",
            error=f"not writing a completed stage report: {detail}",
        )
        return 1
    try:
        report_run = subprocess.run(
            [sys.executable, str(report), "--benchmark-dir", str(benchmark_dir), "--target", str(target)],
            cwd=benchmark_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired as error:
        update_control(
            control_path,
            control,
            status="report_timeout",
            report={
                "timeout_seconds": 120,
                "stdout": (error.stdout or "")[-8000:],
                "stderr": (error.stderr or "")[-8000:],
            },
        )
        return 1
    report_record = {
        "returncode": report_run.returncode,
        "stdout": report_run.stdout[-8000:],
        "stderr": report_run.stderr[-8000:],
    }
    if report_run.returncode:
        update_control(control_path, control, status="report_failed", report=report_record)
        return 1
    try:
        summary = read_json(benchmark_dir / "stage-summary.json")
    except (OSError, json.JSONDecodeError) as error:
        update_control(control_path, control, status="report_invalid", error=f"cannot read generated stage-summary.json: {error}")
        return 1
    if not isinstance(summary, dict) or summary.get("target_complete") is not True or summary.get("completed_contiguous_prefix") != target:
        update_control(
            control_path,
            control,
            status="report_refused_incomplete_target",
            error="generated stage summary is not a complete target prefix",
        )
        return 1
    update_control(control_path, control, status="report_written", report=report_record)
    if not upload:
        update_control(control_path, control, status="complete_without_upload", upload={"skipped": True})
        return 0
    try:
        upload_run = subprocess.run(
            [sys.executable, str(upload_script)],
            cwd=benchmark_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=upload_timeout,
        )
    except subprocess.TimeoutExpired as error:
        log_path.write_text(
            f"[{utc_now()}] checkpoint_upload.py timed out after {upload_timeout:g} seconds\n"
            "--- stdout ---\n"
            f"{error.stdout or ''}\n"
            "--- stderr ---\n"
            f"{error.stderr or ''}\n",
            encoding="utf-8",
        )
        update_control(
            control_path,
            control,
            status="upload_timeout",
            upload={"timeout_seconds": upload_timeout, "log": str(log_path)},
        )
        return 1
    log_path.write_text(
        f"[{utc_now()}] checkpoint_upload.py exit={upload_run.returncode}\n"
        "--- stdout ---\n"
        f"{upload_run.stdout}\n"
        "--- stderr ---\n"
        f"{upload_run.stderr}\n",
        encoding="utf-8",
    )
    upload_record = {
        "returncode": upload_run.returncode,
        "log": str(log_path),
    }
    if upload_run.returncode:
        update_control(control_path, control, status="upload_failed", upload=upload_record)
        return 1
    update_control(control_path, control, status="complete", upload=upload_record)
    return 0


def event_waiter(benchmark_dir: Path):
    """Use the existing inotify helper; retain a safe timeout only for process checks."""
    try:
        from viewer_events import LogEvents

        return LogEvents(benchmark_dir / "results")
    except Exception as error:  # Controller safety does not depend on the viewer helper.
        print(f"warning: inotify watcher unavailable ({error}); using 10 second checks", file=sys.stderr)
        return None


def run_controller(
    benchmark_dir: Path,
    target: int,
    *,
    upload: bool = True,
    exit_timeout: float = 300,
    upload_timeout: float = 900,
    proc_root: Path = Path("/proc"),
    send_signal: Callable[[int, int], None] = os.kill,
) -> int:
    """Run the controller.  ``send_signal`` and ``proc_root`` make safety testable."""
    benchmark_dir = benchmark_dir.resolve()
    control_path = benchmark_dir / "control.json"
    lock = acquire_lock(benchmark_dir / "control.lock")
    watcher = None
    try:
        spec = load_runner_spec(benchmark_dir)
        existing: dict[str, Any] | None = None
        if control_path.exists():
            try:
                loaded = read_json(control_path)
                existing = loaded if isinstance(loaded, dict) else None
            except (OSError, json.JSONDecodeError):
                existing = None
        if existing and existing.get("target") not in (None, target):
            raise RuntimeError(f"existing control target is {existing.get('target')}, requested {target}")

        control = existing or {
            "schema_version": 1,
            "created_at": utc_now(),
            "target": target,
            "signal_sent": False,
        }
        if control.get("signal_sent"):
            prior_identity = control.get("runner")
            if not isinstance(prior_identity, dict) or not isinstance(prior_identity.get("pid"), int) or not isinstance(prior_identity.get("starttime"), int):
                raise IdentityError("existing signal_sent control has no usable original runner identity")
            post_signal_state = has_same_starttime_or_exited(
                prior_identity["pid"], prior_identity["starttime"], proc_root
            )
            if post_signal_state == "exited":
                update_control(control_path, control, status="stopped", stopped_at=utc_now())
                return finish_report_and_upload(benchmark_dir, target, control_path, control, upload, upload_timeout)
            if post_signal_state == "replaced":
                update_control(
                    control_path,
                    control,
                    status="post_signal_identity_changed",
                    error="PID now has a different process starttime; stop verification is unsafe",
                )
                return 1
            # The original runner is still alive.  Verify its full command line but never signal twice.
            identity = live_identity(benchmark_dir, spec, proc_root)
            if identity["starttime"] != prior_identity["starttime"]:
                update_control(control_path, control, status="post_signal_identity_changed")
                return 1
            update_control(control_path, control, status="signal_already_sent_waiting_for_exit")
            return 0

        identity = live_identity(benchmark_dir, spec, proc_root)
        prior_identity = control.get("runner")
        if isinstance(prior_identity, dict) and prior_identity.get("starttime") not in (None, identity["starttime"]):
            update_control(
                control_path,
                control,
                status="identity_mismatch",
                error="process starttime changed since this controller first verified it; no signal sent",
            )
            return 1
        control["runner"] = identity
        update_control(control_path, control, status="watching")

        watcher = event_waiter(benchmark_dir)
        while True:
            complete, detail = results_complete(benchmark_dir, target)
            update_control(control_path, control, status="watching", last_check=detail)
            if complete:
                # Repeat every identity check immediately before the only signal.
                current = live_identity(benchmark_dir, spec, proc_root)
                if current["starttime"] != identity["starttime"]:
                    raise IdentityError("runner starttime changed before signal; no signal sent")
                update_control(
                    control_path,
                    control,
                    status="signal_issued",
                    signal_sent=True,
                    signal="SIGINT",
                    signal_sent_at=utc_now(),
                    completion=detail,
                )
                try:
                    send_signal(spec["pid"], signal.SIGINT)
                except OSError as error:
                    update_control(control_path, control, status="signal_failed", error=str(error))
                    return 1
                # The runner catches SIGINT and writes its stopped summary before exiting.
                stop_deadline = time.monotonic() + exit_timeout
                while True:
                    post_signal_state = has_same_starttime_or_exited(spec["pid"], identity["starttime"], proc_root)
                    if post_signal_state == "exited":
                        update_control(control_path, control, status="stopped", stopped_at=utc_now())
                        return finish_report_and_upload(benchmark_dir, target, control_path, control, upload, upload_timeout)
                    if post_signal_state == "replaced":
                        update_control(
                            control_path,
                            control,
                            status="post_signal_identity_changed",
                            error="PID was reused before stop confirmation; not claiming successful stop",
                        )
                        return 1
                    if time.monotonic() >= stop_deadline:
                        update_control(
                            control_path,
                            control,
                            status="stop_wait_timeout",
                            error=f"runner did not exit within {exit_timeout:g} seconds after SIGINT; no further signal sent",
                        )
                        return 1
                    time.sleep(1)
            # A runner that dies early must not leave an apparently active controller behind.
            runner_state = has_same_starttime_or_exited(spec["pid"], identity["starttime"], proc_root)
            if runner_state == "exited":
                update_control(
                    control_path,
                    control,
                    status="runner_exited_before_target",
                    error=f"runner exited while {detail}",
                )
                return 1
            if runner_state == "replaced":
                update_control(
                    control_path,
                    control,
                    status="identity_mismatch",
                    error="runner PID starttime changed; no signal sent",
                )
                return 1
            if watcher is None:
                time.sleep(10)
            else:
                watcher.wait(timeout=10)
    except (IdentityError, RuntimeError) as error:
        if "control" in locals():
            update_control(control_path, control, status="identity_or_control_error", error=str(error))
        else:
            atomic_write_json(control_path, {"schema_version": 1, "target": target, "status": "identity_or_control_error", "error": str(error), "updated_at": utc_now()})
        print(f"stop controller refused to act: {error}", file=sys.stderr)
        return 1
    finally:
        if watcher is not None:
            watcher.close()
        lock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--target", type=int, default=30)
    parser.add_argument("--exit-timeout", type=float, default=300, help="seconds to wait after SIGINT; no further signal is sent")
    parser.add_argument("--upload-timeout", type=float, default=900, help="seconds to allow checkpoint_upload.py")
    parser.add_argument("--no-upload", action="store_true", help="write the report but do not call checkpoint_upload.py")
    args = parser.parse_args()
    if args.target < 1 or args.exit_timeout <= 0 or args.upload_timeout <= 0:
        parser.error("--target and both timeout values must be positive")
    return run_controller(
        args.benchmark_dir,
        args.target,
        upload=not args.no_upload,
        exit_timeout=args.exit_timeout,
        upload_timeout=args.upload_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
