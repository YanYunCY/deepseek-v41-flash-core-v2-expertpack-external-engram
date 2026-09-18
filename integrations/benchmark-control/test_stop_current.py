#!/usr/bin/env python3
"""Windows-runnable simulation tests for stop-after-current.py.

They mock Linux pidfd/signal behavior; they are not a substitute for a remote
Linux acceptance run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("stop-after-current.py")
SPEC = importlib.util.spec_from_file_location("stop_after_current", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
stop_current = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stop_current)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_benchmark(parent: Path) -> tuple[Path, list[dict[str, object]], dict[str, str]]:
    root = parent / "benchmark"
    root.mkdir()
    dataset: list[dict[str, object]] = [
        {"id": "one", "domain": "physics"},
        {"id": "two", "domain": "chemistry"},
    ]
    write_json(root / "dataset.json", dataset)
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    (root / "run_gpqa.py").write_text("# fixture\n", encoding="utf-8")
    fingerprints = {
        "dataset": digest(root / "dataset.json"),
        "config": digest(root / "config.json"),
        "harness": digest(root / "run_gpqa.py"),
    }
    write_json(root / "results" / "manifest.json", {"fingerprints": fingerprints})
    write_json(root / "results" / "heartbeat.json", {"state": "generating", "index": 0})
    return root, dataset, fingerprints


def write_complete_result(root: Path, dataset: list[dict[str, object]], fingerprints: dict[str, str], index: int, attempt: int | str = 1) -> None:
    case = root / "results" / "cases" / f"{index:03d}"
    attempt_name = f"attempt-{attempt:02d}" if type(attempt) is int else attempt
    attempt_dir = case / attempt_name
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "output.txt").write_text(f"answer-{index}\n", encoding="utf-8")
    write_json(attempt_dir / "request.json", {"question": dataset[index]["id"]})
    write_json(
        case / "result.json",
        {
            "id": dataset[index]["id"],
            "index": index,
            "fingerprints": fingerprints,
            "seed": 1,
            "grade": {"classification": "correct"},
            "tokens": 12,
            "wall_seconds": 2.0,
            "timings": {"predicted_per_second": 6.0},
            "stop_type": "eos",
            "input_truncated": False,
            "attempt": attempt,
            "output_sha256": digest(attempt_dir / "output.txt"),
            "request_sha256": digest(attempt_dir / "request.json"),
        },
    )


class DummyLock:
    def close(self) -> None:
        pass


class StopCurrentTests(unittest.TestCase):
    def test_real_heartbeat_case_format(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _, _ = make_benchmark(Path(temporary))
            write_json(root / "results" / "heartbeat.json", {"state": "generating", "case": "032"})
            self.assertEqual(stop_current.heartbeat_index(root, 198), (32, "generating"))
            write_json(root / "results" / "heartbeat.json", {"state": "generating", "case": "032", "index": 31})
            with self.assertRaises(stop_current.ControlRefused):
                stop_current.heartbeat_index(root, 198)

    def test_rejects_different_arguments_for_same_script_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, _, _ = make_benchmark(Path(temporary))
            proc_root = Path(temporary) / "proc"
            process = proc_root / "77"
            process.mkdir(parents=True)
            (process / "stat").write_text("77 (python) R " + "0 " * 18 + "12345 0 0\n")
            expected = ["python3", str((root / "run_gpqa.py").resolve()), "--output", str((root / "results").resolve()), "--seed", "1"]
            actual = expected[:-1] + ["2"]
            (process / "cmdline").write_bytes(b"\0".join(os.fsencode(x) for x in actual) + b"\0")
            with self.assertRaisesRegex(stop_current.ControlRefused, "differ from process.json"):
                stop_current.identify_runner(root, 77, proc_root, expected)
            self.assertEqual(stop_current.identify_runner(root, 77, proc_root, actual)["starttime"], 12345)

    def test_integer_attempt_one_maps_to_attempt_01_and_validates_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, dataset, fingerprints = make_benchmark(Path(temporary))
            write_complete_result(root, dataset, fingerprints, 0, attempt=1)
            complete, detail, row = stop_current.saved_result(root, dataset, fingerprints, 0)
            self.assertTrue(complete, detail)
            self.assertEqual(row["attempt"], 1)
            self.assertEqual(stop_current.attempt_dir(root / "results" / "cases" / "000", 1).name, "attempt-01")

    def test_current_result_check_does_not_rehash_or_require_old_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, dataset, fingerprints = make_benchmark(Path(temporary))
            # Case 000 intentionally has no result. The frozen current case
            # can still be accepted; stage_report later owns prefix integrity.
            write_complete_result(root, dataset, fingerprints, 1, attempt="attempt-01")
            complete, detail, _ = stop_current.saved_result(root, dataset, fingerprints, 1)
            self.assertTrue(complete, detail)

    def test_pidfd_is_opened_before_runner_identity_is_read(self) -> None:
        order: list[str] = []
        fake_fd = os.open(os.devnull, os.O_RDONLY)
        identity = {"pid": 77, "starttime": 123, "cmdline_sha256": "a" * 64}
        try:
            with patch.object(stop_current, "identify_runner", side_effect=lambda *args: order.append("identify") or identity):
                fd, got = stop_current.open_verified_pidfd(
                    Path("."), 77, pidfd_open=lambda pid: order.append("pidfd_open") or fake_fd
                )
            self.assertEqual(fd, fake_fd)
            self.assertEqual(got, identity)
            self.assertEqual(order, ["pidfd_open", "identify"])
        finally:
            os.close(fake_fd)

    def test_simulated_signal_uses_pidfd_once_after_saved_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, dataset, fingerprints = make_benchmark(Path(temporary))
            identity = {"pid": 42, "starttime": 9001, "cmdline_sha256": "b" * 64}
            fd = os.open(os.devnull, os.O_RDONLY)
            sent: list[int] = []
            def poll(_fd, timeout):
                if timeout == 0.25:
                    self.assertEqual(sent, [])
                    write_complete_result(root, dataset, fingerprints, 0)
                    return False
                return timeout == 120
            def send(received_fd, signum):
                self.assertEqual(received_fd, fd)
                self.assertTrue(stop_current.saved_result(root, dataset, fingerprints, 0)[0])
                sent.append(signum)
            try:
                with (
                    patch.object(stop_current, "acquire_lock", return_value=DummyLock()),
                    patch.object(stop_current, "runner_spec", return_value={"pid": 42, "mode": "full198", "argv": []}),
                    patch.object(stop_current, "open_verified_pidfd", return_value=(fd, identity)),
                    patch.object(stop_current, "read_manifest_and_sources", return_value=fingerprints),
                    patch.object(stop_current, "read_dataset", return_value=dataset),
                    patch.object(stop_current, "assert_same_runner"),
                    patch.object(stop_current, "process_exited", side_effect=poll),
                    patch.object(stop_current, "generate_report", return_value=0) as report,
                    patch.object(stop_current.signal, "pidfd_send_signal", side_effect=send, create=True),
                ):
                    self.assertEqual(stop_current.run(root, upload=False), 0)
                self.assertEqual(sent, [0, stop_current.signal.SIGINT])
                report.assert_called_once_with(root.resolve(), 1, root / "stop-current.json", unittest.mock.ANY, False)
                state = json.loads((root / "stop-current.json").read_text(encoding="utf-8"))
                self.assertTrue(state["signal_attempted"])
                self.assertEqual(state["controller_pid"], os.getpid())
                self.assertIn("started_at", state)
            finally:
                # run() closes the descriptor in its finally block. Closing a
                # second time would be an unrelated test error.
                pass

    def test_existing_signal_attempted_never_signals_again(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root, dataset, fingerprints = make_benchmark(Path(temporary))
            identity = {"pid": 42, "starttime": 9001, "cmdline_sha256": "b" * 64}
            write_json(
                root / "stop-current.json",
                {
                    "status": "installed",
                    "runner": identity,
                    "frozen": {"index": 0},
                    "report_target": 1,
                    "signal_attempted": True,
                },
            )
            fd = os.open(os.devnull, os.O_RDONLY)
            try:
                with (
                    patch.object(stop_current, "acquire_lock", return_value=DummyLock()),
                    patch.object(stop_current, "runner_spec", return_value={"pid": 42, "mode": "full198", "argv": []}),
                    patch.object(stop_current, "open_verified_pidfd", return_value=(fd, identity)),
                    patch.object(stop_current, "read_manifest_and_sources", return_value=fingerprints),
                    patch.object(stop_current, "read_dataset", return_value=dataset),
                    patch.object(stop_current.signal, "pidfd_send_signal", create=True) as send,
                ):
                    self.assertEqual(stop_current.run(root, upload=False), 0)
                send.assert_not_called()
            finally:
                os.close(fd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
