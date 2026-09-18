#!/usr/bin/env python3
"""Small, side-effect-free checks for the stage controller/report scripts."""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import stage_report
import stop_after


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def result(index: int, item_id: str, domain: str = "physics", seconds: float = 2.0) -> dict[str, object]:
    return {
        "id": item_id,
        "index": index,
        "fingerprints": {"harness": "test"},
        "grade": {"classification": "correct" if index % 2 == 0 else "incorrect"},
        "tokens": 100 + index,
        "wall_seconds": seconds,
        "timings": {"first_token_seconds": 0.1},
        "stop_type": "eos",
        "input_truncated": False,
        "attempt": 1,
        "output_sha256": hashlib.sha256(b"answer").hexdigest(),
        "request_sha256": hashlib.sha256(b"{}").hexdigest(),
        "domain": domain,
    }


class StageControlTests(unittest.TestCase):
    def make_benchmark(self, root: Path, count: int = 30) -> Path:
        benchmark = root / "benchmark"
        dataset = [
            {"id": f"item-{index}", "source_index": index, "problem": "test", "answer": "A", "permutation": [0, 1, 2, 3], "domain": "physics" if index < 20 else "chemistry"}
            for index in range(198)
        ]
        write_json(benchmark / "dataset.json", dataset)
        (benchmark / "config.json").write_bytes(b"{}")
        (benchmark / "run_gpqa.py").write_bytes(b"# test fixture\n")
        fingerprints = {name: hashlib.sha256((benchmark / filename).read_bytes()).hexdigest()
                        for name, filename in (("dataset", "dataset.json"), ("config", "config.json"), ("harness", "run_gpqa.py"))}
        write_json(benchmark / "results" / "manifest.json", {"fingerprints": fingerprints})
        for index in range(count):
            row = result(index, dataset[index]["id"], seconds=float(index + 1))
            row["fingerprints"] = fingerprints
            write_json(
                benchmark / "results" / "cases" / f"{index:03d}" / "result.json",
                row,
            )
            attempt = benchmark / "results" / "cases" / f"{index:03d}" / "attempt-01"
            attempt.mkdir()
            (attempt / "output.txt").write_bytes(b"answer")
            (attempt / "request.json").write_bytes(b"{}")
        return benchmark

    def test_requires_all_thirty_valid_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            benchmark = self.make_benchmark(Path(temporary), count=29)
            complete, detail = stop_after.results_complete(benchmark, 30)
            self.assertFalse(complete)
            self.assertIn("029", detail)
            dataset = stage_report.read_json(benchmark / "dataset.json")
            write_json(benchmark / "results" / "cases" / "029" / "result.json", result(29, dataset[29]["id"]))
            self.assertTrue(stop_after.results_complete(benchmark, 30)[0])

    def test_identity_refuses_a_non_runner_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = self.make_benchmark(root, count=0)
            write_json(benchmark / "process.json", {"pid": 77, "mode": "full198", "argv": ["python", "run_gpqa.py", "--output", str(benchmark / "results")]})
            proc = root / "proc" / "77"
            proc.mkdir(parents=True)
            # Field 22 is index 19 after the process state.
            (proc / "stat").write_text("77 (python) R " + "0 " * 18 + "12345 0 0\n", encoding="utf-8")
            (proc / "cmdline").write_bytes(b"python\0unrelated.py\0")
            spec = stop_after.load_runner_spec(benchmark)
            with self.assertRaises(stop_after.IdentityError):
                stop_after.live_identity(benchmark, spec, root / "proc")

    def test_report_uses_only_contiguous_prefix_and_domain_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            benchmark = self.make_benchmark(Path(temporary), count=2)
            summary = stage_report.build_summary(benchmark, 30)
            self.assertEqual(summary["completed_contiguous_prefix"], 2)
            self.assertFalse(summary["target_complete"])
            self.assertEqual(summary["score"]["graded"], 2)
            self.assertIn("physics", summary["domains"])
            self.assertFalse(summary["scope"]["is_random_sample"])
            report = stage_report.render_markdown(summary)
            self.assertIn("| 2 | physics |", report)  # human table is one-based

    def test_report_rejects_changed_request_and_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            benchmark = self.make_benchmark(Path(temporary), count=2)
            case = benchmark / "results" / "cases" / "001"
            (case / "attempt-01" / "request.json").write_bytes(b"changed")
            summary = stage_report.build_summary(benchmark, 2)
            self.assertEqual(summary["completed_contiguous_prefix"], 1)
            self.assertIn("request.json checksum mismatch", summary["missing_or_invalid"][0])
            (case / "attempt-01" / "request.json").write_bytes(b"{}")
            row = stage_report.read_json(case / "result.json")
            row["fingerprints"]["config"] = "0" * 64
            write_json(case / "result.json", row)
            self.assertEqual(stage_report.build_summary(benchmark, 2)["completed_contiguous_prefix"], 1)

    def test_report_refuses_modified_configuration_and_excludes_later_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            benchmark = self.make_benchmark(Path(temporary), count=3)
            summary = stage_report.build_summary(benchmark, 2)
            self.assertEqual(summary["score"]["graded"], 2)
            self.assertTrue(summary["target_complete"])
            (benchmark / "config.json").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "config fingerprint"):
                stage_report.build_summary(benchmark, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
