#!/usr/bin/env python3
"""Create a cautious stage report from a contiguous prefix of GPQA results.

This program deliberately reads only ``results/cases/000`` through the requested
prefix.  It neither starts inference nor touches the runner's files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_RESULT_FIELDS = (
    "id",
    "index",
    "fingerprints",
    "grade",
    "tokens",
    "wall_seconds",
    "timings",
    "stop_type",
    "input_truncated",
    "attempt",
    "output_sha256",
    "request_sha256",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def validate_result(result: Any, expected_index: int, expected_id: str) -> tuple[bool, str]:
    """Accept a fully atomically-written runner result, never a partial attempt."""
    if not isinstance(result, dict):
        return False, "result is not an object"
    missing = [field for field in REQUIRED_RESULT_FIELDS if field not in result]
    if missing:
        return False, "missing " + ", ".join(missing)
    if result["index"] != expected_index:
        return False, f"index is {result['index']!r}, expected {expected_index}"
    if result["id"] != expected_id:
        return False, "result id does not match dataset"
    if not isinstance(result["fingerprints"], dict):
        return False, "fingerprints is not an object"
    if not isinstance(result["grade"], dict) or not isinstance(result["grade"].get("classification"), str):
        return False, "grade.classification is missing"
    if number(result["tokens"]) is None or number(result["tokens"]) < 0:
        return False, "tokens is invalid"
    if number(result["wall_seconds"]) is None or number(result["wall_seconds"]) < 0:
        return False, "wall_seconds is invalid"
    if not isinstance(result["timings"], dict):
        return False, "timings is not an object"
    if not isinstance(result["stop_type"], str):
        return False, "stop_type is invalid"
    if not isinstance(result["input_truncated"], bool):
        return False, "input_truncated is invalid"
    if not ((type(result["attempt"]) is int and result["attempt"] >= 1) or
            (isinstance(result["attempt"], str) and result["attempt"].startswith("attempt-") and
             result["attempt"][8:].isdigit() and int(result["attempt"][8:]) >= 1)):
        return False, "attempt is invalid"
    for field in ("output_sha256", "request_sha256"):
        value = result[field]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            return False, f"{field} is invalid"
    return True, ""


def validate_evidence(path: Path, result: dict[str, Any], fingerprints: dict[str, Any]) -> None:
    if result["fingerprints"] != fingerprints:
        raise ValueError("result fingerprints do not match manifest")
    attempt = result["attempt"]
    name = f"attempt-{attempt:02d}" if type(attempt) is int else attempt
    directory = path.parent / name
    for filename, field in (("output.txt", "output_sha256"), ("request.json", "request_sha256")):
        digest = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        if digest != result[field]:
            raise ValueError(f"{filename} checksum mismatch")


def load_prefix(benchmark_dir: Path, target: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    dataset_path = benchmark_dir / "dataset.json"
    dataset = read_json(dataset_path)
    if not isinstance(dataset, list):
        raise ValueError(f"{dataset_path} is not a list")
    if target < 1 or target > len(dataset):
        raise ValueError(f"target must be between 1 and {len(dataset)}")
    fingerprints = read_json(benchmark_dir / "results" / "manifest.json")["fingerprints"]
    for name, filename in (("dataset", "dataset.json"), ("config", "config.json"), ("harness", "run_gpqa.py")):
        if hashlib.sha256((benchmark_dir / filename).read_bytes()).hexdigest() != fingerprints[name]:
            raise ValueError(f"{name} fingerprint does not match manifest")

    accepted: list[dict[str, Any]] = []
    issues: list[str] = []
    for index in range(target):
        item = dataset[index]
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError(f"dataset item {index} has no string id")
        path = benchmark_dir / "results" / "cases" / f"{index:03d}" / "result.json"
        if not path.is_file():
            issues.append(f"{index:03d}: result.json not present")
            break
        try:
            result = read_json(path)
        except (OSError, json.JSONDecodeError) as error:
            issues.append(f"{index:03d}: unreadable result.json ({error})")
            break
        valid, reason = validate_result(result, index, item["id"])
        if not valid:
            issues.append(f"{index:03d}: invalid result ({reason})")
            break
        try:
            validate_evidence(path, result, fingerprints)
        except (OSError, ValueError) as error:
            issues.append(f"{index:03d}: invalid evidence ({error})")
            break
        accepted.append({"dataset": item, "result": result})

    return dataset, accepted, issues


def percentile50(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def compact_number(value: float | None, decimals: int = 2) -> float | None:
    return round(value, decimals) if value is not None else None


def metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = [float(record["result"]["tokens"]) for record in records]
    seconds = [float(record["result"]["wall_seconds"]) for record in records]
    speeds = [token / second for token, second in zip(tokens, seconds) if second > 0]
    token_total = sum(tokens)
    second_total = sum(seconds)
    return {
        "count": len(records),
        "tokens": {
            "total": compact_number(token_total),
            "mean": compact_number(statistics.mean(tokens) if tokens else None),
            "median": compact_number(percentile50(tokens)),
        },
        "wall_seconds": {
            "total": compact_number(second_total),
            "mean": compact_number(statistics.mean(seconds) if seconds else None),
            "median": compact_number(percentile50(seconds)),
        },
        "tokens_per_second": {
            "weighted": compact_number(token_total / second_total if second_total > 0 else None, 3),
            "median_per_case": compact_number(percentile50(speeds), 3),
        },
    }


def score_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record["result"]["grade"]["classification"]] += 1
    correct = counts.get("correct", 0)
    return {
        "correct": correct,
        "graded": len(records),
        "accuracy": compact_number(correct / len(records) if records else None, 4),
        "classification_counts": dict(sorted(counts.items())),
    }


def build_summary(benchmark_dir: Path, target: int) -> dict[str, Any]:
    dataset, records, issues = load_prefix(benchmark_dir, target)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        domain = str(record["dataset"].get("domain") or "unknown")
        groups[domain].append(record)

    domains = {
        domain: {"score": score_summary(group), "performance": metric_summary(group)}
        for domain, group in sorted(groups.items())
    }
    slowest = sorted(records, key=lambda record: float(record["result"]["wall_seconds"]), reverse=True)[:5]
    slowest_rows = []
    for record in slowest:
        result = record["result"]
        seconds = float(result["wall_seconds"])
        tokens = float(result["tokens"])
        slowest_rows.append(
            {
                "index": int(result["index"]),
                "id": result["id"],
                "domain": record["dataset"].get("domain", "unknown"),
                "classification": result["grade"]["classification"],
                "tokens": tokens,
                "wall_seconds": compact_number(seconds),
                "tokens_per_second": compact_number(tokens / seconds if seconds > 0 else None, 3),
                "stop_type": result["stop_type"],
            }
        )

    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "benchmark_dir": str(benchmark_dir),
        "target_prefix_size": target,
        "dataset_size": len(dataset),
        "completed_contiguous_prefix": len(records),
        "target_complete": len(records) == target,
        "missing_or_invalid": issues,
        "scope": {
            "selection": f"original dataset order, dataset[0:{target}]",
            "is_random_sample": False,
            "is_balanced_by_domain": False,
            "official_198_score": False,
            "claim": "stage-only result; it is not an official GPQA Diamond score or a full-198 reproduction",
        },
        "score": score_summary(records),
        "performance": metric_summary(records),
        "domains": domains,
        "slowest_cases": slowest_rows,
    }


def fmt(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def render_markdown(summary: dict[str, Any]) -> str:
    score = summary["score"]
    performance = summary["performance"]
    completed = summary["completed_contiguous_prefix"]
    target = summary["target_prefix_size"]
    state = "已完成" if summary["target_complete"] else "进行中"
    lines = [
        "# GPQA Diamond 阶段性验收报告",
        "",
        f"生成时间：{summary['generated_at']}  ",
        f"范围：原始数据集顺序的前 {target} 题；当前连续完成 {completed}/{target}（{state}）。",
        "",
        "## 结论边界",
        "",
        "这是一段固定前缀样本，不是随机抽样，也未按学科均衡。它可用于确认部署、思考输出、判分和实际性能；不能替代完整 198 题评测，不能宣称为官方 GPQA Diamond 分数，也不能据此与官方公布分数直接对标。",
        "",
        "## 当前得分与性能",
        "",
        f"- 判分：{score['correct']}/{score['graded']} 正确；准确率 {fmt(score['accuracy'] * 100 if score['accuracy'] is not None else None, '%')}。",
        f"- 输出：总计 {fmt(performance['tokens']['total'])} token；单题中位数 {fmt(performance['tokens']['median'])} token。",
        f"- 耗时：总计 {fmt(performance['wall_seconds']['total'], ' s')}；单题中位数 {fmt(performance['wall_seconds']['median'], ' s')}。",
        f"- 速度：按总 token/总耗时 {fmt(performance['tokens_per_second']['weighted'], ' tok/s')}；单题速度中位数 {fmt(performance['tokens_per_second']['median_per_case'], ' tok/s')}。",
        "",
        "## 按学科",
        "",
        "| 学科 | 题数 | 正确 | 准确率 | token 中位数 | 耗时中位数 | 加权速度 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for domain, values in summary["domains"].items():
        domain_score = values["score"]
        domain_performance = values["performance"]
        lines.append(
            "| {domain} | {count} | {correct} | {accuracy} | {tokens} | {seconds} | {speed} |".format(
                domain=domain,
                count=domain_score["graded"],
                correct=domain_score["correct"],
                accuracy=fmt(domain_score["accuracy"] * 100 if domain_score["accuracy"] is not None else None, "%"),
                tokens=fmt(domain_performance["tokens"]["median"]),
                seconds=fmt(domain_performance["wall_seconds"]["median"], " s"),
                speed=fmt(domain_performance["tokens_per_second"]["weighted"], " tok/s"),
            )
        )
    lines.extend(
        [
            "",
            "## 最慢的已完成题目",
            "",
            "| 序号 | 学科 | 判分 | 输出 token | 耗时 | 速度 | 停止类型 |",
            "|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for row in summary["slowest_cases"]:
        lines.append(
            "| {index} | {domain} | {classification} | {tokens:.0f} | {seconds} | {speed} | {stop_type} |".format(
                # Results retain the runner's 0-based index in JSON.  The
                # Notebook and this human-facing table use one-based question
                # numbers so they are directly comparable.
                index=row["index"] + 1,
                domain=row["domain"],
                classification=row["classification"],
                tokens=row["tokens"],
                seconds=fmt(row["wall_seconds"], " s"),
                speed=fmt(row["tokens_per_second"], " tok/s"),
                stop_type=row["stop_type"],
            )
        )
    if summary["missing_or_invalid"]:
        lines.extend(["", "## 未完成或无效结果", ""])
        lines.extend(f"- {issue}" for issue in summary["missing_or_invalid"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--target", type=int, required=True, help="Frozen completed prefix from stop-current.json")
    args = parser.parse_args()
    benchmark_dir = args.benchmark_dir.resolve()
    summary = build_summary(benchmark_dir, args.target)
    atomic_write_json(benchmark_dir / "stage-summary.json", summary)
    atomic_write_text(benchmark_dir / "STAGE-REPORT.md", render_markdown(summary))
    print(
        f"stage report: {summary['completed_contiguous_prefix']}/{summary['target_prefix_size']} "
        f"contiguous results -> {benchmark_dir / 'STAGE-REPORT.md'}"
    )
    return 0 if summary["target_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
