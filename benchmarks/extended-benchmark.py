#!/usr/bin/env python3
"""Record long-context and concurrent llama-server chat benchmarks.

The script is deliberately standard-library only. It does not start, stop, or
reconfigure llama-server. Run it on the same Linux host as the loopback API.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


class BenchmarkError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def post_json(url: str, payload: Mapping[str, Any], timeout: int) -> Tuple[int, bytes]:
    request = urllib.request.Request(
        url, data=canonical_json(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        raise BenchmarkError(f"request to {url} failed: {exc.reason}") from exc


def tokenize(base_url: str, text: str, timeout: int) -> int:
    """Use llama-server's /tokenize rather than a guessed tokenizer."""
    status, body = post_json(base_url.rstrip("/") + "/tokenize", {"content": text}, timeout)
    if status != 200:
        raise BenchmarkError(f"/tokenize returned HTTP {status}: {body[:1000]!r}")
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise BenchmarkError("/tokenize did not return JSON") from exc
    tokens = parsed.get("tokens") if isinstance(parsed, Mapping) else None
    if not isinstance(tokens, list):
        raise BenchmarkError("/tokenize JSON lacks a tokens array")
    return len(tokens)


def sse_events(chunks: Iterable[bytes]) -> Iterator[str]:
    """Yield data payloads from an SSE byte stream, accepting CRLF and chunks."""
    pending = ""
    for chunk in chunks:
        pending += chunk.decode("utf-8", errors="replace")
        pending = pending.replace("\r\n", "\n")
        while "\n\n" in pending:
            frame, pending = pending.split("\n\n", 1)
            data = [line[5:].lstrip() for line in frame.split("\n") if line.startswith("data:")]
            if data:
                yield "\n".join(data)
    if pending:
        data = [line[5:].lstrip() for line in pending.split("\n") if line.startswith("data:")]
        if data:
            yield "\n".join(data)


def sse_frame_data(frame: str) -> Optional[str]:
    data = [line[5:].lstrip() for line in frame.split("\n") if line.startswith("data:")]
    return "\n".join(data) if data else None


def event_content(event: Mapping[str, Any]) -> str:
    choices = event.get("choices")
    if not isinstance(choices, list):
        return ""
    parts: List[str] = []
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        delta = choice.get("delta")
        message = choice.get("message")
        for container in (delta, message):
            if isinstance(container, Mapping) and isinstance(container.get("content"), str):
                parts.append(container["content"])
    return "".join(parts)


def find_usage(event: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    usage = event.get("usage")
    return dict(usage) if isinstance(usage, Mapping) else None


def find_timings(event: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    timings = event.get("timings")
    return dict(timings) if isinstance(timings, Mapping) else None


def percentile(values: Sequence[float], point: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * point
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def run_chat(base_url: str, messages: List[Dict[str, str]], max_tokens: int, timeout: int,
             output_dir: Path, request_id: str,
             expected_raw_prompt_tokens: Optional[int] = None,
             expected_prompt_token_upper_bound: Optional[int] = None) -> Dict[str, Any]:
    """Perform one SSE request and persist exact stream, text, and metadata."""
    payload: Dict[str, Any] = {
        "messages": messages, "temperature": 0, "cache_prompt": False,
        "max_tokens": max_tokens, "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.monotonic()
    wall_started_at = now()
    raw_parts: List[bytes] = []
    content_parts: List[str] = []
    parsed_events: List[Dict[str, Any]] = []
    first_sse_s: Optional[float] = None
    first_content_s: Optional[float] = None
    status: Optional[int] = None
    error: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    timings: Optional[Dict[str, Any]] = None
    stream_done = False

    def consume_data(data: str) -> None:
        """Process one complete SSE data payload, including the final frame."""
        nonlocal first_sse_s, first_content_s, finish_reason, usage, timings, stream_done
        if first_sse_s is None:
            first_sse_s = time.monotonic() - started
        if data == "[DONE]":
            stream_done = True
            return
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            parsed_events.append({"_invalid_sse_data": data})
            return
        if not isinstance(event, Mapping):
            return
        event_dict = dict(event)
        parsed_events.append(event_dict)
        text = event_content(event_dict)
        if text:
            content_parts.append(text)
            if first_content_s is None:
                first_content_s = time.monotonic() - started
        possible_usage, possible_timings = find_usage(event_dict), find_timings(event_dict)
        if possible_usage is not None:
            usage = possible_usage
        if possible_timings is not None:
            timings = possible_timings
        choices = event_dict.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if isinstance(choice, Mapping) and choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])

    url = base_url.rstrip("/") + "/v1/chat/completions"
    request = urllib.request.Request(url, data=canonical_json(payload).encode("utf-8"), headers={"Content-Type": "application/json", "Accept": "text/event-stream"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            pending = ""
            for line in response:
                raw_parts.append(line)
                pending += line.decode("utf-8", errors="replace").replace("\r\n", "\n")
                while "\n\n" in pending:
                    frame, pending = pending.split("\n\n", 1)
                    data = sse_frame_data(frame)
                    if data is None:
                        continue
                    consume_data(data)
            tail_data = sse_frame_data(pending)
            if tail_data:
                consume_data(tail_data)
    except urllib.error.HTTPError as exc:
        status, error = exc.code, exc.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    wall_s = time.monotonic() - started
    raw_path = output_dir / "requests" / f"{safe_name(request_id)}.sse"
    text_path = output_dir / "requests" / f"{safe_name(request_id)}.txt"
    meta_path = output_dir / "requests" / f"{safe_name(request_id)}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(b"".join(raw_parts))
    text_path.write_text("".join(content_parts), encoding="utf-8")
    usage_prompt = usage.get("prompt_tokens") if usage else None
    usage_completion = usage.get("completion_tokens") if usage else None
    usage_total = usage.get("total_tokens") if usage else None
    validation_errors: List[str] = []
    if status != 200:
        validation_errors.append("http_status_not_200")
    if error:
        validation_errors.append("transport_error")
    if not stream_done:
        validation_errors.append("sse_stream_not_complete")
    if not content_parts:
        validation_errors.append("empty_content")
    if finish_reason is None:
        validation_errors.append("missing_finish_reason")
    if not isinstance(usage_prompt, int) or usage_prompt <= 0:
        validation_errors.append("missing_prompt_tokens")
    if not isinstance(usage_completion, int) or usage_completion < 0:
        validation_errors.append("missing_completion_tokens")
    elif content_parts and usage_completion <= 0:
        validation_errors.append("completion_tokens_not_positive_for_nonempty_content")
    if usage is not None and "total_tokens" in usage:
        if (not isinstance(usage_total, int) or not isinstance(usage_prompt, int)
                or not isinstance(usage_completion, int)
                or usage_total != usage_prompt + usage_completion):
            validation_errors.append("usage_total_tokens_mismatch")
    if expected_raw_prompt_tokens is not None and isinstance(usage_prompt, int) and usage_prompt < expected_raw_prompt_tokens:
        validation_errors.append("usage_prompt_tokens_below_raw_tokenize_count")
    if (expected_prompt_token_upper_bound is not None and isinstance(usage_prompt, int)
            and usage_prompt > expected_prompt_token_upper_bound):
        validation_errors.append("usage_prompt_tokens_above_reserved_capacity")
    if finish_reason not in (None, "stop", "length"):
        validation_errors.append("unsupported_finish_reason")
    valid_for_throughput = not validation_errors and finish_reason in ("stop", "length")
    record: Dict[str, Any] = {
        "id": request_id, "started_at": wall_started_at, "url": url, "http_status": status,
        "error": error, "wall_s": wall_s, "ttft_s": first_content_s,
        "first_sse_s": first_sse_s, "finish_reason": finish_reason,
        "prompt_tokens": usage_prompt,
        "completion_tokens": usage_completion,
        "total_tokens": usage_total,
        "usage": usage, "timings": timings, "request": {k: v for k, v in payload.items() if k != "messages"},
        "message_hashes": [sha256_text(m["content"]) for m in messages],
        "raw_sse": str(raw_path), "text": str(text_path), "text_sha256": sha256_text("".join(content_parts)),
        "event_count": len(parsed_events), "stream_done": stream_done,
        "validation_errors": validation_errors,
        "valid_for_throughput": valid_for_throughput,
        "quality_class": "normal_stop" if valid_for_throughput and finish_reason == "stop" else ("length_cutoff" if valid_for_throughput else "invalid"),
        "expected_raw_prompt_tokens": expected_raw_prompt_tokens,
        "expected_prompt_token_upper_bound": expected_prompt_token_upper_bound,
    }
    write_json(meta_path, record)
    record["meta"] = str(meta_path)
    record["generated_text"] = "".join(content_parts)
    return record


def document_for_target(seed: int, requested_tokens: int, calibration_tokens: int,
                        calibration_chars: int) -> Tuple[str, Dict[str, str], Dict[str, int]]:
    """Create reproducible filler and return character offsets for three needles."""
    chars_per_token = max(1.0, calibration_chars / max(1, calibration_tokens))
    body_chars = max(2048, int(requested_tokens * chars_per_token))
    values = {
        key: "value-" + hashlib.sha256(f"dsv41-needle-v2:{seed}:{key}".encode()).hexdigest()[:16]
        for key in ("A", "B", "C")
    }
    unit = ("第{index:08d}段：这是一段固定的长上下文测试材料，用于验证检索位置和生成稳定性。" )
    chunks: List[str] = []
    inserted = set()
    length = 0
    index = 0
    positions = {"A": int(body_chars * .10), "B": int(body_chars * .50), "C": int(body_chars * .90)}
    while length < body_chars:
        for key in ("A", "B", "C"):
            if key not in inserted and length >= positions[key]:
                needle = f"\n[NEEDLE_{key}] 唯一键值={values[key]}。\n"
                chunks.append(needle); length += len(needle); inserted.add(key)
        line = unit.format(index=index)
        chunks.append(line); length += len(line); index += 1
    for key in ("A", "B", "C"):
        if key not in inserted:
            chunks.append(f"\n[NEEDLE_{key}] 唯一键值={values[key]}。\n")
    body = "".join(chunks)
    offsets = {key: body.find(f"[NEEDLE_{key}]") for key in values}
    return body, values, offsets


def aggregate(records: Sequence[Mapping[str, Any]], batch_wall_s: float) -> Dict[str, Any]:
    valid = [r for r in records if r.get("valid_for_throughput") is True]
    normal = [r for r in valid if r.get("quality_class") == "normal_stop"]
    cutoff = [r for r in valid if r.get("quality_class") == "length_cutoff"]
    completion = sum(int(r["completion_tokens"]) for r in valid if isinstance(r.get("completion_tokens"), int))
    def metrics(group: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        ttfts = [float(r["ttft_s"]) for r in group if isinstance(r.get("ttft_s"), (int, float))]
        walls = [float(r["wall_s"]) for r in group if isinstance(r.get("wall_s"), (int, float))]
        tokens = sum(int(r["completion_tokens"]) for r in group if isinstance(r.get("completion_tokens"), int))
        return {"requests": len(group), "completion_tokens": tokens,
                "completion_tokens_per_batch_wall_s": tokens / batch_wall_s if batch_wall_s and tokens else None,
                "ttft_s": {"p50": percentile(ttfts, .50), "p95": percentile(ttfts, .95)},
                "wall_s": {"p50": percentile(walls, .50), "p95": percentile(walls, .95)}}
    return {
        "requests": len(records), "valid_for_throughput": len(valid), "invalid_requests": len(records) - len(valid), "batch_wall_s": batch_wall_s,
        "completion_tokens_reported": completion,
        "aggregate_completion_tokens_per_s": completion / batch_wall_s if batch_wall_s and completion else None,
        "normal_stop": metrics(normal), "length_cutoff": metrics(cutoff),
    }


def command_long(args: argparse.Namespace, output_dir: Path) -> int:
    available_input_budget = args.context_capacity - args.max_tokens - args.template_reserve
    if available_input_budget <= 0:
        raise BenchmarkError("context capacity leaves no room for input after max output and template reserve")
    effective_target = min(args.input_tokens, available_input_budget)
    sample = "测试分词校准。" * 1024
    calibration = tokenize(args.base_url, sample, args.timeout)
    estimate = effective_target
    prompt = ""
    values: Dict[str, str] = {}
    actual_input_tokens = 0
    converged = False
    needle_char_offsets: Dict[str, int] = {}
    # /tokenize measures raw message content only. Chat-template overhead is
    # represented by the explicit reserve and is not silently called measured.
    for _ in range(8):
        body, values, needle_char_offsets = document_for_target(args.seed, max(64, estimate), calibration, len(sample))
        prompt = ("请阅读下面的文档，只输出三个唯一键值，格式为 A=<值>; B=<值>; C=<值>。\n\n" + body)
        actual_input_tokens = tokenize(args.base_url, prompt, args.timeout)
        tolerance = max(8, effective_target // 100)
        if actual_input_tokens > 0 and actual_input_tokens <= available_input_budget and abs(actual_input_tokens - effective_target) <= tolerance:
            converged = True
            break
        if actual_input_tokens <= 0:
            raise BenchmarkError("/tokenize returned zero tokens while calibrating")
        estimate = max(64, round(estimate * effective_target / actual_input_tokens))
    if not converged:
        raise BenchmarkError(f"prompt token calibration did not converge after 8 iterations (target={effective_target}, actual={actual_input_tokens}, input_budget={available_input_budget})")
    prompt_path = output_dir / "prompts" / "long-context.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")
    prefix = "请阅读下面的文档，只输出三个唯一键值，格式为 A=<值>; B=<值>; C=<值>。\n\n"
    needle_token_offsets = {key: tokenize(args.base_url, prefix + body[:offset], args.timeout) for key, offset in needle_char_offsets.items()}
    record = run_chat(args.base_url, [{"role": "user", "content": prompt}], args.max_tokens, args.timeout,
                      output_dir, "long-000", actual_input_tokens,
                      args.context_capacity - args.max_tokens)
    text = record.pop("generated_text")
    quality = {key: value in text for key, value in values.items()}
    quality_ok = all(quality.values()) and record.get("finish_reason") == "stop"
    result = {"kind": "long", "created_at": now(), "input_token_target": args.input_tokens,
        "effective_input_token_target": effective_target,
        "input_budget_tokens": available_input_budget,
        "actual_tokenize_input_tokens": actual_input_tokens, "context_capacity_config": args.context_capacity,
        "template_reserve_tokens": args.template_reserve, "max_tokens_requested": args.max_tokens,
        "calibration": {"mode": "raw_message_content", "chars": len(sample), "tokenize_tokens": calibration,
                        "iterations": "up_to_8", "converged": converged,
                        "chat_template_reserve_estimate_tokens": args.template_reserve},
        "prompt": {"path": str(prompt_path), "sha256": sha256_text(prompt), "chars": len(prompt)},
        "needle_scheme": "independent-sha256-v2",
        "needles": values, "needle_token_offsets_raw_content": needle_token_offsets,
        "quality": {"needle_values_in_output": quality, "all_needles_found": all(quality.values()),
                    "finish_reason_stop": record.get("finish_reason") == "stop",
                    "quality_pass": quality_ok,
                    "valid_for_throughput": record["valid_for_throughput"],
                    "validation_errors": record["validation_errors"]},
        "request": record}
    write_json(output_dir / "long-summary.json", result)
    print(json.dumps({"outdir": str(output_dir), "actual_tokenize_input_tokens": actual_input_tokens, "quality": result["quality"]}, ensure_ascii=False))
    if record["validation_errors"] or not quality_ok:
        raise BenchmarkError("long benchmark failed response validation or needle quality; see long-summary.json")
    return 0


def command_concurrency(args: argparse.Namespace, output_dir: Path) -> int:
    if args.concurrency not in (1, 2, 4, 8):
        raise BenchmarkError("--concurrency must be one of 1, 2, 4, 8")
    if args.rounds < 2:
        raise BenchmarkError("--rounds must be at least 2")
    all_records: List[Dict[str, Any]] = []
    rounds: List[Dict[str, Any]] = []
    prompt = args.prompt
    prompt_path = output_dir / "prompts" / "concurrency.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True); prompt_path.write_text(prompt, encoding="utf-8")
    prompts: Dict[Tuple[int, int], str] = {}
    prompt_raw_tokens: Dict[Tuple[int, int], int] = {}
    for round_number in range(args.rounds):
        for slot in range(args.concurrency):
            prompts[(round_number, slot)] = f"{args.prompt}\n\n请求轮次={round_number + 1} 槽位={slot + 1}。请根据该编号给出不同但简短的回答。\n" + ("固定材料 " + f"slot-{slot + 1:02d}-round-{round_number + 1:02d}。" * 64)
            variant_path = output_dir / "prompts" / f"r{round_number + 1}s{slot + 1}.txt"
            variant_path.write_text(prompts[(round_number, slot)], encoding="utf-8")
            prompt_raw_tokens[(round_number, slot)] = tokenize(args.base_url, prompts[(round_number, slot)], args.timeout)
        barrier = threading.Barrier(args.concurrency)
        def one(slot: int) -> Dict[str, Any]:
            barrier.wait(timeout=30)
            message = prompts[(round_number, slot)]
            return run_chat(args.base_url, [{"role": "user", "content": message}], args.max_tokens, args.timeout, output_dir, f"concurrency-r{round_number + 1}-s{slot + 1}", prompt_raw_tokens[(round_number, slot)])
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            records = list(pool.map(one, range(args.concurrency)))
        batch = aggregate(records, time.monotonic() - started)
        batch["round"] = round_number + 1
        batch["normal_stop_nonempty"] = all(r.get("finish_reason") == "stop" and bool(r.get("generated_text")) for r in records)
        for record in records:
            record.pop("generated_text", None)
        rounds.append(batch); all_records.extend(records)
    result = {"kind": "concurrency", "created_at": now(), "concurrency": args.concurrency, "rounds_requested": args.rounds,
        "max_tokens_requested": args.max_tokens, "cache_prompt": False,
        "prompt": {"path": str(prompt_path), "sha256": sha256_text(prompt), "variant_count": len(prompts),
                   "variant_hashes": {f"r{r + 1}s{s + 1}": sha256_text(value) for (r, s), value in prompts.items()},
                   "variant_raw_token_counts": {f"r{r + 1}s{s + 1}": count for (r, s), count in prompt_raw_tokens.items()}}, "rounds": rounds,
        "aggregate": aggregate(all_records, sum(float(r["batch_wall_s"]) for r in rounds)), "requests": all_records}
    write_json(output_dir / "concurrency-summary.json", result)
    print(json.dumps({"outdir": str(output_dir), "aggregate": result["aggregate"]}, ensure_ascii=False))
    if result["aggregate"]["invalid_requests"]:
        raise BenchmarkError(f"concurrency benchmark had {result['aggregate']['invalid_requests']} invalid request(s); see concurrency-summary.json")
    return 0


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--base-url", default="http://127.0.0.1:48241")
    common.add_argument("--timeout", type=int, default=1800, help="Per HTTP request timeout in seconds")
    common.add_argument("--outdir", type=Path, default=None)
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    long = sub.add_parser("long", parents=[common], help="Run one measured long-context needle retrieval request")
    long.add_argument("--input-tokens", type=int, required=True, help="Target raw prompt tokens; actual /tokenize count is recorded")
    long.add_argument("--context-capacity", type=int, default=1048576)
    long.add_argument("--template-reserve", type=int, default=256)
    long.add_argument("--max-tokens", type=int, default=64)
    long.add_argument("--seed", type=int, default=20260916)
    conc = sub.add_parser("concurrency", parents=[common], help="Run synchronized concurrent generation batches")
    conc.add_argument("--concurrency", type=int, required=True, choices=(1, 2, 4, 8))
    conc.add_argument("--rounds", type=int, default=2)
    conc.add_argument("--max-tokens", type=int, default=128)
    conc.add_argument("--prompt", default="请用中文写出一个简洁、完整的 AMD GPU 推理性能测试建议。")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.timeout <= 0:
        raise BenchmarkError("--timeout must be positive")
    outdir = args.outdir or Path("extended-benchmark-" + datetime.now().strftime("%Y%m%dT%H%M%S"))
    outdir.mkdir(parents=True, exist_ok=False)
    write_json(outdir / "run.json", {"started_at": now(), "argv": sys.argv[1:] if argv is None else list(argv), "base_url": args.base_url})
    return command_long(args, outdir) if args.command == "long" else command_concurrency(args, outdir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
