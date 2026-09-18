import importlib.util
import json
import sys
import threading
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODULE = Path(__file__).with_name("extended-benchmark.py")
SPEC = importlib.util.spec_from_file_location("extended_benchmark", MODULE)
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)


class BenchmarkTests(unittest.TestCase):
    def test_sse_parser_handles_chunk_boundaries_and_done(self):
        chunks = [b"data: {\"a\":", b"1}\r\n\r\n", b":keepalive\n", b"data: [DONE]\n\n"]
        self.assertEqual(list(mod.sse_events(chunks)), ['{"a":1}', '[DONE]'])

    def test_event_content_and_percentiles(self):
        event = {"choices": [{"delta": {"content": "a"}}, {"delta": {"content": "b"}}]}
        self.assertEqual(mod.event_content(event), "ab")
        self.assertEqual(mod.percentile([1.0, 2.0, 3.0, 4.0], .5), 2.5)

    def test_aggregate_preserves_missing_usage_and_counts_reported_only(self):
        records = [
            {"http_status": 200, "error": None, "completion_tokens": 4, "ttft_s": 1.0, "wall_s": 2.0,
             "valid_for_throughput": True, "quality_class": "normal_stop"},
            {"http_status": 500, "error": "x", "completion_tokens": None, "ttft_s": None, "wall_s": 1.0},
        ]
        result = mod.aggregate(records, 2.0)
        self.assertEqual(result["completion_tokens_reported"], 4)
        self.assertEqual(result["valid_for_throughput"], 1)
        self.assertEqual(result["normal_stop"]["ttft_s"]["p95"], 1.0)

    def test_aggregate_separates_normal_stop_length_and_invalid(self):
        base = {"http_status": 200, "error": None, "completion_tokens": 4, "ttft_s": 1.0, "wall_s": 2.0, "valid_for_throughput": True}
        records = [dict(base, quality_class="normal_stop"), dict(base, quality_class="length_cutoff"), dict(base, valid_for_throughput=False, quality_class="invalid")]
        result = mod.aggregate(records, 2.0)
        self.assertEqual(result["normal_stop"]["requests"], 1)
        self.assertEqual(result["length_cutoff"]["requests"], 1)
        self.assertEqual(result["invalid_requests"], 1)
        self.assertEqual(result["valid_for_throughput"], 2)

    def test_concurrency_requires_two_rounds(self):
        args = type("Args", (), {"concurrency": 1, "rounds": 1})()
        with __import__("tempfile").TemporaryDirectory() as td:
            with self.assertRaises(mod.BenchmarkError):
                mod.command_concurrency(args, Path(td))

    def test_document_needle_values_are_deterministic(self):
        first = mod.document_for_target(7, 100, 10, 100)
        second = mod.document_for_target(7, 100, 10, 100)
        self.assertEqual(first, second)
        body, values, _offsets = first
        self.assertTrue(all(value in body for value in values.values()))
        self.assertEqual(len(set(values.values())), 3)
        self.assertEqual(values, {
            key: "value-" + mod.hashlib.sha256(f"dsv41-needle-v2:7:{key}".encode()).hexdigest()[:16]
            for key in ("A", "B", "C")
        })

    def test_run_chat_requires_complete_stream_usage_and_finish(self):
        class Handler(BaseHTTPRequestHandler):
            payload = None
            def do_POST(self):
                if self.path == "/v1/chat/completions":
                    Handler.payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    frames = [
                        b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n',
                        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n\n',
                        b'data: [DONE]\n\n',
                    ]
                    for frame in frames:
                        self.wfile.write(frame)
                        self.wfile.flush()
                elif self.path == "/tokenize":
                    self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b'{"tokens":[1,2,3]}')
            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with __import__("tempfile").TemporaryDirectory() as td:
                record = mod.run_chat(f"http://127.0.0.1:{server.server_port}", [{"role": "user", "content": "x"}], 8, 5, Path(td), "ok", 2)
                self.assertTrue(record["stream_done"])
                self.assertTrue(record["valid_for_throughput"])
                self.assertEqual(record["quality_class"], "normal_stop")
                self.assertEqual(Handler.payload["stream_options"], {"include_usage": True})
        finally:
            server.shutdown(); thread.join(); server.server_close()

    def test_http_200_without_done_or_content_is_invalid(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers(); self.wfile.write(b'data: {"choices":[{"delta":{}}]}\n\n')
            def log_message(self, *_args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with __import__("tempfile").TemporaryDirectory() as td:
                record = mod.run_chat(f"http://127.0.0.1:{server.server_port}", [{"role": "user", "content": "x"}], 8, 5, Path(td), "bad", 2)
                self.assertFalse(record["valid_for_throughput"])
                self.assertIn("sse_stream_not_complete", record["validation_errors"])
                self.assertIn("empty_content", record["validation_errors"])
        finally:
            server.shutdown(); thread.join(); server.server_close()

    def test_usage_prompt_must_cover_raw_tokenize_count(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n')
                self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n')
                self.wfile.write(b'data: [DONE]\n\n')
            def log_message(self, *_args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with __import__("tempfile").TemporaryDirectory() as td:
                record = mod.run_chat(f"http://127.0.0.1:{server.server_port}", [{"role": "user", "content": "x"}], 8, 5, Path(td), "usage", 2)
                self.assertFalse(record["valid_for_throughput"])
                self.assertIn("usage_prompt_tokens_below_raw_tokenize_count", record["validation_errors"])
        finally:
            server.shutdown(); thread.join(); server.server_close()

    def test_usage_total_must_equal_prompt_plus_completion(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n')
                self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":5}}\n\n')
                self.wfile.write(b'data: [DONE]\n\n')
            def log_message(self, *_args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with __import__("tempfile").TemporaryDirectory() as td:
                record = mod.run_chat(f"http://127.0.0.1:{server.server_port}", [{"role": "user", "content": "x"}], 8, 5, Path(td), "usage-total", 2)
                self.assertFalse(record["valid_for_throughput"])
                self.assertIn("usage_total_tokens_mismatch", record["validation_errors"])
        finally:
            server.shutdown(); thread.join(); server.server_close()

    def test_length_finish_is_not_long_quality_pass(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"alpha-1-10 beta-1-50 gamma-1-90"}}]}\n\n')
                self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":3,"completion_tokens":8}}\n\n')
                self.wfile.write(b'data: [DONE]\n\n')
            def log_message(self, *_args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with __import__("tempfile").TemporaryDirectory() as td:
                record = mod.run_chat(f"http://127.0.0.1:{server.server_port}", [{"role": "user", "content": "x"}], 8, 5, Path(td), "length", 2)
                self.assertTrue(record["valid_for_throughput"])
                self.assertEqual(record["quality_class"], "length_cutoff")
        finally:
            server.shutdown(); thread.join(); server.server_close()

    def test_long_rejects_zero_token_calibration(self):
        args = type("Args", (), {"input_tokens": 10, "max_tokens": 2, "template_reserve": 1, "context_capacity": 20,
                                  "base_url": "http://unused", "timeout": 1, "seed": 1})()
        with __import__("tempfile").TemporaryDirectory() as td:
            with mock.patch.object(mod, "tokenize", return_value=0):
                with self.assertRaises(mod.BenchmarkError):
                    mod.command_long(args, Path(td))

    def test_concurrency_invalid_request_returns_error(self):
        args = type("Args", (), {"concurrency": 1, "rounds": 2, "prompt": "p", "max_tokens": 2,
                                  "base_url": "http://unused", "timeout": 1})()
        invalid = {"http_status": 500, "error": "server", "completion_tokens": None,
                   "ttft_s": None, "wall_s": 0.1, "valid_for_throughput": False,
                   "quality_class": "invalid", "finish_reason": None, "generated_text": ""}
        with __import__("tempfile").TemporaryDirectory() as td:
            with mock.patch.object(mod, "tokenize", return_value=1), mock.patch.object(mod, "run_chat", return_value=dict(invalid)):
                with self.assertRaises(mod.BenchmarkError):
                    mod.command_concurrency(args, Path(td))

    def test_long_reduces_target_to_capacity_budget(self):
        args = type("Args", (), {"input_tokens": 100, "max_tokens": 2, "template_reserve": 2, "context_capacity": 20,
                                  "base_url": "http://unused", "timeout": 1, "seed": 1})()
        needle_values = mod.document_for_target(1, 64, 1, 1)[1]
        good = {"http_status": 200, "error": None, "wall_s": 1.0, "ttft_s": 0.1,
                "finish_reason": "stop", "prompt_tokens": 16, "completion_tokens": 2,
                "total_tokens": 18, "valid_for_throughput": True, "validation_errors": [],
                "generated_text": " ".join(needle_values.values())}
        with __import__("tempfile").TemporaryDirectory() as td:
            with mock.patch.object(mod, "tokenize", side_effect=[1, 16, 4, 8, 14]), mock.patch.object(mod, "run_chat", return_value=dict(good)):
                self.assertEqual(mod.command_long(args, Path(td)), 0)
            summary = json.loads((Path(td) / "long-summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["effective_input_token_target"], 16)
            self.assertLessEqual(summary["actual_tokenize_input_tokens"], 16)
            self.assertEqual(summary["needle_scheme"], "independent-sha256-v2")


if __name__ == "__main__":
    unittest.main()
