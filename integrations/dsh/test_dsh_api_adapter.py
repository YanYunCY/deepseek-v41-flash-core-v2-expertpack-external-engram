#!/usr/bin/env python3
"""Offline unit tests for dsh_api_adapter.

Every native response is synthetic.  These tests never contact the model,
the ModelScope workspace, or a relay.
"""

from __future__ import annotations

import http.client
import json
import threading
import unittest
from typing import Any, Iterable, Mapping
from unittest.mock import patch

import dsh_api_adapter as adapter


def native_event(value: Mapping[str, Any]) -> bytes:
    return ("data: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode("utf-8")


class FakeEncoder:
    def __init__(self, parsed: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
        self.parse_calls: list[tuple[str, str]] = []
        self.parsed = parsed or {"role": "assistant", "content": "answer", "reasoning_content": "first"}

    def encode_messages(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.calls.append((messages, kwargs))
        return "official-encoded-prompt"

    def parse_message_from_completion_text(self, text: str, thinking_mode: str) -> dict[str, Any]:
        self.parse_calls.append((text, thinking_mode))
        return self.parsed


class FakeUpstream:
    def __init__(self, chunks: Iterable[bytes] = ()) -> None:
        self.chunks = list(chunks)
        self.tokenized: list[Any] = []
        self.payloads: list[Mapping[str, Any]] = []
        self.closed = False

    def tokenize(self, encoded: Any) -> list[int]:
        self.tokenized.append(encoded)
        return [128799, 42, 128804]

    def stream_completion(self, payload: Mapping[str, Any]):
        self.payloads.append(payload)
        try:
            yield from self.chunks
        finally:
            self.closed = True

    def health(self) -> dict[str, str]:
        return {"status": "ok"}


def request(**extra: Any) -> dict[str, Any]:
    value = {
        "model": adapter.MODEL_ID,
        "messages": [{"role": "user", "content": "Solve this."}],
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
    }
    value.update(extra)
    return value


class AdapterTests(unittest.TestCase):
    def make_service(self, chunks: Iterable[bytes], parsed: dict[str, Any] | None = None) -> tuple[adapter.AdapterService, FakeEncoder, FakeUpstream]:
        encoder = FakeEncoder(parsed)
        upstream = FakeUpstream(chunks)
        return adapter.AdapterService(encoder, upstream), encoder, upstream

    def test_tool_choice_none_removes_definitions_and_rejects_calls(self) -> None:
        tool = {"type": "function", "function": {"name": "ping", "parameters": {}}}
        parsed = {"role": "assistant", "content": "", "tool_calls": [tool]}
        service, encoder, _ = self.make_service(
            [native_event({"content": "", "stop": True, "stop_type": "eos"})], parsed)
        prepared = service.prepare(request(
            tools=[tool], tool_choice="none",
            messages=[{"role": "system", "content": "s", "tools": [tool]}, {"role": "user", "content": "u"}]))
        self.assertNotIn("tools", encoder.calls[0][0][0])
        with self.assertRaisesRegex(adapter.UpstreamError, "despite tool_choice"):
            service.complete(prepared)

    def test_context_budget_caps_output_and_refuses_full_prompt(self) -> None:
        service, _, upstream = self.make_service([])
        with patch.object(upstream, "tokenize", return_value=[0] * (adapter.CONTEXT_WINDOW - 10)):
            prepared = service.prepare(request(max_tokens=100))
            self.assertEqual(prepared.native_payload["n_predict"], 10)
        with patch.object(upstream, "tokenize", return_value=[0] * adapter.CONTEXT_WINDOW):
            with self.assertRaises(adapter.AdapterError) as raised:
                service.prepare(request())
            self.assertEqual(raised.exception.code, "context_length_exceeded")
        self.assertEqual(upstream.payloads, [])

    def test_reasoning_budget_is_forwarded_and_leaves_answer_room(self) -> None:
        service, _, upstream = self.make_service([])
        prepared = service.prepare(request(
            thinking_token_budget=4096,
            max_tokens=5000,
        ))
        self.assertEqual(prepared.native_payload["reasoning_budget_tokens"], 3976)
        self.assertEqual(prepared.native_payload["n_predict"], 5000)

    def test_reasoning_budget_requires_enabled_thinking(self) -> None:
        service, _, _ = self.make_service([])
        with self.assertRaisesRegex(adapter.AdapterError, "requires thinking.type"):
            service.prepare(request(
                thinking={"type": "disabled"},
                thinking_token_budget=2048,
            ))

    def test_fragmented_thinking_delimiters_and_tool_block_are_safe(self) -> None:
        tool_block = "\n\n<｜DSML｜ calls><｜DSML｜ invoke name=weather>{\"city\":\"Beijing\"}</｜DSML｜ calls>"
        stream = b"".join([
            native_event({"content": "<thi"}),
            native_event({"content": "nk>first thought</th"}),
            native_event({"content": "ink>answer" + tool_block[:10]}),
            native_event({"content": tool_block[10:]}),
            native_event({"content": "", "stop": True, "stop_type": "eos", "tokens_predicted": 9, "tokens_evaluated": 12,
                          "timings": {"cache_n": 8, "prompt_n": 12}}),
        ])
        # Deliberately split in the middle of UTF-8 and of both special tags.
        chunks = [stream[:17], stream[17:63], stream[63:127], stream[127:]]
        parsed = {
            "role": "assistant", "content": "answer",
            "reasoning_content": "first thought",
            "tool_calls": [{"type": "function", "function": {"name": "weather", "arguments": "{\"city\":\"Beijing\"}"}}],
        }
        service, encoder, upstream = self.make_service(chunks, parsed)
        prepared = service.prepare(request(tools=[{"type": "function", "function": {"name": "weather", "parameters": {}}}]))
        emitted: list[tuple[str, str]] = []
        outcome = service.complete(prepared, lambda kind, text: emitted.append((kind, text)))

        self.assertEqual("".join(text for kind, text in emitted if kind == "reasoning_content"), "first thought")
        self.assertEqual("".join(text for kind, text in emitted if kind == "content"), "answer")
        self.assertNotIn("DSML", outcome.content)
        self.assertTrue(encoder.parse_calls[0][0].endswith(adapter.EOS))
        self.assertEqual(encoder.calls[0][1]["thinking_mode"], "thinking")
        self.assertEqual(encoder.calls[0][1]["reasoning_effort"], 100)
        self.assertEqual(encoder.calls[0][0][0]["tools"][0]["function"]["name"], "weather")
        calls_once = adapter.AdapterService.stable_tool_calls(outcome.parsed)
        self.assertEqual(calls_once[0]["id"], adapter.AdapterService.stable_tool_calls(outcome.parsed)[0]["id"])
        self.assertEqual(calls_once[0]["id"][:5], "call_")
        self.assertTrue(upstream.closed)

    def test_usage_and_length_do_not_parse_or_emit_tool_calls(self) -> None:
        chunks = [native_event({
            "content": "partial answer", "stop": True, "stop_type": "limit",
            "tokens_predicted": 44, "tokens_evaluated": 100, "timings": {"cache_n": 70, "prompt_n": 100},
        })]
        service, encoder, _ = self.make_service(chunks)
        prepared = service.prepare(request(thinking={"type": "disabled"}, max_tokens=50))
        outcome = service.complete(prepared)
        self.assertIsNone(outcome.parsed)
        self.assertEqual(adapter.finish_reason(outcome), "length")
        self.assertEqual(outcome.usage, {
            "prompt_tokens": 100, "completion_tokens": 44, "total_tokens": 144,
            "prompt_cache_hit_tokens": 70, "prompt_cache_miss_tokens": 30,
        })
        self.assertEqual(encoder.parse_calls, [])
        self.assertEqual(outcome.content, "partial answer")

    def test_native_error_is_an_explicit_upstream_error(self) -> None:
        service, _, upstream = self.make_service([native_event({"error": {"message": "boom"}})])
        prepared = service.prepare(request())
        with self.assertRaises(adapter.UpstreamError) as raised:
            service.complete(prepared)
        self.assertEqual(raised.exception.code, "upstream_error")
        self.assertTrue(upstream.closed)

    def test_rejects_multimodal_requests_without_contacting_upstream(self) -> None:
        service, _, upstream = self.make_service([])
        bad = request(messages=[{"role": "user", "content": [{"type": "input_text", "text": "x"}, {"type": "image_url", "image_url": "x"}]}])
        with self.assertRaises(adapter.AdapterError) as raised:
            service.prepare(bad)
        self.assertEqual(raised.exception.code, "unsupported_input")
        self.assertEqual(upstream.tokenized, [])

    def test_preserves_tool_history_and_native_payload_contract(self) -> None:
        chunks = [native_event({"content": "<think>x</think>ok", "stop": True, "stop_type": "eos"})]
        service, encoder, upstream = self.make_service(chunks)
        history = [
            {"role": "assistant", "content": "", "reasoning_content": "prior", "tool_calls": [{"id": "call_old", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_old", "content": "result"},
            {"role": "user", "content": "continue"},
        ]
        prepared = service.prepare(request(messages=history, thinking={"type": "disabled"}, stop=["END"], seed=7))
        service.complete(prepared)
        passed_messages, kwargs = encoder.calls[0]
        self.assertEqual(passed_messages, history)
        self.assertTrue(kwargs["drop_thinking"])
        self.assertEqual(kwargs["thinking_mode"], "chat")
        payload = upstream.payloads[0]
        self.assertEqual(payload["prompt"], [128799, 42, 128804])
        self.assertEqual(payload["n_predict"], 262144)
        self.assertEqual(payload["samplers"], ["temperature", "top_p"])
        self.assertFalse(payload["reasoning_control"])
        self.assertEqual(payload["reasoning_budget_tokens"], -1)
        self.assertTrue(payload["cache_prompt"])
        self.assertEqual(payload["stop"], ["END"])
        self.assertEqual(payload["seed"], 7)

    def test_http_sse_sends_tool_delta_finish_and_optional_usage(self) -> None:
        chunks = [native_event({
            "content": "<think>r</think>ok\n\n<｜DSML｜ calls><｜DSML｜ invoke name=ping>{}</｜DSML｜ calls>",
            "stop": True, "stop_type": "eos", "tokens_predicted": 4, "tokens_evaluated": 6,
            "timings": {"cache_n": 5},
        })]
        parsed = {
            "role": "assistant", "content": "ok", "reasoning_content": "r",
            "tool_calls": [{"type": "function", "function": {"name": "ping", "arguments": "{}"}}],
        }
        service, _, _ = self.make_service(chunks, parsed)
        server = adapter.AdapterHTTPServer(("127.0.0.1", 0), service)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            body = json.dumps(request(stream=True, stream_options={"include_usage": True}), ensure_ascii=False).encode("utf-8")
            connection.request("POST", "/v1/chat/completions", body=body, headers={
                "Content-Type": "application/json", "Content-Length": str(len(body)),
            })
            response = connection.getresponse()
            received = b""
            while b"data: [DONE]\n\n" not in received:
                line = response.fp.readline()  # type: ignore[union-attr]
                self.assertTrue(line, "SSE stream ended before [DONE]")
                received += line
            payload = received.decode("utf-8")
            self.assertEqual(response.status, 200)
            self.assertIn('"reasoning_content":"r"', payload)
            self.assertIn('"content":"ok"', payload)
            self.assertIn('"tool_calls"', payload)
            self.assertIn('"finish_reason":"tool_calls"', payload)
            self.assertIn('"prompt_cache_hit_tokens":5', payload)
            self.assertNotIn("DSML", payload)
            self.assertTrue(payload.rstrip().endswith("data: [DONE]"))
            self.assertEqual(response.fp.read(), b"")  # server closes unknown-length SSE body
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
