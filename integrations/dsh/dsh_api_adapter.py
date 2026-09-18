#!/usr/bin/env python3
"""Loopback-only OpenAI-compatible adapter for DeepSeek-V4.1-Flash.

The adapter translates DSH's ``/v1/chat/completions`` requests into the
native llama-server token and completion endpoints.  It deliberately leaves
the model server, its cache configuration, and any benchmark process alone.

Run this on the inference host after the official V4.1 encoder is present::

    python3 dsh_api_adapter.py

It listens only on 127.0.0.1:48242 by default.  A tunnel may forward that
loopback port, but this program must not be exposed directly to a network.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import http.client
import importlib.util
import ipaddress
import json
import secrets
import socket
import sys
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol
from urllib.parse import urlsplit


MODEL_ID = "deepseek-v4.1-flash-local"
DEFAULT_PORT = 48242
DEFAULT_UPSTREAM = "http://127.0.0.1:48241"
DEFAULT_ENCODER = Path("/root/dsv41/capability/encoding.py")
MAX_BODY_BYTES = 32 * 1024 * 1024
CONTEXT_WINDOW = 1048576
MAX_OUTPUT_TOKENS = 262144
MAX_REASONING_BUDGET_TOKENS = MAX_OUTPUT_TOKENS
MIN_ANSWER_TOKENS = 1024
EOS = "<｜end▁of▁sentence｜>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
DSML_CALLS_OPEN = "<｜DSML｜ calls>"


class AdapterError(Exception):
    """A safe error that can be returned to an API client."""

    status = HTTPStatus.BAD_REQUEST
    error_type = "invalid_request_error"
    code = "invalid_request"


class UnknownModel(AdapterError):
    status = HTTPStatus.NOT_FOUND
    code = "model_not_found"


class UpstreamError(AdapterError):
    status = HTTPStatus.BAD_GATEWAY
    error_type = "api_error"
    code = "upstream_error"


class ClientDisconnected(Exception):
    """The downstream client closed its SSE connection."""


class Encoder(Protocol):
    def encode_messages(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any: ...

    def parse_message_from_completion_text(self, text: str, thinking_mode: str) -> dict[str, Any]: ...


class Upstream(Protocol):
    def tokenize(self, encoded: Any) -> list[int]: ...

    def stream_completion(self, payload: Mapping[str, Any]) -> Iterable[bytes]: ...

    def health(self) -> Any: ...


@dataclass(frozen=True)
class PreparedRequest:
    request: dict[str, Any]
    thinking_mode: str
    encoded: Any
    tokens: list[int]
    native_payload: dict[str, Any]


@dataclass(frozen=True)
class CompletionOutcome:
    raw_text: str
    content: str
    reasoning_content: str
    parsed: dict[str, Any] | None
    stop_type: str
    usage: dict[str, int]


def load_official_encoder(path: Path) -> Encoder:
    """Load the local, official V4.1 message encoder without changing sys.path."""
    spec = importlib.util.spec_from_file_location("dsv41_official_encoding", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load encoder from {path}")
    module = importlib.util.module_from_spec(spec)
    # Some encoder implementations use dataclasses or relative imports while
    # defining helpers.  Registering the module follows normal import rules.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    if not hasattr(module, "encode_messages") or not hasattr(module, "parse_message_from_completion_text"):
        raise RuntimeError("The encoder does not expose the required V4.1 functions")
    return module  # type: ignore[return-value]


class NativeUpstream:
    """Small standard-library client for native llama-server endpoints."""

    def __init__(self, address: str, timeout: float | None = None) -> None:
        parsed = urlsplit(address)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("upstream must be an http URL with a host")
        self.host = parsed.hostname
        self.port = parsed.port or 80
        self.base_path = parsed.path.rstrip("/")
        self.timeout = timeout

    def _connection(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def _post_json(self, endpoint: str, value: Mapping[str, Any]) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        connection = self._connection()
        try:
            connection.request(
                "POST", self.base_path + endpoint, body=encoded,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            response = connection.getresponse()
        except OSError as error:
            connection.close()
            raise UpstreamError("Native model server is unavailable") from error
        if response.status >= 400:
            response.close()
            connection.close()
            raise UpstreamError(f"Native model server returned HTTP {response.status}")
        return connection, response

    def tokenize(self, encoded: Any) -> list[int]:
        connection, response = self._post_json(
            "/tokenize", {"content": encoded, "add_special": False, "parse_special": True}
        )
        try:
            try:
                payload = json.loads(response.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise UpstreamError("Native tokenizer returned an invalid response") from error
            tokens = payload.get("tokens") if isinstance(payload, dict) else None
            if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
                raise UpstreamError("Native tokenizer did not return tokens")
            return tokens
        finally:
            response.close()
            connection.close()

    def stream_completion(self, payload: Mapping[str, Any]) -> Iterator[bytes]:
        connection, response = self._post_json("/completion", payload)
        try:
            # ``read1`` is available on HTTPResponse on supported CPython
            # versions and prevents waiting for a large response buffer.
            reader = getattr(response, "read1", response.read)
            while True:
                chunk = reader(4096)
                if not chunk:
                    return
                yield chunk
        finally:
            # Closing both objects promptly cancels a native generation when
            # the DSH client disconnects.
            response.close()
            connection.close()

    def health(self) -> Any:
        connection = self._connection()
        try:
            connection.request("GET", self.base_path + "/health", headers={"Accept": "application/json"})
            response = connection.getresponse()
            if response.status >= 400:
                raise UpstreamError("Native model server health check failed")
            body = response.read()
            if not body:
                return {"status": "ok"}
            try:
                return json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return {"status": "ok"}
        except OSError as error:
            raise UpstreamError("Native model server is unavailable") from error
        finally:
            try:
                response.close()  # type: ignore[has-type]
            except UnboundLocalError:
                pass
            connection.close()


def iter_sse_data(chunks: Iterable[bytes]) -> Iterator[str]:
    """Yield complete SSE data events while accepting arbitrary byte chunks."""
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    buffer = ""
    data_lines: list[str] = []

    def consume(line: str) -> str | None:
        nonlocal data_lines
        if line == "":
            if data_lines:
                event = "\n".join(data_lines)
                data_lines = []
                return event
            return None
        if line.startswith(":"):
            return None
        if line.startswith("data:"):
            payload = line[5:]
            if payload.startswith(" "):
                payload = payload[1:]
            data_lines.append(payload)
        return None

    def accept(text: str) -> Iterator[str]:
        nonlocal buffer
        buffer += text
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            if line.endswith("\r"):
                line = line[:-1]
            event = consume(line)
            if event is not None:
                yield event

    for chunk in chunks:
        if not isinstance(chunk, bytes):
            raise UpstreamError("Native model server returned a non-byte stream")
        yield from accept(decoder.decode(chunk))
    yield from accept(decoder.decode(b"", final=True))
    if buffer:
        event = consume(buffer[:-1] if buffer.endswith("\r") else buffer)
        if event is not None:
            yield event
    if data_lines:
        yield "\n".join(data_lines)


class ToolBlockShield:
    """Withhold a possibly fragmented DSML tool block from text deltas."""

    def __init__(self) -> None:
        self.buffer = ""
        self.holding_tool_block = False

    def feed(self, text: str) -> str:
        if self.holding_tool_block:
            self.buffer += text
            return ""
        value = self.buffer + text
        marker_at = value.find(DSML_CALLS_OPEN)
        if marker_at >= 0:
            # The official grammar prefixes a calls block with two newlines.
            # They are syntax, not assistant text, so hold them with the
            # block rather than leaking an orphaned blank line to DSH.
            if value[max(0, marker_at - 2):marker_at] == "\n\n":
                marker_at -= 2
            self.holding_tool_block = True
            self.buffer = value[marker_at:]
            return value[:marker_at]
        keep = len(DSML_CALLS_OPEN) - 1
        if len(value) <= keep:
            self.buffer = value
            return ""
        self.buffer = value[-keep:]
        return value[:-keep]

    def finish(self) -> str:
        if self.holding_tool_block:
            return ""
        value = self.buffer
        self.buffer = ""
        return value


class ThinkingSplitter:
    """Turn a native stream into safe reasoning/content deltas.

    The native special tokens can straddle arbitrary TCP and SSE chunks.  The
    splitter never emits a partial ``</think>`` or DSML opening marker.
    """

    def __init__(self, thinking_enabled: bool) -> None:
        self.thinking_enabled = thinking_enabled
        self.phase = "start"
        self.start_buffer = ""
        self.reasoning_buffer = ""
        self.tool_shield = ToolBlockShield()

    def _begin(self) -> tuple[str, str]:
        value = self.start_buffer
        if THINK_OPEN.startswith(value) and value != THINK_OPEN:
            return "", ""
        self.start_buffer = ""
        if value.startswith(THINK_OPEN):
            value = value[len(THINK_OPEN):]
            self.phase = "reasoning"
        elif self.thinking_enabled:
            self.phase = "reasoning"
        else:
            self.phase = "content"
        return self._route(value)

    def _route(self, value: str) -> tuple[str, str]:
        reasoning = ""
        content = ""
        if self.phase == "reasoning":
            value = self.reasoning_buffer + value
            end = value.find(THINK_CLOSE)
            if end < 0:
                keep = len(THINK_CLOSE) - 1
                safe, self.reasoning_buffer = value[:-keep], value[-keep:] if len(value) >= keep else value
                if self.thinking_enabled:
                    reasoning += safe
                return reasoning, content
            before = value[:end]
            if self.thinking_enabled:
                reasoning += before
            self.reasoning_buffer = ""
            self.phase = "content"
            value = value[end + len(THINK_CLOSE):]
        if self.phase == "content":
            content += self.tool_shield.feed(value)
        return reasoning, content

    def feed(self, value: str) -> tuple[str, str]:
        if self.phase == "start":
            self.start_buffer += value
            return self._begin()
        return self._route(value)

    def finish(self) -> tuple[str, str]:
        reasoning = ""
        content = ""
        if self.phase == "start":
            value = self.start_buffer
            self.start_buffer = ""
            # At EOF a partial opening marker is ordinary text.  It will not
            # become a malformed special token in a later event.
            if self.thinking_enabled:
                self.phase = "reasoning"
                self.reasoning_buffer += value
            else:
                self.phase = "content"
                content += self.tool_shield.feed(value)
        if self.phase == "reasoning":
            if self.thinking_enabled:
                reasoning += self.reasoning_buffer
            self.reasoning_buffer = ""
        if self.phase == "content":
            content += self.tool_shield.finish()
        return reasoning, content


def _request_error(message: str, code: str = "invalid_request") -> AdapterError:
    error = AdapterError(message)
    error.code = code
    return error


def _is_text_message(message: Mapping[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, (dict, list)):
        return False
    return content is None or isinstance(content, str)


class AdapterService:
    def __init__(self, encoder: Encoder, upstream: Upstream) -> None:
        self.encoder = encoder
        self.upstream = upstream

    def prepare(self, request: Mapping[str, Any]) -> PreparedRequest:
        if not isinstance(request, Mapping):
            raise _request_error("JSON request must be an object")
        value = dict(request)
        model = value.get("model")
        if model != MODEL_ID:
            raise UnknownModel(f"Unknown model: {model!r}")
        messages = value.get("messages")
        if not isinstance(messages, list) or not messages:
            raise _request_error("messages must be a non-empty array")
        if "audio" in value or "input_audio" in value or "file" in value:
            raise _request_error("Audio and file inputs are not supported", "unsupported_input")
        modalities = value.get("modalities")
        if modalities is not None and modalities != ["text"]:
            raise _request_error("Only text input and output are supported", "unsupported_input")
        normal_messages: list[dict[str, Any]] = []
        for position, item in enumerate(messages):
            if not isinstance(item, Mapping):
                raise _request_error(f"messages[{position}] must be an object")
            role = item.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                raise _request_error(f"messages[{position}].role is unsupported")
            if not _is_text_message(item):
                raise _request_error("Image, audio, and file message parts are not supported", "unsupported_input")
            copy = dict(item)
            normal_messages.append(copy)

        tools = value.get("tools")
        if tools is not None:
            if not isinstance(tools, list) or any(
                not isinstance(tool, Mapping)
                or tool.get("type") != "function"
                or not isinstance(tool.get("function"), Mapping)
                or not isinstance(tool["function"].get("name"), str)
                or not tool["function"]["name"]
                for tool in tools
            ):
                raise _request_error("Only OpenAI function tools are supported")
        tool_choice = value.get("tool_choice", "auto")
        if not isinstance(tool_choice, str) or tool_choice not in {"auto", "none"}:
            raise _request_error("Only tool_choice 'auto' and 'none' are supported", "unsupported_tool_choice")
        if tool_choice == "none":
            tools = None
            for message in normal_messages:
                if message["role"] == "system":
                    message.pop("tools", None)
        response_format = value.get("response_format")
        if response_format is not None and not isinstance(response_format, Mapping):
            raise _request_error("response_format must be an object")
        thinking = value.get("thinking", {"type": "disabled"})
        if not isinstance(thinking, Mapping) or thinking.get("type") not in {"enabled", "disabled"}:
            raise _request_error("thinking.type must be 'enabled' or 'disabled'")
        thinking_mode = "thinking" if thinking["type"] == "enabled" else "chat"
        reasoning_effort = value.get("reasoning_effort")
        effort_map = {"low": 50, "high": 75, "max": 100}
        if reasoning_effort is not None and reasoning_effort not in effort_map:
            raise _request_error("reasoning_effort must be 'low', 'high', or 'max'")
        max_tokens = value.get("max_tokens", 262144)
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise _request_error("max_tokens must be a positive integer")
        thinking_budget = value.get("thinking_token_budget")
        if thinking_budget is not None:
            if not isinstance(thinking_budget, int) or isinstance(thinking_budget, bool) or thinking_budget <= 0:
                raise _request_error("thinking_token_budget must be a positive integer")
            if thinking_mode != "thinking":
                raise _request_error("thinking_token_budget requires thinking.type 'enabled'")
            if thinking_budget > MAX_REASONING_BUDGET_TOKENS:
                raise _request_error("thinking_token_budget exceeds the supported limit")
            # The native runtime counts thinking and answer tokens in one
            # n_predict budget. Keep enough room for a final answer even when
            # a caller supplies a smaller max_tokens than DSH normally does.
            thinking_budget = min(
                thinking_budget,
                max(1, max_tokens - MIN_ANSWER_TOKENS),
            )
        stop = value.get("stop")
        if stop is not None and not (isinstance(stop, str) or (isinstance(stop, list) and all(isinstance(item, str) for item in stop))):
            raise _request_error("stop must be a string or an array of strings")
        seed = value.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise _request_error("seed must be an integer")

        # The official encoder expects request-level tool and response-format
        # data on a system message.  Preserve all prior tool history exactly.
        system_index = next((i for i, item in enumerate(normal_messages) if item["role"] == "system"), None)
        if tools is not None or response_format is not None:
            if system_index is None:
                normal_messages.insert(0, {"role": "system", "content": ""})
                system_index = 0
            system = normal_messages[system_index]
            if tools is not None:
                system["tools"] = tools
            if response_format is not None:
                system["response_format"] = dict(response_format)

        encoded = self.encoder.encode_messages(
            normal_messages,
            thinking_mode=thinking_mode,
            drop_thinking=True,
            add_default_bos_token=True,
            reasoning_effort=effort_map.get(reasoning_effort),
        )
        tokens = self.upstream.tokenize(encoded)
        available_output = CONTEXT_WINDOW - len(tokens)
        if available_output <= 0:
            raise _request_error("Prompt exhausts the 1,048,576-token context window", "context_length_exceeded")
        payload: dict[str, Any] = {
            "prompt": tokens,
            "stream": True,
            "n_predict": min(max_tokens, MAX_OUTPUT_TOKENS, available_output),
            "temperature": 1,
            "top_p": 0.95,
            "samplers": ["temperature", "top_p"],
            "top_k": 0,
            "min_p": 0,
            "reasoning_control": False,
            "reasoning_budget_tokens": thinking_budget if thinking_budget is not None else -1,
            "cache_prompt": True,
        }
        if stop is not None:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed
        return PreparedRequest(value, thinking_mode, encoded, tokens, payload)

    @staticmethod
    def _native_event(data: str) -> dict[str, Any]:
        try:
            event = json.loads(data)
        except json.JSONDecodeError as error:
            raise UpstreamError("Native model server emitted invalid SSE JSON") from error
        if not isinstance(event, dict):
            raise UpstreamError("Native model server emitted an invalid SSE event")
        if event.get("error"):
            raise UpstreamError("Native model server reported an error")
        return event

    @staticmethod
    def _usage(event: Mapping[str, Any], fallback_prompt: int) -> dict[str, int]:
        timings = event.get("timings") if isinstance(event.get("timings"), Mapping) else {}
        prompt = event.get("tokens_evaluated", timings.get("prompt_n", fallback_prompt))
        completion = event.get("tokens_predicted", 0)
        cache = timings.get("cache_n", 0)
        if not isinstance(prompt, int) or prompt < 0:
            prompt = fallback_prompt
        if not isinstance(completion, int) or completion < 0:
            completion = 0
        if not isinstance(cache, int) or cache < 0:
            cache = 0
        cache = min(cache, prompt)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_cache_hit_tokens": cache,
            "prompt_cache_miss_tokens": prompt - cache,
        }

    def complete(
        self,
        prepared: PreparedRequest,
        emit: Callable[[str, str], None] | None = None,
    ) -> CompletionOutcome:
        raw_parts: list[str] = []
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        splitter = ThinkingSplitter(prepared.thinking_mode == "thinking")
        terminal: dict[str, Any] | None = None
        stream = self.upstream.stream_completion(prepared.native_payload)
        try:
            for data in iter_sse_data(stream):
                event = self._native_event(data)
                delta = event.get("content", "")
                if delta is None:
                    delta = ""
                if not isinstance(delta, str):
                    raise UpstreamError("Native model server emitted non-text content")
                raw_parts.append(delta)
                reasoning, content = splitter.feed(delta)
                if reasoning:
                    reasoning_parts.append(reasoning)
                    if emit is not None:
                        emit("reasoning_content", reasoning)
                if content:
                    content_parts.append(content)
                    if emit is not None:
                        emit("content", content)
                if event.get("stop") is True:
                    terminal = event
                    break
            if terminal is None:
                raise UpstreamError("Native model server ended without a terminal event")
            reasoning, content = splitter.finish()
            if reasoning:
                reasoning_parts.append(reasoning)
                if emit is not None:
                    emit("reasoning_content", reasoning)
            if content:
                content_parts.append(content)
                if emit is not None:
                    emit("content", content)
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

        stop_type = terminal.get("stop_type", "eos")
        if not isinstance(stop_type, str):
            stop_type = "eos"
        raw_text = "".join(raw_parts)
        parsed: dict[str, Any] | None = None
        if stop_type == "eos":
            parse_text = raw_text if raw_text.endswith(EOS) else raw_text + EOS
            try:
                parsed = self.encoder.parse_message_from_completion_text(parse_text, prepared.thinking_mode)
            except Exception as error:
                raise UpstreamError("Model output could not be parsed as a complete V4.1 response") from error
            if not isinstance(parsed, dict) or parsed.get("role") != "assistant":
                raise UpstreamError("Official response parser returned an invalid assistant message")
            if prepared.request.get("tool_choice") == "none" and parsed.get("tool_calls"):
                raise UpstreamError("Model returned a tool call despite tool_choice='none'")
        return CompletionOutcome(
            raw_text=raw_text,
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts),
            parsed=parsed,
            stop_type=stop_type,
            usage=self._usage(terminal, len(prepared.tokens)),
        )

    @staticmethod
    def stable_tool_calls(parsed: Mapping[str, Any] | None) -> list[dict[str, Any]]:
        if not parsed:
            return []
        raw = parsed.get("tool_calls")
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise UpstreamError("Official response parser returned invalid tool calls")
        calls: list[dict[str, Any]] = []
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise UpstreamError("Official response parser returned invalid tool calls")
            function = item.get("function")
            if not isinstance(function, Mapping) or not isinstance(function.get("name"), str) or not isinstance(function.get("arguments"), str):
                raise UpstreamError("Official response parser returned invalid tool calls")
            identifier = item.get("id")
            if not isinstance(identifier, str) or not identifier:
                canonical = json.dumps({"index": index, "name": function["name"], "arguments": function["arguments"]}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                identifier = "call_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
            calls.append({
                "id": identifier,
                "type": "function",
                "function": {"name": function["name"], "arguments": function["arguments"]},
            })
        return calls


def finish_reason(outcome: CompletionOutcome) -> str:
    if outcome.stop_type == "limit":
        return "length"
    if outcome.stop_type == "eos" and AdapterService.stable_tool_calls(outcome.parsed):
        return "tool_calls"
    return "stop"


def assistant_message(outcome: CompletionOutcome) -> dict[str, Any]:
    if outcome.parsed is not None:
        content = outcome.parsed.get("content", "")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise UpstreamError("Official response parser returned non-text content")
        message: dict[str, Any] = {"role": "assistant", "content": content}
        reasoning = outcome.parsed.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            message["reasoning_content"] = reasoning
        calls = AdapterService.stable_tool_calls(outcome.parsed)
        if calls:
            message["tool_calls"] = calls
        return message
    message = {"role": "assistant", "content": outcome.content}
    if outcome.reasoning_content:
        message["reasoning_content"] = outcome.reasoning_content
    return message


class AdapterHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], service: AdapterService) -> None:
        self.service = service
        super().__init__(address, AdapterHandler)


class AdapterHandler(BaseHTTPRequestHandler):
    server: AdapterHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: Any) -> None:
        # Prompts, tool arguments, and model outputs must never land in logs.
        return

    def _send_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, error: AdapterError) -> None:
        self._send_json(error.status, {"error": {"message": str(error), "type": error.error_type, "code": error.code}})

    def _read_json(self) -> dict[str, Any]:
        header = self.headers.get("Content-Length")
        if header is None:
            raise _request_error("Content-Length is required")
        try:
            length = int(header)
        except ValueError as error:
            raise _request_error("Content-Length is invalid") from error
        if length < 0 or length > MAX_BODY_BYTES:
            error = _request_error("Request body exceeds the 32 MiB limit", "request_too_large")
            error.status = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
            raise error
        try:
            parsed = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _request_error("Request body is not valid JSON") from error
        if not isinstance(parsed, dict):
            raise _request_error("JSON request must be an object")
        return parsed

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models":
            self._send_json(HTTPStatus.OK, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]})
            return
        if self.path == "/health":
            try:
                self.server.service.upstream.health()
            except AdapterError as error:
                self._send_error(error)
            else:
                self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        self._send_error(_request_error("Not found", "not_found"))

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._send_error(_request_error("Not found", "not_found"))
            return
        try:
            request = self._read_json()
            prepared = self.server.service.prepare(request)
            if request.get("stream") is True:
                self._stream(prepared)
            else:
                outcome = self.server.service.complete(prepared)
                created = int(time.time())
                self._send_json(HTTPStatus.OK, {
                    "id": "chatcmpl-" + secrets.token_hex(12), "object": "chat.completion", "created": created,
                    "model": MODEL_ID, "choices": [{"index": 0, "message": assistant_message(outcome), "finish_reason": finish_reason(outcome)}],
                    "usage": outcome.usage,
                })
        except AdapterError as error:
            self._send_error(error)

    def _write_sse(self, value: Mapping[str, Any]) -> None:
        try:
            body = ("data: " + json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n\n").encode("utf-8")
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as error:
            raise ClientDisconnected from error

    def _stream(self, prepared: PreparedRequest) -> None:
        created = int(time.time())
        completion_id = "chatcmpl-" + secrets.token_hex(12)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        # Unknown-length HTTP/1.1 bodies must be framed. Closing after DONE
        # lets clients observe EOF without waiting forever for another chunk.
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        try:
            self._write_sse({
                "id": completion_id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            })
        except ClientDisconnected:
            return

        def emit(kind: str, text: str) -> None:
            self._write_sse({
                "id": completion_id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {kind: text}, "finish_reason": None}],
            })

        try:
            outcome = self.server.service.complete(prepared, emit)
            calls = AdapterService.stable_tool_calls(outcome.parsed)
            if calls:
                self._write_sse({
                    "id": completion_id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {"tool_calls": [dict(call, index=index) for index, call in enumerate(calls)]}, "finish_reason": None}],
                })
            self._write_sse({
                "id": completion_id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason(outcome)}],
            })
            options = prepared.request.get("stream_options")
            if isinstance(options, Mapping) and options.get("include_usage") is True:
                self._write_sse({
                    "id": completion_id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                    "choices": [], "usage": outcome.usage,
                })
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except ClientDisconnected:
            # ``complete`` closes its upstream generator in a finally block.
            return
        except AdapterError as error:
            try:
                self._write_sse({"error": {"message": str(error), "type": error.error_type, "code": error.code}})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except ClientDisconnected:
                return


def loopback_address(value: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("listen address must be a numeric loopback address") from error
    if not address.is_loopback:
        raise argparse.ArgumentTypeError("adapter may listen only on a loopback address")
    # The server class uses IPv4 by default.  Keeping the public default and
    # validation explicit avoids silently exposing an IPv6 wildcard.
    if address.version != 4:
        raise argparse.ArgumentTypeError("this adapter currently accepts an IPv4 loopback address")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=loopback_address, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--upstream", default=DEFAULT_UPSTREAM)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--upstream-timeout", type=float, default=None,
                        help="socket timeout in seconds; omitted means long reasoning runs are not cut off")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.port <= 65535:
        print("adapter: port must be between 1 and 65535", file=sys.stderr)
        return 2
    if args.upstream_timeout is not None and args.upstream_timeout <= 0:
        print("adapter: upstream timeout must be positive", file=sys.stderr)
        return 2
    try:
        encoder = load_official_encoder(args.encoder)
        upstream = NativeUpstream(args.upstream, args.upstream_timeout)
        server = AdapterHTTPServer((args.host, args.port), AdapterService(encoder, upstream))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"adapter: {error}", file=sys.stderr)
        return 2
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
