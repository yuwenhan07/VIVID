from __future__ import annotations

import json
import os
import queue
import select
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from vivid import __version__
from vivid.errors import VividError
from vivid.workspace import atomic_json

from .base import BackendResult, EventCallback, NativeBackend, Redactor, token_count


class CodexAppServerBackend(NativeBackend):
    """The native Codex agent loop, driven over its documented JSON-RPC transport."""

    name = "codex"

    def command(self, session_id: str | None, segment_dir: Path) -> list[str]:
        from .provider import api_provider_overrides

        config = self.config
        mcp = {
            "command": config.mcp_command[0],
            "args": config.mcp_command[1:],
            "startup_timeout_sec": 60,
            "tool_timeout_sec": config.tool_timeout_seconds,
            "enabled": True,
            "required": True,
            "env_vars": [
                "VIVID_MEDIA_PYTHON",
                "VIVID_WHISPER_MODEL",
                "VIVID_ALLOW_MODEL_DOWNLOAD",
                "VIVID_ASR_BACKEND",
                "VIVID_ASR_THREADS",
                "VIVID_ASR_TIMEOUT",
            ],
        }

        def toml(value: Any) -> str:
            if isinstance(value, dict):
                return "{" + ",".join(f"{key}={toml(item)}" for key, item in value.items()) + "}"
            return json.dumps(value, ensure_ascii=False)

        args = [self.executable(), "app-server", "--stdio"]
        if config.isolate_user_config:
            args.extend(["-c", "mcp_servers=" + toml({"vivid": mcp})])
        else:
            for key, value in mcp.items():
                args.extend(["-c", f"mcp_servers.vivid.{key}={toml(value)}"])
        for key, value in api_provider_overrides(self.environment()).items():
            args.extend(["-c", f"{key}={toml(value)}"])
        args.extend(config.extra_args)
        return args

    def consume(self, event: dict[str, Any], state: Any) -> None:
        raise NotImplementedError("App-server consumes RPC notifications directly")

    def _send(self, message: dict[str, Any]) -> None:
        if not self.process or not self.process.stdin:
            raise VividError("backend_protocol", "Codex app-server stdin is unavailable")
        self._event_file.write(
            json.dumps(
                self._redactor.value(
                    {
                        "type": "vivid.rpc_request",
                        "message": message,
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )
        self._event_file.flush()
        try:
            payload = memoryview((json.dumps(message) + "\n").encode())
            fd = self.process.stdin.fileno()
            while payload:
                remaining = self._deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Codex app-server input exceeded task wall-clock budget")
                _, writable, _ = select.select([], [fd], [], min(0.1, remaining))
                if not writable:
                    continue
                try:
                    written = os.write(fd, payload[:65536])
                except BlockingIOError:
                    continue
                payload = payload[written:]
        except TimeoutError:
            raise
        except (OSError, BrokenPipeError) as exc:
            raise VividError("backend_protocol", "Codex app-server closed its input") from exc

    def _request(self, method: str, params: dict) -> int:
        self._rpc_id += 1
        request_id = self._rpc_id
        self._send({"id": request_id, "method": method, "params": params})
        return request_id

    def _receive(self) -> dict[str, Any]:
        while True:
            if time.monotonic() >= self._deadline:
                raise TimeoutError("Codex app-server exceeded task wall-clock budget")
            try:
                channel, raw = self._queue.get(
                    timeout=min(0.1, max(0.001, self._deadline - time.monotonic()))
                )
            except queue.Empty:
                continue
            if raw is None:
                if channel == "stdout":
                    raise VividError(
                        "backend_protocol", "Codex app-server exited before turn completion"
                    )
                continue
            text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if channel == "stderr":
                self._stderr_file.write(self._redactor.text(text) + "\n")
                self._stderr_file.flush()
                continue
            try:
                message = json.loads(text)
                if not isinstance(message, dict):
                    raise ValueError("non-object RPC message")
            except (ValueError, json.JSONDecodeError):
                message = {"type": "vivid.native_output", "text": text}
            self._event_file.write(
                json.dumps(self._redactor.value(message), ensure_ascii=False) + "\n"
            )
            self._event_file.flush()
            if self._on_event:
                decision = self._on_event(message)
                if decision and self._turn_id and not self._interrupt_sent:
                    self._interrupt_sent = True
                    self._checkpoint_requested = decision in (True, "checkpoint")
                    self._request(
                        "turn/interrupt", {"threadId": self._thread_id, "turnId": self._turn_id}
                    )
            return message

    def _rpc(self, method: str, params: dict) -> dict:
        request_id = self._request(method, params)
        while request_id not in self._responses:
            self._dispatch(self._receive())
        response = self._responses.pop(request_id)
        if "error" in response:
            raise VividError("backend_rpc", f"{method} failed", response["error"])
        result = response.get("result")
        if not isinstance(result, dict):
            raise VividError("backend_rpc", f"{method} returned a non-object result")
        return result

    def _dispatch(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        request_id = message.get("id")
        if method is None and request_id is not None:
            if request_id == self._steer_id:
                self._steer_accepted = "error" not in message
            else:
                self._responses[request_id] = message
            return
        if method is not None and request_id is not None:
            # Automated runs never grant an unexpected approval. Native tools operate
            # within thread approvalPolicy/sandbox; a request outside that policy is denied.
            if method in {
                "item/commandExecution/requestApproval",
                "item/fileChange/requestApproval",
            }:
                self._send({"id": request_id, "result": {"decision": "decline"}})
            elif method == "item/tool/requestUserInput":
                self._send({"id": request_id, "result": {"answers": {}}})
            else:
                self._send(
                    {
                        "id": request_id,
                        "error": {"code": -32601, "message": "Unsupported host request"},
                    }
                )
            return
        params = message.get("params")
        if not isinstance(params, dict):
            return
        if method == "turn/started":
            turn = params.get("turn") or {}
            if params.get("threadId") == self._thread_id:
                self._turn_id = turn.get("id") or self._turn_id
        elif method == "item/completed":
            if params.get("threadId") != self._thread_id or (
                self._turn_id is not None and params.get("turnId") != self._turn_id
            ):
                return
            item = params.get("item") or {}
            if item.get("type") == "agentMessage" and isinstance(item.get("text"), str):
                self._final_text = item["text"]
        elif method == "thread/tokenUsage/updated" and params.get("threadId") == self._thread_id:
            if self._turn_id is not None and params.get("turnId") != self._turn_id:
                return
            self._tokens(params)
        elif method == "turn/completed" and params.get("threadId") == self._thread_id:
            turn = params.get("turn") or {}
            if self._turn_id is None or turn.get("id") == self._turn_id:
                self._completed_turn = turn
        elif method == "error":
            error = params.get("error") or params
            # Recoverable provider events can be followed by a successful native retry.
            self._last_native_error = str(error.get("message", error))

    @staticmethod
    def _breakdown(value: dict) -> dict[str, int]:
        return {
            "input_tokens": token_count(value.get("inputTokens")),
            "output_tokens": token_count(value.get("outputTokens")),
            "cached_input_tokens": token_count(value.get("cachedInputTokens")),
            "reasoning_output_tokens": token_count(value.get("reasoningOutputTokens")),
            "cache_write_input_tokens": token_count(value.get("cacheWriteInputTokens")),
        }

    def _tokens(self, params: dict) -> None:
        value = params.get("tokenUsage") or {}
        total = self._breakdown(value.get("total") or {})
        last_raw = value.get("last") or {}
        last = self._breakdown(last_raw)
        if self._baseline is None:
            # For an externally resumed thread with no VIVID billing snapshot, the
            # first active notification supplies both cumulative and last-call counts.
            self._baseline = (
                total
                if self._turn_id is None
                else {key: max(0, amount - last[key]) for key, amount in total.items()}
            )
        self._total_usage = total
        self._usage = {
            key: max(0, amount - self._baseline.get(key, 0)) for key, amount in total.items()
        }
        if self._turn_id is None:
            return
        occupancy = (
            token_count(last_raw.get("totalTokens")) or last["input_tokens"] + last["output_tokens"]
        )
        self._peak_context_tokens = max(self._peak_context_tokens, occupancy)
        limit = self.config.context_token_limit
        if (
            limit
            and occupancy >= limit
            and not self._steer_requested
            and self._turn_id
            and self._completed_turn is None
        ):
            self._steer_requested = True
            self._checkpoint_requested = True
            self._steer_id = self._request(
                "turn/steer",
                {
                    "threadId": self._thread_id,
                    "expectedTurnId": self._turn_id,
                    "input": [
                        {
                            "type": "text",
                            "text": (
                                "The VIVID context handoff threshold has been reached. Finish any current tool "
                                "operation safely. Update GUIDE.md and WORKING.md using native file tools "
                                "to preserve current editing state, pending work, running jobs and evidence "
                                "references. End this turn with VIVID_CHECKPOINT so a fresh context can continue."
                            ),
                        }
                    ],
                },
            )

    def _run(
        self,
        prompt: str,
        session_id: str | None,
        on_event: EventCallback | None,
    ) -> BackendResult:
        started = time.monotonic()
        segment_dir = self.config.log_dir / f"codex-app-server-{uuid.uuid4().hex}"
        segment_dir.mkdir(parents=True, mode=0o700)
        self._redactor = Redactor(self.environment())
        self._on_event = on_event
        self._queue: queue.Queue = queue.Queue(maxsize=64)
        self._deadline = started + self.config.timeout_seconds
        self._rpc_id = 0
        self._responses: dict = {}
        self._thread_id = session_id
        self._turn_id: str | None = None
        self._completed_turn: dict | None = None
        self._steer_id: int | None = None
        self._steer_requested = False
        self._steer_accepted = False
        self._interrupt_sent = False
        self._checkpoint_requested = False
        self._peak_context_tokens = 0
        self._final_text = ""
        self._last_native_error: str | None = None
        self._baseline: dict | None = None if session_id else self._breakdown({})
        self._usage = self._breakdown({})
        self._total_usage = None
        totals_path = self.config.log_dir / "thread-token-totals.json"
        try:
            totals = json.loads(totals_path.read_text()) if totals_path.is_file() else {}
            if session_id and isinstance(totals.get(session_id), dict):
                self._baseline = totals[session_id]
        except (OSError, json.JSONDecodeError):
            totals = {}
        command = self.command(session_id, segment_dir)
        atomic_json(
            segment_dir / "invocation.json",
            self._redactor.value(
                {
                    "backend": "codex",
                    "transport": "app-server",
                    "argv": command,
                    "model": self.config.model,
                    "workspace": str(self.config.workspace),
                    "session_id": session_id,
                    "prompt": prompt,
                    "context_measurement": "last_call_total_tokens",
                }
            ),
        )
        events_path = segment_dir / "events.jsonl"
        stderr_path = segment_dir / "stderr.log"
        native_version = self.version()
        status, error = "failed", None
        readers = []
        reader_stop = threading.Event()
        with (
            events_path.open("w", encoding="utf-8") as self._event_file,
            stderr_path.open("w", encoding="utf-8") as self._stderr_file,
        ):
            events_path.chmod(0o600)
            stderr_path.chmod(0o600)
            try:
                self.process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=self.config.workspace,
                    env=self.environment(),
                    start_new_session=True,
                )
                os.set_blocking(self.process.stdin.fileno(), False)
                for name, pipe in (
                    ("stdout", self.process.stdout),
                    ("stderr", self.process.stderr),
                ):
                    reader = threading.Thread(
                        target=self._reader,
                        args=(pipe, name, self._queue, reader_stop),
                        daemon=True,
                    )
                    reader.start()
                    readers.append(reader)
                self._rpc(
                    "initialize",
                    {
                        "clientInfo": {
                            "name": "vivid",
                            "title": "VIVID Video Editing",
                            "version": __version__,
                        }
                    },
                )
                self._send({"method": "initialized", "params": {}})
                thread_params: dict[str, Any] = {
                    "model": self.config.model
                    if "/" in self.config.model
                    else f"openai/{self.config.model}",
                    "cwd": str(self.config.workspace),
                    "approvalPolicy": "never",
                    "sandbox": self.config.codex_sandbox,
                }
                if self.config.effort:
                    thread_params["config"] = {"model_reasoning_effort": self.config.effort}
                if session_id:
                    thread_params.update(threadId=session_id, excludeTurns=True)
                    response = self._rpc("thread/resume", thread_params)
                else:
                    response = self._rpc("thread/start", thread_params)
                self._thread_id = (response.get("thread") or {}).get("id")
                if not self._thread_id:
                    raise VividError(
                        "backend_protocol", "thread/start or resume returned no thread ID"
                    )
                turn = self._rpc(
                    "turn/start",
                    {
                        "threadId": self._thread_id,
                        "input": [{"type": "text", "text": prompt}],
                    },
                )
                self._turn_id = (turn.get("turn") or {}).get("id") or self._turn_id
                if not self._turn_id:
                    raise VividError("backend_protocol", "turn/start returned no turn ID")
                while self._completed_turn is None:
                    self._dispatch(self._receive())
                native_status = self._completed_turn.get("status")
                if native_status == "completed":
                    status = "checkpoint" if self._checkpoint_requested else "completed"
                elif native_status == "interrupted" and self._checkpoint_requested:
                    status = "checkpoint"
                else:
                    turn_error = self._completed_turn.get("error") or {}
                    error = str(
                        turn_error.get("message")
                        or self._last_native_error
                        or f"Native turn {native_status}"
                    )
            except TimeoutError as exc:
                status, error = "timeout", str(exc)
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                error = self._redactor.text(f"{type(exc).__name__}: {exc}")
                if isinstance(exc, VividError) and exc.details:
                    error += ": " + self._redactor.text(json.dumps(exc.details))
            finally:
                if self._total_usage is not None and self._thread_id:
                    totals[self._thread_id] = self._total_usage
                    try:
                        atomic_json(totals_path, totals)
                    except OSError as exc:
                        status = "failed"
                        error = error or f"Failed to persist billing baseline: {exc}"
                reader_stop.set()
                if self.process:
                    try:
                        self.process.stdin.close()
                    except OSError:
                        pass
                    try:
                        self.process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        self._stop_group(self.process, signal.SIGTERM)
                        try:
                            self.process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            self._stop_group(self.process, signal.SIGKILL)
                            self.process.wait(timeout=3)
                    self._stop_group(self.process, signal.SIGTERM)
                for reader in readers:
                    reader.join(timeout=0.5)
        result = BackendResult(
            self._thread_id,
            status,
            self._redactor.text(self._final_text),
            self._usage,
            self.process.returncode if self.process else -1,
            events_path,
            self._redactor.text(error) if error else None,
            self._peak_context_tokens,
            time.monotonic() - started,
            stderr_path,
            native_version,
            context_measurement="last_call_total_tokens",
            usage_complete=status in {"completed", "checkpoint"},
        )
        atomic_json(segment_dir / "result.json", self._redactor.value(result.as_dict()))
        return result
