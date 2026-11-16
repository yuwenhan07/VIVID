from __future__ import annotations

import json
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from vivid import DEFAULT_TIMEOUT_SECONDS
from vivid.errors import VividError

EventCallback = Callable[[dict[str, Any]], str | bool | None]


@dataclass(frozen=True)
class BackendConfig:
    backend: str
    model: str
    workspace: Path
    mcp_command: list[str]
    log_dir: Path
    effort: str | None = None
    executable: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    context_token_limit: int | None = None
    codex_sandbox: str = "workspace-write"
    claude_permission_mode: str = "acceptEdits"
    env: dict[str, str] = field(default_factory=dict)
    extra_args: tuple[str, ...] = ()
    max_turns: int | None = None
    tool_timeout_seconds: int = 600
    isolate_user_config: bool = True
    codex_transport: str = "exec"

    def __post_init__(self) -> None:
        if self.codex_transport not in {"exec", "app-server"}:
            raise ValueError("codex_transport must be exec or app-server")
        if self.backend not in {"codex", "claude", "opencode"}:
            raise ValueError("backend must be codex, claude or opencode")
        if not self.model or not self.model.strip():
            raise ValueError("A native model identifier is required")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        if self.context_token_limit is not None and self.context_token_limit < 1:
            raise ValueError("context_token_limit must be positive")
        if self.max_turns is not None and self.max_turns < 1:
            raise ValueError("max_turns must be positive")
        if self.tool_timeout_seconds < 1:
            raise ValueError("tool_timeout_seconds must be positive")
        if self.codex_sandbox not in {"read-only", "workspace-write", "danger-full-access"}:
            raise ValueError("Unknown Codex sandbox mode")
        if self.claude_permission_mode not in {
            "default",
            "acceptEdits",
            "plan",
            "auto",
            "dontAsk",
            "bypassPermissions",
        }:
            raise ValueError("Unknown Claude Code permission mode")
        if not self.mcp_command or any(not isinstance(p, str) or not p for p in self.mcp_command):
            raise ValueError("mcp_command must be a nonempty argv array")
        object.__setattr__(self, "workspace", Path(self.workspace).resolve())
        object.__setattr__(self, "log_dir", Path(self.log_dir).resolve())
        object.__setattr__(self, "mcp_command", list(self.mcp_command))
        object.__setattr__(self, "env", dict(self.env))


@dataclass
class BackendResult:
    session_id: str | None
    status: str
    final_text: str
    usage: dict[str, int]
    returncode: int
    events_path: Path
    error: str | None = None
    peak_context_tokens: int = 0
    elapsed_seconds: float = 0
    stderr_path: Path | None = None
    native_version: str | None = None
    cost_usd: float | None = None
    context_measurement: str = "unknown"
    usage_complete: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "final_text": self.final_text,
            "usage": self.usage,
            "returncode": self.returncode,
            "events_path": str(self.events_path),
            "error": self.error,
            "peak_context_tokens": self.peak_context_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            "stderr_path": str(self.stderr_path) if self.stderr_path else None,
            "native_version": self.native_version,
            "cost_usd": self.cost_usd,
            "context_measurement": self.context_measurement,
            "usage_complete": self.usage_complete,
        }


class Redactor:
    def __init__(self, env: dict[str, str]):
        self.secrets = sorted(
            {
                v
                for k, v in env.items()
                if len(v) >= 8
                and any(
                    marker in k.upper() for marker in ("TOKEN", "KEY", "SECRET", "PASSWORD", "AUTH")
                )
            },
            key=len,
            reverse=True,
        )

    def text(self, value: str) -> str:
        for secret in self.secrets:
            value = value.replace(secret, "<redacted>")
        value = re.sub(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]{12,}", r"\1<redacted>", value)
        value = re.sub(r"(?i)(https?://)[^/\s@]+@", r"\1<redacted>@", value)
        value = re.sub(
            r"(?i)([?&](?:api[_-]?key|key|access[_-]?token|token|password|secret|signature|credential)=)[^&#\s\"<>]+",
            r"\1<redacted>",
            value,
        )
        return re.sub(r"\bsk-[A-Za-z0-9_-]{16,}", "<redacted>", value)

    def value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.value(v) for v in value]
        if isinstance(value, dict):
            return {k: self.value(v) for k, v in value.items()}
        return value


def token_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value)) if math.isfinite(value) else 0


@dataclass
class StreamState:
    session_id: str | None = None
    final_text: str = ""
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
        }
    )
    peak_context_tokens: int = 0
    completed: bool = False
    failed: bool = False
    error: str | None = None
    message_ids: set[str] = field(default_factory=set)
    cost_usd: float | None = None


class NativeBackend:
    name: str

    def __init__(self, config: BackendConfig):
        self.config = config
        self.process: subprocess.Popen[bytes] | None = None
        self._run_lock = threading.Lock()

    def command(self, session_id: str | None, segment_dir: Path) -> list[str]:
        raise NotImplementedError

    def consume(self, event: dict[str, Any], state: StreamState) -> None:
        raise NotImplementedError

    def environment(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.config.env)
        if self.config.executable and Path(self.config.executable).is_absolute():
            env["PATH"] = (
                str(Path(self.config.executable).parent) + os.pathsep + env.get("PATH", "")
            )
        env.setdefault("NO_COLOR", "1")
        return env

    def executable(self) -> str:
        requested = self.config.executable or self.name
        found = shutil.which(requested, path=self.environment().get("PATH"))
        if not found:
            raise VividError("backend_unavailable", f"Native executable not found: {requested}")
        return found

    def version(self) -> str | None:
        try:
            result = subprocess.run(
                [self.executable(), "--version"],
                capture_output=True,
                text=True,
                timeout=min(10, self.config.timeout_seconds),
                env=self.environment(),
                check=False,
            )
            if result.returncode == 0:
                return Redactor(self.environment()).text(result.stdout.strip()[:500])
        except (OSError, subprocess.SubprocessError, VividError):
            pass
        return None

    def interrupt(self) -> None:
        if self.process and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    @staticmethod
    def _reader(
        stream: Any,
        channel: str,
        events: queue.Queue,
        stop: threading.Event | None = None,
    ) -> None:
        def enqueue(value: Any) -> bool:
            while stop is None or not stop.is_set():
                try:
                    events.put((channel, value), timeout=0.1)
                    return True
                except queue.Full:
                    pass
            return False

        try:
            while stop is None or not stop.is_set():
                line = stream.readline(64 * 1024 * 1024)
                if not line or not enqueue(line):
                    break
        finally:
            stream.close()
            enqueue(None)

    @staticmethod
    def _stop_group(process: subprocess.Popen, sig: signal.Signals) -> None:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    def run(
        self,
        prompt: str,
        session_id: str | None = None,
        on_event: EventCallback | None = None,
    ) -> BackendResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise VividError("invalid_prompt", "The prompt must be nonempty text")
        if not self.config.workspace.is_dir():
            raise VividError(
                "workspace_missing", f"Workspace does not exist: {self.config.workspace}"
            )
        if not self._run_lock.acquire(blocking=False):
            raise VividError("backend_busy", "This backend is already running")
        try:
            return self._run(prompt, session_id, on_event)
        finally:
            self.process = None
            self._run_lock.release()

    def _run(
        self,
        prompt: str,
        session_id: str | None,
        on_event: EventCallback | None,
    ) -> BackendResult:
        started = time.monotonic()
        segment_dir = self.config.log_dir / f"{self.name}-{uuid.uuid4().hex}"
        segment_dir.mkdir(parents=True, mode=0o700)
        events_path = segment_dir / "events.jsonl"
        stderr_path = segment_dir / "stderr.log"
        env = self.environment()
        redactor = Redactor(env)
        state = StreamState(session_id=session_id)
        command = self.command(session_id, segment_dir)
        manifest = {
            "backend": self.name,
            "model": self.config.model,
            "effort": self.config.effort,
            "session_id": session_id,
            "argv": command,
            "workspace": str(self.config.workspace),
            "timeout_seconds": self.config.timeout_seconds,
            "prompt": prompt,
        }
        manifest_path = segment_dir / "invocation.json"
        manifest_path.write_text(json.dumps(redactor.value(manifest), indent=2), encoding="utf-8")
        manifest_path.chmod(0o600)
        native_version = self.version()
        stream_queue: queue.Queue = queue.Queue(maxsize=64)
        stop_reason: str | None = None
        stop_started: float | None = None
        error: str | None = None
        returncode = -1
        readers: list[threading.Thread] = []
        reader_stop = threading.Event()
        with (
            events_path.open("w", encoding="utf-8") as event_file,
            stderr_path.open("w", encoding="utf-8") as stderr_file,
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
                    env=env,
                    start_new_session=True,
                )
                for name, pipe in (
                    ("stdout", self.process.stdout),
                    ("stderr", self.process.stderr),
                ):
                    reader = threading.Thread(
                        target=self._reader,
                        args=(pipe, name, stream_queue, reader_stop),
                        daemon=True,
                    )
                    reader.start()
                    readers.append(reader)

                # A writer thread prevents a child that never reads stdin from blocking the timeout.
                def send_prompt() -> None:
                    try:
                        self.process.stdin.write((prompt + "\n").encode("utf-8"))
                        self.process.stdin.flush()
                    except (BrokenPipeError, OSError):
                        pass
                    finally:
                        try:
                            self.process.stdin.close()
                        except OSError:
                            pass

                writer = threading.Thread(target=send_prompt, daemon=True)
                writer.start()
                closed: set[str] = set()
                exited_at: float | None = None
                while len(closed) < 2:
                    now = time.monotonic()
                    if self.process.poll() is not None:
                        exited_at = exited_at or now
                        if now - exited_at > 1:
                            self._stop_group(self.process, signal.SIGTERM)
                        if now - exited_at > 2:
                            self._stop_group(self.process, signal.SIGKILL)
                    if not stop_reason and now - started >= self.config.timeout_seconds:
                        stop_reason, stop_started = "timeout", now
                        self._stop_group(self.process, signal.SIGTERM)
                    if stop_started is not None and now - stop_started > 3:
                        self._stop_group(self.process, signal.SIGKILL)
                    try:
                        channel, line = stream_queue.get(timeout=0.1)
                    except queue.Empty:
                        # Grandchildren may inherit pipes after the CLI has already exited.
                        if (
                            self.process.poll() is not None
                            and now - started > self.config.timeout_seconds
                        ):
                            self._stop_group(self.process, signal.SIGKILL)
                        continue
                    if line is None:
                        closed.add(channel)
                        continue
                    text = line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if channel == "stderr":
                        stderr_file.write(redactor.text(text) + "\n")
                        stderr_file.flush()
                        continue
                    try:
                        event = json.loads(text)
                        if not isinstance(event, dict):
                            raise ValueError("non-object native event")
                    except (ValueError, json.JSONDecodeError):
                        event = {"type": "vivid.native_output", "text": text}
                    event_file.write(json.dumps(redactor.value(event), ensure_ascii=False) + "\n")
                    event_file.flush()
                    self.consume(event, state)
                    if on_event:
                        decision = on_event(event)
                        if decision and not stop_reason:
                            stop_reason = (
                                "checkpoint" if decision in (True, "checkpoint") else "interrupted"
                            )
                            stop_started = time.monotonic()
                            self._stop_group(self.process, signal.SIGINT)
                self.process.wait(timeout=5)
                returncode = self.process.returncode
                writer.join(timeout=1)
            except BaseException as exc:
                if self.process:
                    self._stop_group(self.process, signal.SIGKILL)
                    try:
                        self.process.wait(timeout=5)
                        returncode = self.process.returncode
                    except subprocess.TimeoutExpired:
                        pass
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                error = redactor.text(f"{type(exc).__name__}: {exc}")
            finally:
                reader_stop.set()
                for reader in readers:
                    reader.join(timeout=1)
        status = "completed"
        if stop_reason in {"timeout", "checkpoint"}:
            status = stop_reason
        elif stop_reason or error or returncode != 0 or state.failed or not state.completed:
            status = "failed"
        elif self.config.context_token_limit and (
            state.peak_context_tokens >= self.config.context_token_limit
        ):
            # Native CLIs compact internally. A harness rollover is requested only after
            # native completion; never terminate a render midway merely to count tokens.
            status = "checkpoint"
        error = error or state.error
        if status == "failed" and not error:
            error = f"Native {self.name} exited {returncode} without a successful completion event"
        if status == "timeout":
            error = f"Native {self.name} exceeded {self.config.timeout_seconds:g} seconds"
        result = BackendResult(
            state.session_id,
            status,
            redactor.text(state.final_text),
            state.usage,
            returncode,
            events_path,
            redactor.text(error) if error else None,
            state.peak_context_tokens,
            time.monotonic() - started,
            stderr_path,
            native_version,
            state.cost_usd,
            "aggregate_input_proxy" if self.name == "codex" else "assistant_message_input",
            status in {"completed", "checkpoint"} and state.completed,
        )
        (segment_dir / "result.json").write_text(
            json.dumps(redactor.value(result.as_dict()), indent=2),
            encoding="utf-8",
        )
        return result
