"""One editing deadline shared by the controller, MCP tools and media children."""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import fcntl
import math
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable

from vivid.errors import VividError
from vivid.workspace import _read_object, atomic_json, file_lock

_deadline = contextvars.ContextVar("vivid_deadline", default=None)
_cancelled = contextvars.ContextVar("vivid_cancelled", default=None)
_state = contextvars.ContextVar("vivid_runtime_state", default=None)
_validated = contextvars.ContextVar("vivid_validated_snapshots", default=None)
_stage_outcome = contextvars.ContextVar("vivid_stage_outcome", default=None)


def mark_stage_failure(reason):
    outcome = _stage_outcome.get()
    if outcome is not None:
        outcome.update(status="failed", reason=reason)


_children = set()
_children_lock = threading.Lock()
_stopping = threading.Event()


def _kill(process):
    try:
        if os.getpgid(process.pid) == process.pid:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _stop_children():
    _stopping.set()
    with _children_lock:
        children = list(_children)
    for process in children:
        try:
            _kill(process)
            process.wait(timeout=1)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass


atexit.register(_stop_children)


def remaining(state) -> float | None:
    value = _read_object(state / "runtime_budget.json", maximum=16384).get("deadline")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise VividError("invalid_state", "Editing deadline must be a finite timestamp")
    return max(0.0, value - time.time())


@contextlib.contextmanager
def operation_budget(
    state, cancelled: Callable[[], bool] | None = None, *, enforce_deadline=True, reserve_seconds=0
):
    seconds = remaining(state) if enforce_deadline else None
    if seconds is not None and reserve_seconds > 0 and seconds <= reserve_seconds:
        raise VividError(
            "observation_budget",
            "Source exploration reached the finalization reserve; finish/export and inspect actual output instead",
            {"remaining_seconds": seconds, "reserved_seconds": reserve_seconds},
        )
    deadline = None if seconds is None else time.monotonic() + max(0, seconds - reserve_seconds)
    parent = _deadline.get()
    if parent is not None:
        deadline = parent if deadline is None else min(parent, deadline)
    token = _deadline.set(deadline)
    state_token = _state.set(state)
    validated_token = _validated.set({} if _validated.get() is None else _validated.get())
    cancel_token = _cancelled.set(cancelled if cancelled is not None else _cancelled.get())
    try:
        check_budget()
        yield
    finally:
        _deadline.reset(token)
        _cancelled.reset(cancel_token)
        _state.reset(state_token)
        _validated.reset(validated_token)


def snapshot_matches(path, digest):
    """Hash an archived snapshot once per operation; rehash on any stat change.

    Mutable workspace files never use this cache. It is discarded between tool
    calls, and ctime/inode are included so overwrites cannot reuse an earlier check.
    """
    from vivid.memory import sha256_file

    stat = path.stat()
    key = (
        str(path),
        digest,
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        stat.st_mode,
    )
    validated = _validated.get()
    # Some overlay filesystems coalesce timestamps for rapid same-size writes.
    # Newly created/modified snapshots and writable files must always be hashed.
    cacheable = (
        not (stat.st_mode & 0o222)
        and max(stat.st_mtime_ns, stat.st_ctime_ns) < (time.time() - 1) * 1e9
    )
    if cacheable and validated is not None and key in validated:
        return True
    matches = sha256_file(path) == digest
    if matches and cacheable and validated is not None:
        validated[key] = True
    return matches


@contextlib.contextmanager
def stage(name, **details):
    """Aggregate managed-stage wall time, including failures; nested times overlap."""
    started = time.monotonic()
    status = "completed"
    outcome = {}
    outcome_token = _stage_outcome.set(outcome)
    try:
        yield
    except BaseException:
        status = "failed"
        raise
    finally:
        _stage_outcome.reset(outcome_token)
        status = outcome.get("status", status)
        state = _state.get()
        if state is not None:
            try:
                with file_lock(state / ".performance.lock"):
                    path = state / "performance.json"
                    data = _read_object(path)
                    row = data.setdefault("stages", {}).setdefault(
                        name,
                        {
                            "calls": 0,
                            "failed": 0,
                            "elapsed_seconds": 0,
                        },
                    )
                    row["calls"] += 1
                    row["failed"] += status == "failed"
                    row["elapsed_seconds"] += round(time.monotonic() - started, 6)
                    row["last"] = {"status": status, **details, **outcome}
                    if name == "asr":
                        metric = (
                            "failed_audio_seconds"
                            if status == "failed"
                            else "completed_audio_seconds"
                        )
                        row[metric] = row.get(metric, 0) + details.get("requested_seconds", 0)
                    data["semantics"] = (
                        "Managed wall time only; nested stages overlap. Native shell work is in native logs."
                    )
                    atomic_json(path, data)
            except (OSError, ValueError, VividError):
                pass  # Observability must not change an operation's outcome.


def check_budget() -> None:
    if _stopping.is_set():
        raise VividError("operation_cancelled", "Media runtime is shutting down")
    cancelled = _cancelled.get()
    if cancelled is not None and cancelled():
        raise VividError("operation_cancelled", "Media operation was cancelled")
    deadline = _deadline.get()
    if deadline is not None and time.monotonic() >= deadline:
        raise VividError("time_budget", "Original editing time budget exhausted")


def timeout_seconds(maximum: float) -> float:
    check_budget()
    deadline = _deadline.get()
    return max(0.001, min(maximum, deadline - time.monotonic())) if deadline else maximum


@contextlib.contextmanager
def bounded_lock(path):
    """Cross-process single-flight lock that respects cancellation and deadlines."""
    from vivid.workspace import managed_path

    path = managed_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        while True:
            check_budget()
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(min(0.05, timeout_seconds(0.05)))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run_process(arguments, *, timeout, on_start=None, **kwargs):
    """Drain pipes while checking cancellation; always reap a timed-out child."""
    check_budget()
    deadline = time.monotonic() + timeout_seconds(timeout)
    kwargs.setdefault("start_new_session", True)
    with subprocess.Popen(arguments, **kwargs) as process:
        with _children_lock:
            _children.add(process)
        try:
            if on_start is not None:
                on_start(process)
            while True:
                check_budget()
                seconds = deadline - time.monotonic()
                if seconds <= 0:
                    raise subprocess.TimeoutExpired(arguments, timeout)
                try:
                    stdout, stderr = process.communicate(timeout=min(seconds, 0.25))
                    return subprocess.CompletedProcess(
                        arguments, process.returncode, stdout, stderr
                    )
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            _kill(process)
            process.communicate()
            raise
        finally:
            with _children_lock:
                _children.discard(process)
