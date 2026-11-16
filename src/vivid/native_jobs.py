"""Detached, deadline-bound native renders with receipts and atomic publication.

No project engine is required. Commands are argv arrays; {output} is replaced by a
private temporary filename. Declare all source/script dependencies in inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from vivid.errors import VividError
from vivid.memory import sha256_file
from vivid.runtime import operation_budget, remaining, run_process, stage
from vivid.verification import verify_file
from vivid.workspace import Workspace, _read_object, atomic_json, file_lock, managed_path


def estimate_render(
    sample_wall_seconds,
    sample_media_seconds,
    output_duration,
    remaining_seconds=None,
    validation_seconds=30,
):
    values = [sample_wall_seconds, sample_media_seconds, output_duration, validation_seconds]
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
        for v in values
    ):
        raise VividError(
            "invalid_estimate", "Sample/duration/validation times must be finite and positive"
        )
    if remaining_seconds is not None and (
        not math.isfinite(remaining_seconds) or remaining_seconds < 0
    ):
        raise VividError("invalid_estimate", "Remaining time must be finite and nonnegative")
    estimate = sample_wall_seconds / sample_media_seconds * output_duration
    required = estimate * 1.25 + validation_seconds
    return {
        "estimated_render_seconds": estimate,
        "recommended_budget_seconds": required,
        "fits_budget": None if remaining_seconds is None else required <= remaining_seconds,
        "basis": "sample throughput with 25% margin; estimate only",
    }


def _process_identity(pid):
    try:
        # Field 22 is process start time; comm can contain spaces/parentheses.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


class NativeJobs:
    def __init__(self, workspace):
        self.workspace = workspace
        self.root = managed_path(workspace.state / "jobs")
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch(r"native_[a-f0-9]{64}", job_id):
            raise VividError("invalid_job", "Use the job_id returned by start")
        return self.root / (job_id + ".json")

    def _output(self, value):
        path = Path(value)
        path = (
            (self.workspace.root / path).absolute() if not path.is_absolute() else path.absolute()
        )
        path = managed_path(path)
        if not path.is_relative_to(self.workspace.output) or path == self.workspace.output:
            raise VividError(
                "invalid_output", "Managed native jobs publish inside the output directory"
            )
        return path

    def start(
        self,
        command,
        output,
        inputs,
        *,
        estimated_seconds=None,
        validation_seconds=30,
        expected=None,
        retry=False,
    ):
        if (
            not isinstance(command, list)
            or not command
            or len(command) > 4096
            or any(not isinstance(a, str) or not a or "\0" in a for a in command)
            or not any("{output}" in a for a in command)
        ):
            raise VividError("invalid_command", "Provide a nonempty argv array containing {output}")
        if not isinstance(inputs, list) or not inputs or len(inputs) > 256:
            raise VividError("invalid_inputs", "Declare 1–256 source/script input paths")
        if type(retry) is not bool:
            raise VividError("invalid_arguments", "retry must be a boolean")
        destination = self._output(output)
        identities = []
        for value in inputs:
            p = Path(value)
            p = (self.workspace.root / p).resolve() if not p.is_absolute() else p.resolve()
            if not p.is_relative_to(self.workspace.root) or not p.is_file() or p == destination:
                raise VividError(
                    "invalid_inputs",
                    "Input files must exist inside the workspace and differ from output",
                )
            identities.append({"path": str(p), "sha256": sha256_file(p)})
        expected = expected or {}
        from jsonschema import Draft202012Validator

        from vivid.harness_protocol import EXPECTED, obj

        if not isinstance(expected, dict) or set(expected) - set(EXPECTED):
            raise VividError("invalid_expected", "expected accepts verify's flat expected_* fields")
        errors = list(Draft202012Validator(obj(EXPECTED)).iter_errors(expected))
        if errors:
            raise VividError("invalid_expected", errors[0].message[:500])
        if (
            isinstance(validation_seconds, bool)
            or not isinstance(validation_seconds, (int, float))
            or not math.isfinite(validation_seconds)
            or validation_seconds <= 0
        ):
            raise VividError("invalid_estimate", "validation_seconds must be positive and finite")
        if estimated_seconds is not None and (
            isinstance(estimated_seconds, bool)
            or not isinstance(estimated_seconds, (int, float))
            or not math.isfinite(estimated_seconds)
            or estimated_seconds <= 0
        ):
            raise VividError("invalid_estimate", "estimated_seconds must be positive and finite")
        identity = {
            "command": command,
            "output": str(destination),
            "inputs": identities,
            "expected": expected,
        }
        job_id = (
            "native_" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        )
        path = self._path(job_id)
        with file_lock(self.root / ".jobs.lock"):
            old = _read_object(path)
            if old:
                current = self.status(job_id)
                if not retry or current["status"] in {
                    "pending",
                    "running",
                    "completed",
                    "orphaned",
                }:
                    return current
                history = self.root / "history" / f"{job_id}-{old.get('attempt', 1)}.json"
                atomic_json(history, current)
                (self.root / (job_id + ".cancel")).unlink(missing_ok=True)
            for other in self.root.glob("native_*.json"):
                prior = _read_object(other)
                if prior.get("output") == str(destination) and prior.get("status") in {
                    "pending",
                    "running",
                }:
                    status = self.status(prior["job_id"])
                    if status["status"] in {"pending", "running", "orphaned"}:
                        raise VividError(
                            "render_in_progress", "Query/cancel the existing output job", status
                        )
            seconds = remaining(self.workspace.state)
            if (
                estimated_seconds is not None
                and seconds is not None
                and estimated_seconds * 1.25 + validation_seconds > seconds
            ):
                return {
                    "ok": False,
                    "status": "insufficient_budget",
                    "remaining_seconds": seconds,
                    "recommended_budget_seconds": estimated_seconds * 1.25 + validation_seconds,
                }
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(
                f".{destination.stem}-{job_id[-12:]}.partial{destination.suffix}"
            )
            record = {
                **identity,
                "job_id": job_id,
                "status": "pending",
                "temporary": str(temporary),
                "workspace": self.workspace.manifest(),
                "created_at": time.time(),
                "estimated_seconds": estimated_seconds,
                "attempt": old.get("attempt", 0) + 1,
                "validation_seconds": validation_seconds,
                "log_path": str(self.root / (job_id + ".log")),
                "progress_path": str(self.root / (job_id + ".progress")),
            }
            atomic_json(path, record)
            try:
                with Path(record["log_path"]).open("ab") as log:
                    child = subprocess.Popen(
                        [sys.executable, "-m", "vivid.native_jobs", str(path)],
                        cwd=self.workspace.root,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                    )
                record.update(owner_pid=child.pid, process_identity=_process_identity(child.pid))
                atomic_json(path, record)
                threading.Thread(target=child.wait, daemon=True).start()
            except OSError as exc:
                record.update(status="failed", error=str(exc), finished_at=time.time())
                atomic_json(path, record)
            return {**record, "ok": record["status"] != "failed"}

    def status(self, job_id):
        record = _read_object(self._path(job_id))
        if not record:
            raise VividError("unknown_job", "No such native job")
        if record["status"] in {"pending", "running"}:
            pid = record.get("owner_pid")
            identity = _process_identity(pid) if pid else None
            if pid and (identity is None or identity != record.get("process_identity")):
                latest = _read_object(self._path(job_id))
                if latest.get("status") not in {"pending", "running"}:
                    return self.status(job_id)
                encoder = latest.get("encoder_pid")
                encoder_identity = _process_identity(encoder) if encoder else None
                if encoder_identity is not None and encoder_identity == latest.get(
                    "encoder_identity"
                ):
                    return {
                        **latest,
                        "ok": False,
                        "status": "orphaned",
                        "error": "Supervisor exited but native process is alive; cancel before retrying.",
                    }
                # Status is advisory; never overwrite a worker's terminal receipt.
                return {
                    **record,
                    "ok": False,
                    "status": "interrupted",
                    "error": "Worker no longer exists; inspect logs. No automatic replay.",
                }
            progress = {}
            p = Path(record["progress_path"])
            if p.exists():
                with p.open("rb") as stream:
                    stream.seek(max(0, p.stat().st_size - 8192))
                    for line in stream.read().decode(errors="replace").splitlines():
                        if "=" in line:
                            k, v = line.split("=", 1)
                            if k in {"out_time_us", "speed", "frame", "progress"}:
                                progress[k] = v
            record["progress"] = progress
            record["elapsed_seconds"] = time.time() - record.get("started_at", record["created_at"])
            try:
                encoded_seconds = int(progress["out_time_us"]) / 1e6
                speed = float(progress["speed"].rstrip("x"))
                duration = record["expected"].get("expected_duration")
                if speed > 0 and duration is not None:
                    record["estimated_remaining_seconds"] = (
                        max(0, duration - encoded_seconds) / speed
                    )
            except (KeyError, ValueError, TypeError):
                pass
        elif record["status"] == "completed":
            output = Path(record["output"])
            if not output.is_file() or sha256_file(output) != record["output_sha256"]:
                return {**record, "ok": False, "status": "output_changed"}
        return {**record, "ok": record["status"] in {"pending", "running", "completed"}}

    def cancel(self, job_id):
        record = self.status(job_id)
        if record["status"] in {"pending", "running"}:
            atomic_json(self.root / (job_id + ".cancel"), {"requested_at": time.time()})
        elif record["status"] == "orphaned":
            pid = record["encoder_pid"]
            if _process_identity(pid) == record["encoder_identity"]:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            record.update(status="cancelled", finished_at=time.time())
            atomic_json(self._path(job_id), record)
        return self.status(job_id)


def _execute(path):
    # Parent commits the PID identity while holding the same lock.
    with file_lock(path.parent / ".jobs.lock"):
        record = _read_object(path)
        record.update(status="running", started_at=time.time())
        atomic_json(path, record)
    workspace = Workspace.open(**record["workspace"])
    temporary = Path(record["temporary"])
    try:

        def cancelled():
            return path.with_suffix(".cancel").exists()

        with operation_budget(workspace.state, cancelled), stage("native_render"):
            for item in record["inputs"]:
                if sha256_file(Path(item["path"])) != item["sha256"]:
                    raise VividError("input_changed", "A declared input changed before rendering")
            command = [a.replace("{output}", str(temporary)) for a in record["command"]]
            if Path(command[0]).name == "ffmpeg":
                command[1:1] = ["-nostdin", "-y", "-progress", record["progress_path"]]
            with Path(record["log_path"]).open("ab") as log:
                render_started = time.monotonic()

                def started(process):
                    record.update(
                        encoder_pid=process.pid, encoder_identity=_process_identity(process.pid)
                    )
                    atomic_json(path, record)

                process = run_process(
                    command,
                    timeout=remaining(workspace.state) or 86400,
                    cwd=workspace.root,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    on_start=started,
                )
                record["render_elapsed_seconds"] = time.monotonic() - render_started
            if process.returncode:
                raise VividError(
                    "render_failed", f"Native command exited {process.returncode}; inspect log"
                )
            for item in record["inputs"]:
                if sha256_file(Path(item["path"])) != item["sha256"]:
                    raise VividError("input_changed", "A declared input changed during rendering")
            with stage("verification"):
                checked = verify_file(
                    temporary, cache_dir=workspace.state / "verification", **record["expected"]
                )
            if not checked["ok"]:
                raise VividError("invalid_render", "Rendered output failed verification", checked)
            from vivid.runtime import check_budget

            check_budget()
            os.replace(managed_path(temporary), managed_path(Path(record["output"])))
            record.update(
                status="completed",
                output_sha256=checked["checked_sha256"],
                verification=checked,
                finished_at=time.time(),
            )
    except BaseException as exc:
        if isinstance(exc, VividError):
            record["error_details"] = exc.as_dict()["error"]
        record.update(
            status="cancelled" if path.with_suffix(".cancel").exists() else "failed",
            error=str(exc),
            finished_at=time.time(),
        )
    finally:
        temporary.unlink(missing_ok=True)
        record["elapsed_seconds"] = time.time() - record["started_at"]
        atomic_json(path, record)


if __name__ == "__main__":
    _execute(Path(sys.argv[1]).resolve())
