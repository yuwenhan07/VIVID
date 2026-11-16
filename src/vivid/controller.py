from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

from vivid import DEFAULT_TIMEOUT_SECONDS
from vivid.backends import BackendConfig, create_backend
from vivid.errors import VividError
from vivid.notes import RecoveryNotes
from vivid.prompts import task_prompt
from vivid.workspace import Workspace, _read_object, atomic_json, file_lock


@dataclass
class RunConfig:
    workspace: Workspace
    instruction: str
    backend: str
    model: str
    effort: str | None = None
    executable: str | None = None
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    max_segments: int = 16
    context_token_limit: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int = 10000
    required_outputs: tuple[str, ...] = ()
    codex_sandbox: str = "workspace-write"
    codex_transport: str = "exec"
    claude_permission_mode: str = "acceptEdits"

    def __post_init__(self):
        if self.backend not in {"codex", "claude", "opencode"} or not self.model.strip():
            raise VividError(
                "invalid_backend", "A supported backend and explicit model are required"
            )
        if not self.instruction.strip():
            raise VividError("empty_instruction", "Task instruction is empty")
        if (
            not math.isfinite(self.timeout)
            or self.timeout <= 0
            or self.max_segments < 1
            or self.max_tool_calls < 1
        ):
            raise VividError("invalid_budget", "Time, segment and tool budgets must be positive")
        for value in (self.max_input_tokens, self.max_output_tokens, self.context_token_limit):
            if value is not None and value < 1:
                raise VividError("invalid_budget", "Token limits must be positive")


class Controller:
    def __init__(self, config: RunConfig, backend_factory: Callable = create_backend):
        self.config = config
        self.workspace = config.workspace
        self.backend_factory = backend_factory
        self.notes = RecoveryNotes(self.workspace)

    def required_outputs(self) -> list[str]:
        if self.config.required_outputs:
            names = list(self.config.required_outputs)
        else:
            names = re.findall(r"/workspace/output/([A-Za-z0-9_./-]+)", self.config.instruction)
            names += re.findall(
                re.escape(str(self.workspace.output)) + r"/([A-Za-z0-9_./-]+)",
                self.config.instruction,
            )
            names += re.findall(
                r"(?<![A-Za-z0-9_./-])output/([A-Za-z0-9_./-]+)", self.config.instruction
            )
        result = []
        for name in names:
            name = name.rstrip(".,")
            path = (self.workspace.output / name).resolve()
            if not path.is_relative_to(self.workspace.output) or path == self.workspace.output:
                raise VividError(
                    "invalid_deliverable", "Deliverable is outside the output directory"
                )
            if name not in result:
                result.append(name)
        return result

    def budget(self, state: dict) -> dict:
        return {
            "remaining_seconds": max(0, self.config.timeout - state["elapsed_seconds"]),
            "segments_remaining": max(
                0, self.config.max_segments - state.get("segment_attempts", len(state["segments"]))
            ),
            "usage": state["usage"],
            "max_input_tokens": self.config.max_input_tokens,
            "max_output_tokens": self.config.max_output_tokens,
            "context_token_limit": self.config.context_token_limit,
            "required_outputs": self.required_outputs(),
        }

    def policy(self) -> dict[str, Any]:
        config = self.config
        return {
            "timeout": config.timeout,
            "max_segments": config.max_segments,
            "max_tool_calls": config.max_tool_calls,
            "max_input_tokens": config.max_input_tokens,
            "max_output_tokens": config.max_output_tokens,
            "context_token_limit": config.context_token_limit,
            "codex_transport": config.codex_transport,
            "codex_sandbox": config.codex_sandbox,
            "claude_permission_mode": config.claude_permission_mode,
            "required_outputs": list(config.required_outputs),
            "executable": config.executable,
        }

    def _complete(
        self, expected_hashes: dict[str, str] | None = None
    ) -> tuple[bool, list[str], str | None]:
        from vivid.runtime import check_budget, operation_budget
        from vivid.verification import verify_file

        required = self.required_outputs()
        if not required:
            return (
                False,
                [],
                "No explicit deliverable paths were supplied; native completion cannot establish delivery.",
            )
        outputs = []
        self._checked_hashes = {}
        for name in required:
            path = self.workspace.output / name
            if not path.resolve().is_relative_to(self.workspace.output):
                return False, [], f"Deliverable escaped the output directory: {name}"
            try:
                with operation_budget(self.workspace.state):
                    result = verify_file(
                        path,
                        expected_sha256=(expected_hashes or {}).get(str(path)),
                        cache_dir=self.workspace.state / "verification",
                    )
                    check_budget()
            except (VividError, OSError) as exc:
                return False, [], str(exc)
            if not result.get("ok"):
                return False, [], json.dumps(result, ensure_ascii=False)
            self._checked_hashes[str(path)] = result["checked_sha256"]
            outputs.append(str(path))
        return True, outputs, None

    def _accept_outputs(self, state: dict, outputs: list[str]) -> None:
        state.update(
            status="completed",
            outputs=outputs,
            output_hashes=dict(self._checked_hashes),
            completion={
                "scope": "explicit_deliverables_deterministic_checks",
                "semantic_success": None,
            },
        )
        state.pop("continuation", None)

    def _completion_budget_error(self, state: dict) -> str | None:
        if state["elapsed_seconds"] >= self.config.timeout:
            return "Cumulative task time budget reached"
        if state.get("unmeasured_attempts") and (
            self.config.max_input_tokens is not None or self.config.max_output_tokens is not None
        ):
            return "Interrupted native usage is unknown under a finite token budget"
        for key, limit in (
            ("input_tokens", self.config.max_input_tokens),
            ("output_tokens", self.config.max_output_tokens),
        ):
            if limit is not None and state["usage"][key] > limit:
                return "Cumulative task token budget exceeded"
        return None

    @staticmethod
    def _add_result(state: dict, result: dict) -> None:
        state["segments"].append(result)
        if not result.get("usage_complete", True):
            state["unmeasured_attempts"] = state.get("unmeasured_attempts", 0) + 1
        for key in state["usage"]:
            state["usage"][key] += result.get("usage", {}).get(key, 0)
        state["usage_complete"] = not state.get("unmeasured_attempts") and all(
            s.get("usage_complete", True) for s in state["segments"]
        )
        costs = [segment.get("cost_usd") for segment in state["segments"]]
        state["cost_usd"] = sum(costs) if all(isinstance(c, (int, float)) for c in costs) else None
        state["session_id"] = result.get("session_id")

    def _recover_inflight(self, state: dict) -> bool:
        """Charge a reserved native attempt even when the controller died abruptly."""
        inflight = state.pop("inflight", None)
        if not inflight:
            return False
        recovered = None
        known = {s.get("events_path") for s in state["segments"]}
        for path in (self.workspace.state / "native_logs").glob("*/result.json"):
            if str(path.parent) in inflight.get("existing_native_directories", []):
                continue
            if "existing_results" in inflight:
                if str(path) in inflight["existing_results"]:
                    continue
            elif path.stat().st_mtime < inflight["started_at"] - 1:
                continue
            try:
                result = _read_object(path)
                if result.get("events_path") not in known and result.get("status"):
                    recovered = result
                    break
            except (OSError, ValueError, AttributeError, VividError):
                continue
        if recovered is not None:
            self._add_result(state, recovered)
            charge = max(0, recovered.get("elapsed_seconds", 0))
        else:
            charge = min(
                max(0, time.time() - inflight["started_at"]), inflight["remaining_seconds"]
            )
            state["unmeasured_attempts"] = state.get("unmeasured_attempts", 0) + 1
            state["usage_complete"] = False
        state["elapsed_seconds"] += charge
        # A partially executed native turn is never replayed as a normal resumed turn.
        state["session_id"] = None
        state["continuation"] = (
            "The prior native attempt was interrupted. Reconcile actual files, archived evidence and incomplete native job logs before continuing; do not replay work automatically."
        )
        return recovered is not None and recovered["status"] == "completed"

    def run(self) -> dict[str, Any]:
        with file_lock(self.workspace.state / ".controller.lock"):
            return self._run()

    def _run(self) -> dict[str, Any]:
        from vivid.harness import HarnessMemory
        from vivid.memory import MemoryStore
        from vivid.runtime import check_budget, operation_budget

        config = self.config
        start = time.monotonic()
        run_path = self.workspace.state / "run.json"
        existing = run_path.exists()
        state = (
            _read_object(run_path)
            if existing
            else {
                "schema_version": 1,
                "identity": None,
                "backend": config.backend,
                "model": config.model,
                "effort": config.effort,
                "status": "running",
                "session_id": None,
                "segments": [],
                "segment_attempts": 0,
                "elapsed_seconds": 0,
                "usage": {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0},
                "outputs": [],
                "workspace": self.workspace.manifest(),
                "policy": self.policy(),
            }
        )
        self.notes.initialize(config.instruction)
        if state.get("policy") != self.policy():
            raise VividError(
                "policy_mismatch",
                "Task policy changed; resume with the original budget/settings or use a new state directory",
            )
        if (state.get("backend"), state.get("model"), state.get("effort")) != (
            config.backend,
            config.model,
            config.effort,
        ):
            raise VividError(
                "task_mismatch",
                "This memory belongs to another task/model; use a new state directory",
            )
        state.setdefault("segment_attempts", len(state["segments"]))
        startup = state.pop("startup_inflight", None)
        if startup:
            state["elapsed_seconds"] += min(
                max(0, time.time() - startup["started_at"]), startup["remaining_seconds"]
            )
        recovered_completed = self._recover_inflight(state)
        prior_elapsed = state["elapsed_seconds"]
        remaining_seconds = max(0, config.timeout - prior_elapsed - (time.monotonic() - start))
        atomic_json(
            self.workspace.state / "runtime_budget.json",
            {"deadline": time.time() + remaining_seconds},
        )
        # Startup discovery is part of the same cumulative editing budget, including
        # a controller killed while probing or hashing a large source collection.
        state["startup_inflight"] = {
            "started_at": time.time(),
            "remaining_seconds": remaining_seconds,
        }
        atomic_json(run_path, state)
        store = HarnessMemory(self.workspace.state / "lean-memory", self.workspace)
        try:
            with operation_budget(self.workspace.state):
                source_store = MemoryStore(
                    self.workspace.state / "source-index", self.workspace.materials
                )
                catalog = source_store.catalog(refresh=True)
                check_budget()
        except VividError as exc:
            if exc.code != "time_budget":
                raise
            state.pop("startup_inflight", None)
            state.update(
                status="budget_exhausted",
                error="Cumulative task time budget reached during source discovery",
                elapsed_seconds=prior_elapsed + time.monotonic() - start,
            )
            atomic_json(run_path, state)
            self.notes.refresh(store=store)
            return state
        state.pop("startup_inflight", None)
        if not catalog["assets"]:
            raise VividError("empty_materials", "No supplied media files were found")
        sources = sorted(
            (a["asset_id"], a.get("sha256"), a.get("relative_path", a.get("name")))
            for a in catalog["assets"]
        )
        experiment = {
            "instruction": config.instruction,
            "backend": config.backend,
            "model": config.model,
            "effort": config.effort,
            "sources": sources,
            "codex_transport": config.codex_transport,
            "workspace": self.workspace.manifest(),
        }
        identity = hashlib.sha256(json.dumps(experiment, sort_keys=True).encode()).hexdigest()
        if state.get("identity") is not None and state["identity"] != identity:
            raise VividError(
                "task_mismatch",
                "This memory belongs to another task/model; use a new state directory",
            )
        state["identity"] = identity
        if existing:
            if state["status"] == "unverified" and not self.required_outputs():
                state["elapsed_seconds"] += time.monotonic() - start
                atomic_json(run_path, state)
                return state
            if state["status"] == "completed":
                # Idle wall time between invocations is not editing time. Refresh
                # the absolute deadline from the persisted cumulative budget.
                atomic_json(
                    self.workspace.state / "runtime_budget.json",
                    {
                        "deadline": time.time()
                        + max(
                            0,
                            config.timeout - state["elapsed_seconds"] - (time.monotonic() - start),
                        ),
                    },
                )
                complete, _, error = self._complete(state.get("output_hashes"))
                if complete:
                    state["elapsed_seconds"] += time.monotonic() - start
                    atomic_json(run_path, state)
                    return state
                state["elapsed_seconds"] += time.monotonic() - start
                atomic_json(run_path, state)
                raise VividError(
                    "changed_output", "Previously completed output is missing or invalid", error
                )
        if "software" not in state:
            from vivid.diagnostics import software_manifest

            state["software"] = software_manifest()
        prior_elapsed = state["elapsed_seconds"]
        atomic_json(
            self.workspace.state / "runtime_budget.json",
            {
                "deadline": time.time()
                + max(0, config.timeout - prior_elapsed - (time.monotonic() - start)),
                "observation_reserve_seconds": 0,
            },
        )
        state["status"] = "running"
        state.pop("error", None)
        atomic_json(run_path, state)
        atomic_json(self.workspace.state / "tool_budget.json", {"max_calls": config.max_tool_calls})
        session = state.get("session_id")
        if recovered_completed and not self._completion_budget_error(state):
            complete, outputs, _ = self._complete()
            if complete:
                self._accept_outputs(state, outputs)
        while state["status"] == "running" and state["segment_attempts"] < config.max_segments:
            state["elapsed_seconds"] = prior_elapsed + time.monotonic() - start
            budget = self.budget(state)
            unknown_limited_usage = state.get("unmeasured_attempts") and (
                config.max_input_tokens is not None or config.max_output_tokens is not None
            )
            if (
                budget["remaining_seconds"] <= 0
                or unknown_limited_usage
                or any(
                    limit is not None and state["usage"][key] >= limit
                    for key, limit in (
                        ("input_tokens", config.max_input_tokens),
                        ("output_tokens", config.max_output_tokens),
                    )
                )
            ):
                state.update(
                    status="budget_exhausted",
                    error=(
                        "Interrupted native usage is unknown under a finite token budget"
                        if unknown_limited_usage
                        else "Cumulative task budget reached"
                    ),
                )
                break
            prompt = task_prompt(
                config.instruction, self.workspace, budget, resume=bool(state["segment_attempts"])
            )
            if state.get("continuation"):
                prompt += "\n\nContinue the same task. Last output check:\n" + state["continuation"]
            mcp_command = [
                sys.executable,
                "-m",
                "vivid",
                "serve",
                "--workspace",
                str(self.workspace.root),
                "--materials",
                str(self.workspace.materials),
                "--state",
                str(self.workspace.state),
                "--output",
                str(self.workspace.output),
            ]
            native = self.backend_factory(
                BackendConfig(
                    backend=config.backend,
                    model=config.model,
                    effort=config.effort,
                    executable=config.executable,
                    workspace=self.workspace.root,
                    mcp_command=mcp_command,
                    log_dir=self.workspace.state / "native_logs",
                    # Allow the controller to record/validate the final files inside
                    # the same original budget, instead of starting another conversation.
                    timeout_seconds=max(
                        0.001,
                        budget["remaining_seconds"] - min(15, budget["remaining_seconds"] * 0.05),
                    ),
                    context_token_limit=config.context_token_limit,
                    codex_sandbox=config.codex_sandbox,
                    codex_transport=config.codex_transport,
                    claude_permission_mode=config.claude_permission_mode,
                )
            )
            # Reserve the attempt and elapsed time before control enters the native CLI.
            state["segment_attempts"] += 1
            state["elapsed_seconds"] = prior_elapsed + time.monotonic() - start
            state["inflight"] = {
                "started_at": time.time(),
                "remaining_seconds": max(0, config.timeout - state["elapsed_seconds"]),
                "existing_native_directories": [
                    str(path)
                    for path in (self.workspace.state / "native_logs").glob("*")
                    if path.is_dir()
                ],
                "existing_results": [
                    str(path)
                    for path in (self.workspace.state / "native_logs").glob("*/result.json")
                ],
            }
            atomic_json(run_path, state)
            try:
                result = native.run(prompt, session_id=session)
            except (VividError, OSError, RuntimeError) as exc:
                state.pop("inflight", None)
                state["unmeasured_attempts"] = state.get("unmeasured_attempts", 0) + 1
                state.update(status="failed", error=str(exc), usage_complete=False)
                break
            state.pop("inflight", None)
            self._add_result(state, result.as_dict())
            state["elapsed_seconds"] = prior_elapsed + time.monotonic() - start
            store.record_event("backend_segment", result.as_dict())
            atomic_json(run_path, state)
            self.notes.refresh(store=store)
            if result.status in {"failed", "timeout"}:
                state.update(
                    status=result.status, error=result.error or "Native backend did not finish"
                )
                state["outputs"] = [
                    str(self.workspace.output / name)
                    for name in self.required_outputs()
                    if (self.workspace.output / name).is_file()
                ]
                break
            budget_error = self._completion_budget_error(state)
            if budget_error:
                state.update(status="budget_exhausted", error=budget_error)
                break
            # Native signals and an explicit marker both roll context without a special tool.
            handoff = (
                result.status == "checkpoint"
                or "VIVID_CHECKPOINT" in result.final_text
                or (
                    config.context_token_limit is not None
                    and result.peak_context_tokens >= config.context_token_limit
                )
            )
            if handoff:
                session = None
                state["session_id"] = None
                state["continuation"] = (
                    "Continue in a fresh context using the actual files, GUIDE.md, WORKING.md and archived evidence IDs. Check pending native jobs before launching work."
                )
                atomic_json(run_path, state)
                continue
            complete, outputs, error = self._complete()
            if complete:
                self._accept_outputs(state, outputs)
                break
            if not self.required_outputs():
                state.update(
                    status="unverified",
                    outputs=[],
                    completion={
                        "scope": "native_completion_only",
                        "semantic_success": False,
                        "reason": error,
                    },
                )
                break
            state["continuation"] = error or "Required output is not ready; continue editing."
            session = result.session_id
        if state["status"] == "running":
            state.update(
                status="budget_exhausted", error="Maximum native conversation segments reached"
            )
        state["elapsed_seconds"] = prior_elapsed + time.monotonic() - start
        atomic_json(run_path, state)
        self.notes.refresh(store=store)
        store.record_event(
            "run_finished",
            {
                "status": state["status"],
                "usage": state["usage"],
                "elapsed_seconds": state["elapsed_seconds"],
            },
        )
        return state
