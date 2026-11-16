from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from vivid import DEFAULT_TIMEOUT_SECONDS, __version__
from vivid.errors import VividError
from vivid.workspace import Workspace


def print_json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def workspace_options(parser):
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--materials", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--output", type=Path)


def workspace(args):
    return Workspace.open(args.workspace, args.materials, args.state, args.output)


def runtime_options(parser):
    parser.add_argument("--backend", choices=("codex", "claude", "opencode"), default="codex")
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort")
    parser.add_argument("--executable")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Cumulative runtime budget in seconds (default: 1800)",
    )
    parser.add_argument("--max-segments", type=int, default=16)
    parser.add_argument("--context-token-limit", type=int)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--max-tool-calls", type=int, default=10000)
    parser.add_argument(
        "--codex-sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default="workspace-write",
        help="Native Codex sandbox (default: workspace-write)",
    )
    parser.add_argument("--codex-transport", choices=("exec", "app-server"), default="exec")
    parser.add_argument("--claude-permission-mode", default="acceptEdits")


def parser():
    root = argparse.ArgumentParser(
        description="VIVID audiovisual memory harness for native editing agents"
    )
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser(
        "doctor", help="Check local media tools, Python and optional agent CLIs"
    )
    doctor.add_argument("--backend", choices=("codex", "claude", "opencode"))
    worker = commands.add_parser(
        "worker", help="Run the complete controller inside an existing workspace"
    )
    workspace_options(worker)
    runtime_options(worker)
    worker.add_argument("--instruction-file", type=Path, required=True)
    worker.add_argument("--required-output", action="append", default=[])
    serve = commands.add_parser("serve", help="Expose VIVID tools through an MCP server")
    workspace_options(serve)
    serve.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    tool = commands.add_parser("tool", help="Invoke a tool locally, with JSON arguments")
    workspace_options(tool)
    tool.add_argument("name", nargs="?")
    tool.add_argument("--arguments", default="{}")
    return root


def doctor(args):
    checks = {}
    for name in ("ffmpeg", "ffprobe", "codex", "claude", "opencode"):
        path = shutil.which(name)
        checks[name] = {"available": bool(path), "path": path}
        if path and name in {"ffmpeg", "ffprobe", args.backend}:
            try:
                flag = "-version" if name in {"ffmpeg", "ffprobe"} else "--version"
                result = subprocess.run([path, flag], capture_output=True, text=True, timeout=10)
                checks[name]["version"] = (
                    result.stdout.splitlines()[0] if result.stdout else "unknown"
                )
            except (OSError, subprocess.SubprocessError):
                pass
    checks["python"] = sys.version.split()[0]
    checks["vivid"] = __version__
    checks["audio"] = (
        "Analysis/transcription available through tools; native planner audio input is not assumed"
    )
    required = ["ffmpeg", "ffprobe"] + ([args.backend] if args.backend else [])
    checks["ok"] = all(checks[name]["available"] for name in required)
    return checks


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(arguments)
    try:
        if args.command == "doctor":
            result = doctor(args)
            print_json(result)
            return 0 if result["ok"] else 1
        if args.command == "worker":
            import signal

            # Unwind native backend finally blocks on termination, reaping CLI/MCP
            # process groups instead of letting edits continue during verification.
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
            from vivid.controller import Controller, RunConfig

            config = RunConfig(
                workspace(args),
                args.instruction_file.read_text(),
                args.backend,
                args.model,
                effort=args.effort,
                executable=args.executable,
                timeout=args.timeout,
                max_segments=args.max_segments,
                context_token_limit=args.context_token_limit,
                max_input_tokens=args.max_input_tokens,
                max_output_tokens=args.max_output_tokens,
                max_tool_calls=args.max_tool_calls,
                required_outputs=tuple(args.required_output),
                codex_sandbox=args.codex_sandbox,
                codex_transport=args.codex_transport,
                claude_permission_mode=args.claude_permission_mode,
            )
            result = Controller(config).run()
            print_json(result)
            return 0 if result["status"] == "completed" else 1
        if args.command == "serve":
            import signal

            signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
            from vivid.server import serve

            serve(workspace(args), args.transport, args.host, args.port)
            return 0
        if args.command == "tool":
            from vivid.harness import HarnessService

            service = HarnessService(workspace(args))
            result = (
                service.dispatch(args.name, json.loads(args.arguments))
                if args.name
                else service.definitions()
            )
            print_json(result)
            return int(isinstance(result, dict) and result.get("ok") is False)
    except (VividError, ValueError, TypeError, KeyError, OSError) as exc:
        print_json(
            exc.as_dict()
            if isinstance(exc, VividError)
            else VividError("command_failed", str(exc)).as_dict()
        )
        return 1
    return 0
