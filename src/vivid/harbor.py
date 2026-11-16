"""Harbor adapter for the AgenticVBench Greta repurposing task."""

from __future__ import annotations

import json
import shlex
import shutil
import tempfile
from importlib import metadata
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent, NonZeroAgentExitCodeError
from harbor.agents.installed.codex import Codex

from vivid import __version__
from vivid.backends.base import Redactor


class GretaAgent(BaseInstalledAgent):
    def __init__(self, logs_dir: Path, model_name: str | None = None, *, cli_version=None, **kwargs):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.cli_version = cli_version
        self.codex_executable = None
        self.result = None

    @staticmethod
    def name() -> str:
        return "vivid"

    def version(self) -> str:
        return __version__

    async def install(self, environment):
        native = Codex(logs_dir=self.logs_dir, model_name=self.model_name, version=self.cli_version)
        await native.install(environment)
        resolved = await self._execute(
            environment,
            'if [ -s "$HOME/.nvm/nvm.sh" ]; then\n'
            '  . "$HOME/.nvm/nvm.sh"\n'
            'fi\ncommand -v codex',
        )
        self.codex_executable = resolved.stdout.strip().splitlines()[-1]

        # Install VIVID separately from the task's verifier dependencies.
        requirements = [
            requirement
            for requirement in metadata.requires("vivid-harness") or []
            if "extra ==" not in requirement
        ]
        with tempfile.TemporaryDirectory(prefix="vivid-harbor-") as directory:
            package = Path(directory)
            shutil.copytree(
                Path(__file__).parent,
                package / "src" / "vivid",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            (package / "pyproject.toml").write_text(
                '[build-system]\nrequires = ["setuptools>=77"]\n'
                'build-backend = "setuptools.build_meta"\n'
                '[project]\nname = "vivid-harness"\n'
                f'version = {json.dumps(__version__)}\nrequires-python = ">=3.11"\n'
                f"dependencies = {json.dumps(requirements)}\n"
                '[tool.setuptools.packages.find]\nwhere = ["src"]\n',
                encoding="utf-8",
            )
            await self._execute(environment, "mkdir -p /installed-agent/vivid/package", user="root")
            await environment.upload_dir(package, "/installed-agent/vivid/package")

        await self._execute(
            environment,
            "python3 -m venv /installed-agent/vivid/venv",
            user="root",
        )
        await self._execute(
            environment,
            "/installed-agent/vivid/venv/bin/python -m pip install "
            "--disable-pip-version-check /installed-agent/vivid/package",
            user="root",
            timeout_sec=600,
        )
        await self._execute(
            environment, "mkdir -p /workspace/materials /workspace/output /workspace/work/vivid"
        )
        await self._execute(
            environment, "cp -n /baked/source.mp4 /workspace/materials/source.mp4"
        )

    async def _execute(self, environment, command, *, user=None, timeout_sec=None):
        result = await environment.exec(command=command, user=user, timeout_sec=timeout_sec)
        if result.return_code:
            detail = Redactor(self._extra_env).text(result.stderr or result.stdout or "")
            raise RuntimeError(f"VIVID setup failed: {detail[-1000:]}")
        return result

    async def run(self, instruction, environment, context):
        if not self.model_name:
            raise ValueError("Provide an OpenRouter model with --model")

        with tempfile.TemporaryDirectory(prefix="vivid-instruction-") as directory:
            path = Path(directory) / "instruction.md"
            path.write_text(self.render_instruction(instruction), encoding="utf-8")
            await environment.upload_file(path, "/workspace/work/vivid/instruction.md")

        command = [
            "/installed-agent/vivid/venv/bin/python",
            "-m", "vivid", "worker",
            "--workspace", "/workspace",
            "--instruction-file", "/workspace/work/vivid/instruction.md",
            "--backend", "codex",
            "--model", self.model_name,
            "--executable", self.codex_executable,
            "--timeout", "1740",
            "--codex-sandbox", "danger-full-access",
            "--required-output", "repurpose.mp4",
        ]
        env = {
            **self._extra_env,
            "VIVID_MEDIA_PYTHON": "/usr/local/bin/python3",
            "VIVID_ASR_BACKEND": "whisper",
        }
        result = await environment.exec(command=shlex.join(command), cwd="/workspace", env=env)
        redactor = Redactor(env)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "vivid-worker.log").write_text(
            redactor.text((result.stdout or "") + "\n" + (result.stderr or "")), encoding="utf-8"
        )
        if result.stdout:
            self.result = json.loads(result.stdout)
            self.populate_context_post_run(context)
        if result.return_code:
            raise NonZeroAgentExitCodeError(
                f"VIVID worker exited {result.return_code}; see vivid-worker.log"
            )

    def populate_context_post_run(self, context):
        if not self.result:
            return
        usage = self.result.get("usage", {})
        context.n_input_tokens = usage.get("input_tokens")
        context.n_output_tokens = usage.get("output_tokens")
        context.n_cache_tokens = usage.get("cached_input_tokens")
        context.metadata = {"vivid_status": self.result.get("status")}
