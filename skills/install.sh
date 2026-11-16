#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime="$HOME/.local/share/vivid/venv"
case "${1:-}" in
  codex)
    skill_dir="${CODEX_HOME:-$HOME/.codex}/skills/vivid"
    invocation='$vivid'
    ;;
  claude)
    skill_dir="$HOME/.claude/skills/vivid"
    invocation='/vivid'
    ;;
  *)
    printf 'Usage: bash skills/install.sh {codex|claude}\n' >&2
    exit 1
    ;;
esac

python=""
for candidate in python3 python3.13 python3.12 python3.11; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    python="$candidate"
    break
  fi
done
if [ -z "$python" ]; then
  printf 'Python 3.11 or newer is required.\n' >&2
  exit 1
fi

"$python" -m venv "$runtime"
"$runtime/bin/python" -m pip install "$repository"
mkdir -p "$skill_dir"
cp "$repository/skills/vivid/SKILL.md" "$skill_dir/SKILL.md"

printf '\nVIVID skill installed for %s.\n' "$1"
printf 'Restart your agent session, then use %s.\n' "$invocation"
printf 'FFmpeg and ffprobe must be available on PATH.\n'
