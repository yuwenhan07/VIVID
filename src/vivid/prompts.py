from __future__ import annotations

import json

from vivid.notes import RecoveryNotes
from vivid.workspace import Workspace

METHOD = """Complete the original task using your native tools and any useful Vivid media tools.
The original task brief governs source permissions, constraints and deliverables.
Keep supplied sources unchanged.

Vivid's optional media helpers are described by the tool schemas. Use native tools
and choose your own editing methods; no plan registration or Vivid execution is required.
Sampled frames are not continuous coverage. A transcript or spectrogram is not
proof of listening. Technical checks do not establish content quality.

TASK.md preserves the brief. GUIDE.md/WORKING.md are optional native-file notes;
MEMORY.md indexes archived evidence. When a fresh session is necessary, update notes
and end with VIVID_CHECKPOINT. On continuation inspect actual files and pending jobs.
Report delivered files and any unverified requirements.
"""


def task_prompt(instruction: str, workspace: Workspace, budget: dict, resume: bool = False) -> str:
    notes = RecoveryNotes(workspace)
    notes.initialize(instruction)
    notes.refresh()
    state = {
        "workspace": workspace.manifest(),
        "budget": budget,
        "continuation": resume,
        "recovery": notes.snapshot(),
    }
    return (
        METHOD
        + "\n\nOriginal task brief (preserved in full):\n"
        + instruction
        + "\n\nCurrent durable state:\n"
        + json.dumps(state, ensure_ascii=False)
    )
