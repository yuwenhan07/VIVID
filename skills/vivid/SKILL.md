---
name: vivid
description: Edit or review video using VIVID's audiovisual inspection, evidence memory, and output verification tools. Use when a task involves selecting, assembling, repairing, or repurposing video and VIVID is installed.
---

# VIVID video editing

Use the current agent to plan and perform the edit, and VIVID to inspect media, preserve evidence, and review the result. Do not start `vivid worker`: it launches another agent. `vivid tool` does not require an OpenRouter key.

Choose a workspace that contains the input and output files. Paths passed to VIVID must be inside it. If `vivid` is not on PATH but `~/.local/share/vivid/venv/bin/vivid` exists, prepend `~/.local/share/vivid/venv/bin` to PATH in each shell session that runs VIVID. Check `vivid doctor`; VIVID, Python 3.11+, FFmpeg and ffprobe must be installed. Full managed rendering requires Linux. If VIVID is unavailable, ask the user to install it from the VIVID repository.

1. Inspect the source with `vivid tool --workspace WORKSPACE inspect --arguments '{"path":"materials/source.mp4","mode":"overview","max_frames":24}'`, substituting the actual workspace and source path. Use local windows or audio inspection where the task needs more detail. Open the image paths returned in the JSON with an available image-viewing tool; sampling alone cannot establish everything in the video.
2. Use `transcribe` when speech matters and local ASR is available. Retrieve saved observations by evidence ID or search indexed text with `retrieve`. Keep source files unchanged while working so evidence remains traceable.
3. Edit with the agent's shell and programming tools or VIVID's `edit` tool. Follow the user's requested output path and specifications.
4. Inspect the rendered output, especially cuts, content, framing and audiovisual alignment. Use `verify` for technical checks; successful decoding does not prove the edit meets the brief. Revise when needed and report the output path and any unverified requirements.

Run `vivid tool --workspace WORKSPACE` for the current tool schemas and arguments. For example, verify a result with `vivid tool --workspace WORKSPACE verify --arguments '{"path":"output/final.mp4"}'`. Transcription requires a separately installed ASR backend and model; see the repository README for setup.
