"""Version/configuration-bound interval transcripts, shared across native/MCP calls."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path

from vivid.errors import VividError
from vivid.runtime import bounded_lock, check_budget, stage

_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_MODEL_HASHES = {}


def _assemble(pieces, start, end):
    """Keep boundary-crossing segments and remove only exact overlap text.

    Segment timestamps remain the original ASR span after text is deduplicated;
    they do not become word alignment. Uncertain joins are explicitly marked.
    """
    segments, origins, seen = [], [], set()
    for piece_index, (lo, hi, result) in enumerate(pieces):
        for segment in result.get("segments", []):
            a, b = segment["source_start"], segment["source_end"]
            if not (a < hi and b > lo):
                continue
            item = copy.deepcopy(segment)
            words = item.get("words")
            if words:
                words = [w for w in words if lo <= (w["source_start"] + w["source_end"]) / 2 < hi]
                if not words:
                    continue
                item.update(
                    words=words,
                    text="".join(w["word"] for w in words),
                    source_start=words[0]["source_start"],
                    source_end=words[-1]["source_end"],
                )
                a, b = item["source_start"], item["source_end"]
            else:
                # Segments can straddle the cache ownership boundary; dropping a
                # whole segment by its midpoint silently drops words on one side.
                if a < lo or b > hi:
                    item["crosses_cache_boundary"] = True
            signature = (round(a, 6), round(b, 6), item["text"])
            if signature in seen:
                continue
            seen.add(signature)
            overlapping = [
                previous
                for previous, origin in zip(segments, origins)
                if origin != piece_index
                and previous["source_start"] < b
                and previous["source_end"] > a
            ]
            if overlapping:
                earlier = [
                    m.group().casefold()
                    for m in _WORD.finditer(" ".join(row["text"] for row in overlapping))
                ]
                current = list(_WORD.finditer(item["text"]))
                later = [m.group().casefold() for m in current]
                common = next(
                    (
                        n
                        for n in range(min(len(earlier), len(later), 40), 1, -1)
                        if earlier[-n:] == later[:n]
                    ),
                    0,
                )
                if common and words:
                    token_count = 0
                    split_at = None
                    for index, word in enumerate(words):
                        token_count += len(_WORD.findall(word["word"]))
                        if token_count == common:
                            split_at = index + 1
                            break
                        if token_count > common:
                            break
                    if split_at is not None:
                        words = words[split_at:]
                        if not words:
                            continue
                        item.update(
                            words=words,
                            text="".join(word["word"] for word in words),
                            source_start=words[0]["source_start"],
                            source_end=words[-1]["source_end"],
                            deduplicated_overlap_words=common,
                        )
                        a, b = item["source_start"], item["source_end"]
                    else:
                        item["overlap_review_required"] = True
                elif common:
                    item["text"] = re.sub(
                        r"^[\s,.;:!?…]+", "", item["text"][current[common - 1].end() :]
                    )
                    item["deduplicated_overlap_words"] = common
                    item["timing_warning"] = (
                        "Segment timestamps include removed overlap; they are not word alignment."
                    )
                    if not item["text"]:
                        continue
                else:
                    item["overlap_review_required"] = True
            item.update(start=a - start, end=b - start)
            for word in item.get("words", []):
                word.update(start=word["source_start"] - start, end=word["source_end"] - start)
            if a < start or b > end:
                item["context_outside_request"] = True
            segments.append(item)
            origins.append(piece_index)
    return segments


_FASTER_ALIASES = {
    **{
        name: f"Systran/faster-whisper-{name}"
        for name in (
            "tiny.en",
            "tiny",
            "base.en",
            "base",
            "small.en",
            "small",
            "medium.en",
            "medium",
            "large-v1",
            "large-v2",
            "large-v3",
        )
    },
    "large": "Systran/faster-whisper-large-v3",
    **{
        name: f"Systran/faster-{name.replace('distil-', 'distil-whisper-')}"
        for name in (
            "distil-large-v2",
            "distil-medium.en",
            "distil-small.en",
            "distil-large-v3",
        )
    },
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}


def _file_identity(path):
    try:
        stat = path.stat()
        if not path.is_file():
            return None
        identity = [
            str(path.resolve()),
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        ]
        key = tuple(identity)
        # Overlay filesystems can coalesce timestamps for rapid same-size writes.
        # Recently changed files are rehashed rather than trusting their metadata.
        stable = max(stat.st_mtime_ns, stat.st_ctime_ns) < (time.time() - 1) * 1e9
        digest = _MODEL_HASHES.get(key) if stable else None
        if digest is None:
            from vivid.memory import sha256_file

            digest = sha256_file(path)
            after = path.stat()
            if (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) != (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns):
                return None
            if stable:
                if len(_MODEL_HASHES) >= 128:
                    _MODEL_HASHES.clear()
                _MODEL_HASHES[key] = digest
        return [*identity, digest]
    except OSError:
        return None


def _model_identity(backend, model):
    path = Path(model).expanduser()
    if backend == "whisper":
        if not path.is_file():
            cache = Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache")))
            name = {"large": "large-v3", "turbo": "large-v3-turbo"}.get(model, model)
            path = cache / "whisper" / (name + ".pt")
        return _file_identity(path)
    if backend != "faster-whisper":
        return None
    if not path.is_dir():
        repo = _FASTER_ALIASES.get(model, model if "/" in model else None)
        if not repo:
            return None
        default_home = (
            Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface"
        )
        hub_home = Path(os.getenv("HF_HOME", str(default_home))).expanduser() / "hub"
        hub = Path(
            os.getenv("HF_HUB_CACHE", os.getenv("HUGGINGFACE_HUB_CACHE", str(hub_home)))
        ).expanduser()
        repo_path = hub / ("models--" + repo.replace("/", "--"))
        try:
            revision = (repo_path / "refs" / "main").read_text().strip()
        except OSError:
            return None
        if not re.fullmatch(r"[a-fA-F0-9]{40,64}", revision):
            return None
        path = repo_path / "snapshots" / revision
    weights = _file_identity(path / "model.bin")
    tokenizer = _file_identity(path / "tokenizer.json")
    # faster-whisper falls back to a separate remotely resolved tokenizer when
    # this file is absent; its identity is not established here.
    if weights is None or tokenizer is None:
        return None
    return {
        "model_bin": weights,
        "config": _file_identity(path / "config.json"),
        "tokenizer": tokenizer,
        "vocabulary_json": _file_identity(path / "vocabulary.json"),
        "vocabulary_txt": _file_identity(path / "vocabulary.txt"),
        "preprocessor": _file_identity(path / "preprocessor_config.json"),
    }


def profile(word_timestamps=False):
    model = os.getenv("VIVID_WHISPER_MODEL", "base")
    backend = os.getenv("VIVID_ASR_BACKEND", "faster-whisper")
    return {
        "version": 3,
        "backend": backend,
        "model": model,
        "weights": _model_identity(backend, model),
        "python": os.getenv("VIVID_MEDIA_PYTHON", "default"),
        "pythonpath": os.getenv("PYTHONPATH", ""),
        "word_timestamps": word_timestamps,
    }


class TranscriptCache:
    def __init__(self, store, asset, language, word_timestamps=False):
        self.store = store
        self.identity = {
            "source_sha256": asset["sha256"],
            "language": language,
            **profile(word_timestamps),
        }
        if self.identity["weights"] is None:
            # Never reuse persistent transcripts when the loaded model cannot be identified.
            self.identity["unresolved_session"] = uuid.uuid4().hex
        self.key = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        with store._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS transcript_chunks "
                "(profile TEXT, start REAL, end REAL, payload TEXT)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS transcript_profile "
                "ON transcript_chunks(profile,start,end)"
            )

    def _check_model(self):
        if self.identity["weights"] is not None:
            current = profile(self.identity["word_timestamps"])
            if current != {k: self.identity[k] for k in current}:
                raise VividError(
                    "asr_model_changed", "ASR model changed during transcription; retry."
                )

    def resolve(self, start, end, load):
        """Transcribe uncovered intervals only, retaining context at cache joins.

        Text spanning a boundary is never silently declared word-accurate. Source
        timestamps remain authoritative; local start/end are rebuilt for this request.
        """
        with bounded_lock(self.store.root / "transcripts" / (self.key + ".lock")):
            self._check_model()
            profiles = [self.key]
            if not self.identity["word_timestamps"]:
                word_profile = {**self.identity, "word_timestamps": True}
                profiles.append(
                    hashlib.sha256(json.dumps(word_profile, sort_keys=True).encode()).hexdigest()
                )
            with self.store._connect() as connection:
                rows = connection.execute(
                    "SELECT start,end,payload FROM transcript_chunks WHERE profile IN ("
                    + ",".join("?" for _ in profiles)
                    + ") "
                    "AND end>? AND start<? ORDER BY start,end DESC",
                    (*profiles, start, end),
                ).fetchall()
            chunks = [(r[0], r[1], json.loads(r[2])) for r in rows]
            cursor, pieces, missing = start, [], []
            while cursor < end - 1e-6:
                candidates = [c for c in chunks if c[0] <= cursor + 1e-6 and c[1] > cursor + 1e-6]
                if candidates:
                    chosen = max(candidates, key=lambda c: c[1])
                    stop = min(end, chosen[1])
                    pieces.append((cursor, stop, chosen[2]))
                    cursor = stop
                    continue
                stop = min([c[0] for c in chunks if c[0] > cursor] + [end])
                # Keep requests short even when called from a native script.
                stop = min(stop, cursor + (10 if self.identity["word_timestamps"] else 25))
                lo, hi = max(start, cursor - 2), min(end, stop + 2)
                check_budget()
                with stage("asr", requested_seconds=hi - lo):
                    result = load(lo, hi)
                    self._check_model()
                    if not result.get("available"):
                        from vivid.runtime import mark_stage_failure

                        mark_stage_failure(result.get("error_kind", "asr_unavailable"))
                if not result.get("available"):
                    partial = _assemble(pieces, start, end)
                    return {
                        **result,
                        "partial": bool(pieces),
                        "segments": partial,
                        "text": " ".join(s["text"].strip() for s in partial),
                        "covered_ranges": [[a, b] for a, b, _ in pieces],
                        "pending_ranges": [[cursor, end]],
                        "cache_hit": False,
                    }
                with self.store._connect() as connection:
                    connection.execute(
                        "INSERT INTO transcript_chunks VALUES(?,?,?,?)",
                        (self.key, lo, hi, json.dumps(result, allow_nan=False)),
                    )
                chunks.append((lo, hi, result))
                missing.append([lo, hi])
            segments = _assemble(pieces, start, end)
            base = pieces[0][2] if pieces else {}
            text = " ".join(s["text"].strip() for s in segments)
            if not segments and len(pieces) == 1:
                text = base.get("text", "")
            self._check_model()
            return {
                **base,
                "segments": segments,
                "text": text,
                "word_timestamps_available": bool(
                    self.identity["word_timestamps"] and any(s.get("words") for s in segments)
                ),
                "cache_hit": not missing,
                "word_timestamps_requested": self.identity["word_timestamps"],
                "transcribed_ranges": missing,
                "covered_ranges": [[start, end]],
                "pending_ranges": [],
                "time_range": [start, end],
                "profile": self.identity,
                "boundary_warning": "ASR estimates are not forced alignment. Check joins locally; segment-only text can span request boundaries.",
            }
