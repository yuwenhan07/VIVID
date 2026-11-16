from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
from pathlib import Path

from vivid.harness import HarnessService
from vivid.workspace import Workspace

MAX_IMAGE_BYTES = 256_000
MAX_IMAGE_COUNT = 12
# Codex may wrap or truncate MCP image blocks when the encoded response exceeds
# its transport limit. Leave room for base64 expansion and the text metadata.
MAX_IMAGE_TOTAL = 660_000
MAX_TEXT_BYTES = 48_000


def _compact(value):
    if isinstance(value, dict):
        return {
            k: _compact(v)
            for k, v in value.items()
            if k != "files" and not (k == "payload" and "evidence_id" in value)
        }
    if isinstance(value, list):
        return [_compact(v) for v in value]
    return value


def _preview(data, max_bytes=MAX_IMAGE_BYTES):
    """Keep small lossless evidence exact; bound larger transport derivatives."""
    from PIL import Image

    if len(data) <= max_bytes:
        return data, None
    with Image.open(io.BytesIO(data)) as original:
        preview = original.convert("RGB")
        preview.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        for _ in range(12):
            for quality in (85, 70, 55):
                stream = io.BytesIO()
                preview.save(stream, format="JPEG", quality=quality, optimize=True)
                encoded = stream.getvalue()
                if len(encoded) <= max_bytes:
                    return encoded, {
                        "encoding": "jpeg",
                        "size": list(preview.size),
                        "quality": quality,
                        "original_bytes": len(data),
                    }
            preview.thumbnail((max(1, int(preview.width * 0.8)), max(1, int(preview.height * 0.8))))
    raise ValueError("Cannot produce a bounded image preview")


def make_server(workspace: Workspace, *, service: HarnessService | None = None):
    from mcp.server import Server
    from mcp.types import Tool

    server = Server("vivid")
    service = service or HarnessService(workspace)
    server.harness_service = service

    @server.list_tools()
    async def list_tools():
        return [Tool(**spec) for spec in service.definitions()]

    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict):
        from vivid.runtime import operation_budget

        cancelled = threading.Event()

        def dispatch():
            from vivid.errors import VividError

            try:
                # The service enforces the deadline inside its logged dispatch.
                # This outer context only supplies cooperative RPC cancellation.
                with operation_budget(workspace.state, cancelled.is_set, enforce_deadline=False):
                    return service.dispatch(name, arguments)
            except VividError as exc:
                return exc.as_dict()

        try:
            result = await asyncio.to_thread(dispatch)
        finally:
            cancelled.set()
        return encode_result(workspace, result)

    return server


def encode_result(workspace: Workspace, result: dict):
    """Report delivery separately from retrieval so missing images are never silent."""
    from mcp.types import CallToolResult, ImageContent, TextContent
    from PIL import Image

    result = _compact(result)
    images = result.pop("images", [])
    offset = result.pop("image_offset", 0)
    visual_content = []
    skipped = []
    delivered_indices = []
    total_bytes = 0
    end = min(len(images), offset + MAX_IMAGE_COUNT)
    per_image_budget = min(MAX_IMAGE_BYTES, MAX_IMAGE_TOTAL // max(1, end - offset))
    for index in range(offset, end):
        item = images[index]
        reason = None
        try:
            path = Path(item["path"]).resolve()
            mime = item.get("mime_type", "image/png")
            if not (path.is_relative_to(workspace.state) or path.is_relative_to(workspace.output)):
                reason = "outside_workspace_artifacts"
            elif not path.is_file():
                reason = "missing_image"
            elif mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
                reason = "unsupported_image_type"
            elif path.stat().st_size > 16 * 1024 * 1024:
                reason = "image_size_limit"
            if reason is None:
                with path.open("rb") as stream:
                    data = stream.read(16 * 1024 * 1024 + 1)
                if len(data) > 16 * 1024 * 1024:
                    raise ValueError("image exceeds the encoded byte limit")
                with Image.open(io.BytesIO(data)) as image:
                    if Image.MIME.get(image.format) != mime:
                        raise ValueError("image MIME type does not match file contents")
                    if image.width * image.height > 40_000_000:
                        raise ValueError("image exceeds the decoded pixel limit")
                    image.verify()
                data, transform = _preview(data, per_image_budget)
                if transform:
                    mime = "image/jpeg"
                    result.setdefault("transport_previews", []).append(
                        {"image_index": index, **transform}
                    )
                if total_bytes + len(data) > MAX_IMAGE_TOTAL:
                    skipped.append(
                        {"index": index, "reason": "aggregate_preview_limit", "path": str(path)}
                    )
                    continue
                total_bytes += len(data)
                visual_content.append(
                    ImageContent(type="image", mimeType=mime, data=base64.b64encode(data).decode())
                )
                delivered_indices.append(index)
        except (KeyError, TypeError, OSError, ValueError, Image.DecompressionBombError) as exc:
            reason = "image_read_failed: " + str(exc)
        if reason:
            skipped.append({"index": index, "reason": reason})
    if images:
        result["image_delivery"] = {
            "requested": len(images),
            "delivered": len(visual_content),
            "complete": offset == 0 and len(visual_content) == len(images),
            "page_complete": not skipped,
            "offset": offset,
            "next_offset": end if end < len(images) else None,
            "remaining": len(images) - end,
            "delivered_indices": delivered_indices,
            "skipped": skipped,
            "delivered_bytes": total_bytes,
            "scope": "server_encoded_only; complete refers to the whole evidence in this response; page_complete refers to the current page",
        }
        if end < len(images) and result.get("evidence_id"):
            result["next_image_page"] = {
                "tool": "retrieve",
                "arguments": {"evidence_id": result["evidence_id"], "image_offset": end},
            }
        for name in ("image_groups", "groups"):
            if isinstance(result.get(name), list):
                result[name] = [
                    group for group in result[name] if group.get("image_index") in delivered_indices
                ]
        if not visual_content:
            result["ok"] = False
            result["delivery_error"] = (
                "No requested images reached the planner; inspect the delivery report and retry."
            )
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode()) > MAX_TEXT_BYTES:
        import uuid

        from vivid.workspace import atomic_json

        detail = workspace.state / "responses" / (uuid.uuid4().hex + ".json")
        atomic_json(detail, result)
        if isinstance(result.get("image_groups"), list):
            compact_groups = []
            for group in result["image_groups"]:
                sources = {}
                for sample in group.get("samples", []):
                    sources.setdefault(sample["path"], []).append(sample["timestamp"])
                compact_groups.append(
                    {
                        "image_index": group["image_index"],
                        "sources": [{"path": p, "timestamps": t} for p, t in sources.items()],
                    }
                )
            result["image_groups"] = compact_groups
        result = {
            k: v
            for k, v in result.items()
            if k
            not in {
                "observation_summary",
                "frames",
                "items",
                "matches",
                "segments",
                "text",
                "windows",
                "assets",
                "transcription",
            }
        }
        result.update(
            detail_path=str(detail),
            response_truncated=True,
            guidance="Full metadata is available in detail_path; image blocks remain separate.",
        )
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode()) > MAX_TEXT_BYTES:
            result = {
                "ok": result.get("ok", True),
                "detail_path": str(detail),
                "response_truncated": True,
                "image_delivery": result.get("image_delivery"),
                "next_image_page": result.get("next_image_page"),
                "evidence_id": result.get("evidence_id"),
                "error": result.get("error"),
            }
            encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
    content = [TextContent(type="text", text=encoded)]
    content.extend(visual_content)
    return CallToolResult(
        content=content, isError=not result.get("ok", True), structuredContent=result
    )


def serve(
    workspace: Workspace, transport: str = "stdio", host: str = "127.0.0.1", port: int = 8765
):
    from mcp.server.stdio import stdio_server

    server = make_server(workspace)
    if transport == "stdio":

        async def run():
            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())

        asyncio.run(run())
        return
    import contextlib

    import uvicorn
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.routing import Mount

    manager = StreamableHTTPSessionManager(app=server, json_response=True)

    async def handle(scope, receive, send):
        await manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with manager.run():
            yield

    app = Starlette(routes=[Mount("/mcp", app=handle)], lifespan=lifespan)
    uvicorn.run(app, host=host, port=port, log_level="warning")
