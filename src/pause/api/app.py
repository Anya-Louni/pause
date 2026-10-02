# Adapted from claimcheck (github.com/Anya-Louni/claimcheck) at commit 4dca7b7.
# Kept as a copy so this repo runs on its own.

"""FastAPI app: streams agent runs as Server-Sent Events.

Each thread gets its own simulated workspace, so visitors never see each
other's calendar. Keys stay on the server.
"""

import hmac
import json
import os
import sqlite3
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel

from pause.agent import build_agent, daily_cap
from pause.config import settings
from pause.media import MediaError, audio_block, check_audio, clean_image, image_block
from pause.streaming import STREAM_MODES, to_events
from pause.tools import PauseContext, workspace_of

RUNS_PER_HOUR = 20
MAX_IMAGES = 2

app = FastAPI(title="Pause")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_origin],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_checkpoints = SqliteSaver(
    sqlite3.connect(settings.data_dir / "checkpoints.sqlite", check_same_thread=False)
)
_agent = build_agent(checkpointer=_checkpoints, setup="gate")
_hits: dict[str, deque[float]] = defaultdict(deque)


def client_ip(request: Request) -> str:
    """The caller's address, from the web proxy's signed header when present."""
    secret = os.getenv("PROXY_SECRET")
    if secret and hmac.compare_digest(request.headers.get("x-proxy-secret", ""), secret):
        visitor = request.headers.get("x-client-ip", "").strip()
        if visitor:
            return visitor
    forwarded = request.headers.get("x-forwarded-for", "")
    if os.getenv("TRUST_PROXY") == "1" and forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def _rate_limit(request: Request) -> None:
    """Allow RUNS_PER_HOUR agent runs per client IP."""
    window = _hits[client_ip(request)]
    now = time.time()
    while window and now - window[0] > 3600:
        window.popleft()
    if len(window) >= RUNS_PER_HOUR:
        raise HTTPException(429, "Too many runs from this address. Try again later.")
    window.append(now)


def _check_thread(thread_id: str) -> None:
    if not (0 < len(thread_id) <= 64) or not thread_id.replace("-", "").isalnum():
        raise HTTPException(400, "thread_id must be 1 to 64 letters, digits or dashes")


def context_for(thread_id: str) -> PauseContext:
    """The thread's own workspace file and the current local time."""
    now = datetime.now(ZoneInfo(settings.timezone)).replace(tzinfo=None)
    return PauseContext(
        now=now.isoformat(timespec="seconds"),
        timezone=settings.timezone,
        db_path=str(settings.data_dir / f"workspace-{thread_id}.sqlite"),
    )


def _config(thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "tags": ["pause"],
        "metadata": {"agent": "pause", "thread_id": thread_id},
    }


def _sse(events: Iterator[dict[str, Any]]) -> StreamingResponse:
    def body() -> Iterator[str]:
        for event in events:
            yield f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(body(), media_type="text/event-stream")


def _stream(payload: Any, thread_id: str) -> Iterator[dict[str, Any]]:
    """Run the agent, then send the workspace so the UI can redraw the calendar."""
    context = context_for(thread_id)
    stream = _agent.stream(payload, _config(thread_id), context=context, stream_mode=STREAM_MODES)
    yield from to_events(stream)
    yield {"type": "trace", "kind": "workspace", "snapshot": workspace_of(context).snapshot()}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "model": settings.model}


@app.get("/limits")
def limits() -> dict[str, Any]:
    caps = settings.limits
    return {
        "daily_calls_used": daily_cap.used(),
        "daily_calls_limit": daily_cap.limit,
        "model_calls_per_run": caps.model_calls_per_run,
        "tool_calls_per_run": caps.tool_calls_per_run,
        "tokens_per_run": caps.tokens_per_run,
        "seconds_per_run": caps.seconds_per_run,
        "gate_samples": settings.gate_samples,
    }


@app.post("/runs")
async def start_run(
    request: Request,
    thread_id: str = Form(...),
    message: str = Form(""),
    images: list[UploadFile] = File(default=[]),
    audio: UploadFile | None = File(default=None),
) -> StreamingResponse:
    """Start or continue a thread. Accepts text, up to 2 images, and one audio clip."""
    _check_thread(thread_id)
    if len(images) > MAX_IMAGES:
        raise HTTPException(400, f"at most {MAX_IMAGES} images")
    content: list[dict[str, Any]] = []
    if message.strip():
        content.append({"type": "text", "text": message.strip()[:2000]})
    try:
        for upload in images:
            content.append(image_block(clean_image(await upload.read(), upload.content_type or "")))
        if audio is not None:
            data = await audio.read()
            check_audio(data, audio.content_type or "")
            content.append(audio_block(data, audio.content_type or ""))
    except MediaError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not content:
        raise HTTPException(400, "send a message, an image or audio")
    _rate_limit(request)
    return _sse(_stream({"messages": [{"role": "user", "content": content}]}, thread_id))


class Decision(BaseModel):
    type: Literal["approve", "reject"]
    message: str | None = None


class Resume(BaseModel):
    decisions: list[Decision]


@app.post("/runs/{thread_id}/resume")
def resume_run(thread_id: str, body: Resume) -> StreamingResponse:
    """Answer a pending approval."""
    _check_thread(thread_id)
    if not _agent.get_state(_config(thread_id)).interrupts:
        raise HTTPException(409, "nothing is waiting for approval on this thread")
    decisions = [d.model_dump(exclude_none=True) for d in body.decisions]
    command: Command = Command(resume={"decisions": decisions})
    return _sse(_stream(command, thread_id))


@app.get("/workspace/{thread_id}")
def get_workspace(thread_id: str) -> dict[str, Any]:
    """The thread's calendar, tasks, contacts, notes and undo list."""
    _check_thread(thread_id)
    return workspace_of(context_for(thread_id)).snapshot()


@app.post("/workspace/{thread_id}/undo")
def undo(thread_id: str) -> dict[str, Any]:
    """Roll back the most recent change in this thread's workspace."""
    _check_thread(thread_id)
    workspace = workspace_of(context_for(thread_id))
    undone = workspace.undo_last()
    if undone is None:
        raise HTTPException(409, "nothing to undo")
    return {"undone": undone, "snapshot": workspace.snapshot()}
