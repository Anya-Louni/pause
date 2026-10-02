# Vendored from claimcheck (github.com/Anya-Louni/claimcheck) at commit 4dca7b7.
# Kept as a copy so this repo runs on its own.

"""Turn a LangGraph stream into typed events for the web app.

Event types: trace, token, interrupt, result, error. Every event is a plain dict
with a "type" key so the API can send it as one Server-Sent Event.
"""

import json
import time
from collections.abc import Iterable, Iterator
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

STREAM_MODES = ["updates", "messages", "custom"]


def _summary(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


def _data(text: str, limit: int = 20_000) -> Any:
    """Parsed JSON tool output for the UI, when it is JSON and small enough."""
    if len(text) > limit:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _node_events(node: str, update: Any, ms: int) -> Iterator[dict[str, Any]]:
    if node == "__interrupt__":
        for item in update:
            yield {"type": "interrupt", "id": item.id, "value": _jsonable(item.value)}
        return
    if not isinstance(update, dict):
        return
    for message in update.get("messages", []) or []:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                yield {
                    "type": "trace",
                    "kind": "tool_call",
                    "name": call["name"],
                    "args": call["args"],
                    "id": call["id"],
                    "ms": ms,
                }
            if not message.tool_calls and message.text:
                yield {
                    "type": "trace",
                    "kind": "model",
                    "name": node,
                    "detail": _summary(message.text),
                    "ms": ms,
                }
        elif isinstance(message, ToolMessage):
            yield {
                "type": "trace",
                "kind": "tool_result",
                "name": message.name,
                "id": message.tool_call_id,
                "status": message.status,
                "detail": _summary(message.text),
                "data": _data(message.text),
                "ms": ms,
            }
    if node != "model" and node != "tools" and not update.get("messages"):
        yield {"type": "trace", "kind": "middleware", "name": node, "ms": ms}


def to_events(stream: Iterable[tuple[str, Any]]) -> Iterator[dict[str, Any]]:
    """Map (mode, chunk) pairs from agent.stream(stream_mode=STREAM_MODES) to events."""
    last = time.monotonic()
    report: Any = None
    try:
        for mode, chunk in stream:
            if mode == "messages":
                message, _meta = chunk
                if isinstance(message, AIMessageChunk) and message.text:
                    yield {"type": "token", "text": message.text}
            elif mode == "custom":
                fields = {k: v for k, v in chunk.items() if k != "type"}
                yield {**fields, "type": "trace", "kind": chunk.get("type", "custom")}
            elif mode == "updates":
                now = time.monotonic()
                ms = int((now - last) * 1000)
                last = now
                for node, update in chunk.items():
                    if isinstance(update, dict) and update.get("structured_response") is not None:
                        report = update["structured_response"]
                    yield from _node_events(node, update, ms)
        # Middleware may correct the report after the model writes it, so only the
        # last version is sent, once, when the run ends.
        if report is not None:
            yield {"type": "result", "report": _jsonable(report)}
    except Exception as exc:  # the UI needs a clean error event, not a broken stream
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
