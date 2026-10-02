"""A fake chat model that replays scripted messages and records how it was bound."""

from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool


class ScriptedModel(GenericFakeChatModel):
    """Replays messages in order. Streaming is off because the base fake drops tool calls."""

    disable_streaming: bool = True
    bind_log: list[dict[str, Any]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        self.bind_log.append(kwargs)
        return self


def scripted(*messages: AIMessage) -> ScriptedModel:
    return ScriptedModel(messages=iter(messages), bind_log=[])


def call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


EXECUTED: list[tuple[str, dict[str, Any]]] = []


@tool
def move_event(event_id: str, new_start: str) -> str:
    """Move an event to a new start time."""
    EXECUTED.append(("move_event", {"event_id": event_id, "new_start": new_start}))
    return "moved"


@tool
def delete_event(event_id: str) -> str:
    """Delete an event."""
    EXECUTED.append(("delete_event", {"event_id": event_id}))
    return "deleted"


@tool
def list_events() -> str:
    """List events."""
    EXECUTED.append(("list_events", {}))
    return "[]"


TOOLS = [move_event, delete_event, list_events]
RISK = {"move_event": "reversible", "delete_event": "irreversible"}
