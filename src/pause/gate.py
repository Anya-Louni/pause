"""How Pause configures the uncertainty gate: what counts as the same action, and how to ask.

Agreement is measured on who, what and when. Free text such as a new event's
title or a note's wording is left out, because three samples rarely phrase it
the same way and the wording is not what makes an action wrong.
"""

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from langchain.agents.middleware import ModelRequest, ModelResponse

from langchain_uncertainty_gate import GateDecision, UncertaintyGateMiddleware, default_question
from pause.tools import RISK, PauseContext, workspace_of
from pause.workspace import NotFound, Workspace

FREE_TEXT = {"title", "text", "location"}
DATE_ARGS = {"start", "end", "new_start", "due"}

_workspace: ContextVar[Workspace | None] = ContextVar("pause_workspace", default=None)


def _minute(value: Any) -> Any:
    try:
        return datetime.fromisoformat(str(value)).replace(tzinfo=None).isoformat(timespec="minutes")
    except ValueError:
        return str(value).strip().lower()


def canonicalize(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """Normalize arguments so two proposals match when they mean the same action."""
    out: dict[str, Any] = {}
    for key, value in sorted(args.items()):
        if key in FREE_TEXT or value in (None, ""):
            continue
        if key == "due":
            out[key] = str(value)[:10]
        elif key in DATE_ARGS:
            out[key] = _minute(value)
        elif isinstance(value, list):
            out[key] = sorted(str(v).strip().lower() for v in value)
        else:
            out[key] = str(value).strip().lower()
    return out


def _when(iso: str) -> str:
    try:
        moment = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    text = f"{moment:%a} {moment.day} {moment:%b}"
    return text if "t" not in iso.lower() else f"{text} {moment:%H:%M}"


def _event(event_id: str) -> str:
    workspace = _workspace.get()
    try:
        event = workspace.event(event_id) if workspace else None
    except NotFound:
        event = None
    return f"'{event['title']}' on {_when(event['start'])}" if event else event_id


def _contact(contact_id: str) -> str:
    workspace = _workspace.get()
    contact = workspace.contact(contact_id) if workspace else None
    return f"{contact['name']} ({contact['team']})" if contact else contact_id


def _value(argument: str, value: Any) -> str:
    if value is None:
        return "none"
    if argument in DATE_ARGS:
        return _when(str(value))
    if argument == "event_id":
        return _event(str(value))
    if argument == "attendee_ids":
        return ", ".join(_contact(v) for v in value) or "nobody"
    return str(value)


def describe(tool: str, args: dict[str, Any]) -> str:
    """One canonical action as a short phrase a person can choose from."""
    if tool == "move_event":
        return f"move {_event(args.get('event_id', '?'))} to {_when(args.get('new_start', '?'))}"
    if tool == "delete_event":
        return f"delete {_event(args.get('event_id', '?'))}"
    if tool == "create_event":
        people = _value("attendee_ids", args.get("attendee_ids", []))
        return f"create an event on {_when(args.get('start', '?'))} with {people}"
    if tool == "add_task":
        return "add a task" + (
            f" due {_when(args['due'])}" if args.get("due") else " with no due date"
        )
    if tool == "complete_task":
        return f"complete task {args.get('task_id', '?')}"
    if tool == "save_note":
        return "save a note"
    return tool


LEAD = {
    "event_id": "Which event do you mean",
    "attendee_ids": "Who do you mean",
    "new_start": "Which time do you mean",
    "start": "Which time do you mean",
    "end": "When should it end",
    "due": "Which due date do you mean",
    "task_id": "Which task do you mean",
}


def write_question(decision: GateDecision) -> str:
    """One specific question that names the disagreement in the user's terms."""
    if decision.decision == "ask" and len(decision.disagreements) == 1:
        gap = decision.disagreements[0]
        choices = [_value(gap["argument"], v) for v in gap["values"]]
        lead = LEAD.get(gap["argument"], f"Which {gap['argument']} do you mean")
        return f"{lead}: {' or '.join(choices)}?"
    return default_question(decision)


class PauseGate(UncertaintyGateMiddleware):
    """The uncertainty gate with this workspace's lookups available while it describes options."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            RISK,
            canonicalize=canonicalize,
            describe=describe,
            write_question=write_question,
            **kwargs,
        )

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> Any:
        token = _workspace.set(_workspace_from(request))
        try:
            return super().wrap_model_call(request, handler)
        finally:
            _workspace.reset(token)

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> Any:
        token = _workspace.set(_workspace_from(request))
        try:
            return await super().awrap_model_call(request, handler)
        finally:
            _workspace.reset(token)


def _workspace_from(request: ModelRequest) -> Workspace | None:
    context = getattr(request.runtime, "context", None)
    return workspace_of(context) if isinstance(context, PauseContext) else None
