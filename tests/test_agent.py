"""The whole agent with a scripted model. No API calls, no cost."""

import json
from typing import Any

from conftest import call, scripted
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from pause.agent import build_agent
from pause.config import Limits
from pause.streaming import STREAM_MODES, to_events
from pause.tools import PauseContext
from pause.workspace import Workspace

MOVE_FRI = {"event_id": "e3", "new_start": "2026-10-09T15:00"}
MOVE_NEXT_FRI = {"event_id": "e3", "new_start": "2026-10-16T15:00"}
MOVE_SAT = {"event_id": "e3", "new_start": "2026-10-10T15:00"}


def ask(text: str) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": text}]}


def cfg(thread: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread}}


def test_tools_read_the_seeded_workspace(context: PauseContext, workspace: Workspace) -> None:
    agent = build_agent(
        model=scripted(call("find_contact", {"name": "Sam"}), AIMessage("Two Sams.")),
        setup="no_gate",
    )
    result = agent.invoke(ask("who is Sam"), context=context)
    tool_message = next(m for m in result["messages"] if m.type == "tool")
    assert [c["id"] for c in json.loads(tool_message.content)] == ["c1", "c2"]


def test_resolve_date_uses_the_runs_clock(context: PauseContext) -> None:
    agent = build_agent(
        model=scripted(call("resolve_date", {"expression": "next Friday"}), AIMessage("?")),
        setup="no_gate",
    )
    result = agent.invoke(ask("when is next friday"), context=context)
    resolved = json.loads(next(m for m in result["messages"] if m.type == "tool").content)
    assert resolved["ambiguous"] and resolved["alternatives"] == ["2026-10-09", "2026-10-16"]


def test_without_the_gate_the_first_guess_runs(context: PauseContext, workspace: Workspace) -> None:
    agent = build_agent(
        model=scripted(call("move_event", MOVE_FRI), AIMessage("Moved.")), setup="no_gate"
    )
    agent.invoke(ask("move the dentist to next friday at 3pm"), context=context)
    assert workspace.event("e3")["start"] == "2026-10-09T15:00"


def test_gate_asks_when_samples_disagree(context: PauseContext, workspace: Workspace) -> None:
    model = scripted(
        call("move_event", MOVE_FRI),
        call("move_event", MOVE_NEXT_FRI),
        call("move_event", MOVE_SAT),
    )
    result = build_agent(model=model, setup="gate").invoke(
        ask("move the dentist to next friday at 3pm"), context=context
    )
    assert workspace.event("e3")["start"] == "2026-10-07T16:00"  # unchanged
    assert result["messages"][-1].text == (
        "Which time do you mean: Fri 9 Oct 15:00 or Fri 16 Oct 15:00 or Sat 10 Oct 15:00?"
    )
    assert result["gate_log"][0]["decision"] == "ask"


def test_gate_acts_then_the_change_can_be_undone(
    context: PauseContext, workspace: Workspace
) -> None:
    model = scripted(
        *[call("move_event", MOVE_FRI)] * 3, AIMessage("Moved the dentist to Friday at 3pm.")
    )
    result = build_agent(model=model, setup="gate").invoke(
        ask("move the dentist to this friday at 3pm"), context=context
    )
    assert workspace.event("e3")["start"] == "2026-10-09T15:00"
    assert result["gate_log"][0]["decision"] == "act"
    workspace.undo_last()
    assert workspace.event("e3")["start"] == "2026-10-07T16:00"


def test_reply_after_a_question_is_gated_again(context: PauseContext, workspace: Workspace) -> None:
    model = scripted(
        call("move_event", MOVE_FRI),
        call("move_event", MOVE_NEXT_FRI),
        call("move_event", MOVE_SAT),
        *[call("move_event", MOVE_NEXT_FRI)] * 3,
        AIMessage("Moved."),
    )
    agent = build_agent(model=model, setup="gate", checkpointer=InMemorySaver())
    agent.invoke(ask("move the dentist to next friday at 3pm"), cfg("clarify"), context=context)
    result = agent.invoke(ask("the 16th"), cfg("clarify"), context=context)
    assert workspace.event("e3")["start"] == "2026-10-16T15:00"
    assert [r["decision"] for r in result["gate_log"]] == ["ask", "act"]


def test_delete_needs_approval_even_when_samples_agree(
    context: PauseContext, workspace: Workspace
) -> None:
    model = scripted(*[call("delete_event", {"event_id": "e3"})] * 3, AIMessage("Deleted."))
    agent = build_agent(model=model, setup="gate", checkpointer=InMemorySaver())
    first = agent.invoke(ask("delete the dentist appointment"), cfg("del"), context=context)
    assert first["__interrupt__"] and workspace.event("e3")
    agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), cfg("del"), context=context)
    assert [e["id"] for e in workspace.events()] == ["e1", "e2", "e4", "e5", "e6"]


def test_rejected_delete_changes_nothing(context: PauseContext, workspace: Workspace) -> None:
    model = scripted(*[call("delete_event", {"event_id": "e3"})] * 3, AIMessage("Kept it."))
    agent = build_agent(model=model, setup="gate", checkpointer=InMemorySaver())
    agent.invoke(ask("delete the dentist appointment"), cfg("keep"), context=context)
    decision = {"decisions": [{"type": "reject", "message": "no"}]}
    agent.invoke(Command(resume=decision), cfg("keep"), context=context)
    assert workspace.event("e3")["title"] == "Dentist"


def test_bad_dates_are_refused_by_the_tool(context: PauseContext, workspace: Workspace) -> None:
    model = scripted(
        call("move_event", {"event_id": "e3", "new_start": "next friday"}), AIMessage(".")
    )
    result = build_agent(model=model, setup="no_gate").invoke(ask("x"), context=context)
    error = json.loads(next(m for m in result["messages"] if m.type == "tool").content)
    assert error["error"] == "invalid" and "resolve_date" in error["detail"]
    assert workspace.event("e3")["start"] == "2026-10-07T16:00"


def test_model_call_limit_stops_the_run(context: PauseContext) -> None:
    loop = [call("list_tasks", {}, f"c{i}") for i in range(6)]
    agent = build_agent(
        model=scripted(*loop), setup="no_gate", limits=Limits(model_calls_per_run=2)
    )
    result = agent.invoke(ask("x"), context=context)
    assert "limit" in result["messages"][-1].text.lower()


def test_stream_carries_the_gate_decision(context: PauseContext) -> None:
    model = scripted(
        call("move_event", MOVE_FRI),
        call("move_event", MOVE_NEXT_FRI),
        call("move_event", MOVE_SAT),
    )
    agent = build_agent(model=model, setup="gate")
    stream = agent.stream(ask("move the dentist"), context=context, stream_mode=STREAM_MODES)
    events = list(to_events(stream))
    gate = next(e for e in events if e.get("kind") == "gate")
    assert gate["decision"] == "ask" and gate["samples"] == 3 and len(gate["options"]) == 3
    assert all(e["type"] in {"trace", "token", "interrupt", "result", "error"} for e in events)
