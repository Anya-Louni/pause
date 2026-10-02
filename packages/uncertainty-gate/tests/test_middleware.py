"""The gate inside a real create_agent loop, driven by a scripted model. No API calls."""

from typing import Any

import pytest
from fakes import EXECUTED, RISK, TOOLS, call, scripted
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_uncertainty_gate import METADATA_KEY, UncertaintyGateMiddleware

FRI = {"event_id": "e1", "new_start": "2026-10-09T15:00"}
SAT = {"event_id": "e1", "new_start": "2026-10-10T15:00"}
SUN = {"event_id": "e1", "new_start": "2026-10-11T15:00"}


@pytest.fixture(autouse=True)
def clear() -> None:
    EXECUTED.clear()


def ask(text: str = "move the dentist to friday") -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": text}]}


def agent(model: Any, **gate: Any) -> Any:
    return create_agent(model, tools=TOOLS, middleware=[UncertaintyGateMiddleware(RISK, **gate)])


def test_agreement_lets_the_action_run_once() -> None:
    model = scripted(
        call("move_event", FRI), call("move_event", FRI), call("move_event", FRI), AIMessage("done")
    )
    result = agent(model).invoke(ask())
    assert EXECUTED == [("move_event", FRI)]
    record = result["gate_log"][0]
    assert (record["decision"], record["agreement"], record["extra_calls"]) == ("act", 1.0, 2)


def test_disagreement_asks_and_runs_nothing() -> None:
    model = scripted(call("move_event", FRI), call("move_event", SAT), call("move_event", SUN))
    result = agent(model).invoke(ask())
    assert EXECUTED == []
    last = result["messages"][-1]
    assert isinstance(last, AIMessage) and not last.tool_calls
    assert "new_start" in last.text and "2026-10-09t15:00 or 2026-10-10t15:00" in last.text
    assert last.additional_kwargs[METADATA_KEY]["decision"] == "ask"


def test_irreversible_split_escalates_with_every_option() -> None:
    model = scripted(*[call("delete_event", {"event_id": e}) for e in ("e1", "e2", "e3")])
    result = agent(model).invoke(ask("delete the meeting"))
    assert EXECUTED == []
    assert result["gate_log"][0]["decision"] == "escalate"
    assert result["messages"][-1].text.count("delete_event(") == 3


def test_safe_tools_are_not_gated() -> None:
    model = scripted(call("list_events", {}), AIMessage("nothing today"))
    result = agent(model).invoke(ask("what is on today"))
    assert EXECUTED == [("list_events", {})]
    assert result.get("gate_log", []) == []
    assert len(model.bind_log) == 2  # one model call per step, no extra samples


def test_extra_samples_use_the_sampling_temperature() -> None:
    model = scripted(
        call("move_event", FRI), call("move_event", FRI), call("move_event", FRI), AIMessage("ok")
    )
    agent(model, temperature=0.9).invoke(ask())
    temperatures = [log.get("temperature") for log in model.bind_log[:3]]
    assert temperatures == [None, 0.9, 0.9]


def test_extra_calls_are_reported_for_budgets() -> None:
    charged: list[int] = []
    model = scripted(
        call("move_event", FRI), call("move_event", SAT), call("move_event", FRI), AIMessage("ok")
    )
    agent(model, on_extra_calls=charged.append).invoke(ask())
    assert charged == [2]


def test_the_models_own_question_is_kept() -> None:
    question = AIMessage("Do you mean this Friday or next Friday?")
    model = scripted(
        call("move_event", FRI), question, AIMessage("Do you mean this Friday or next Friday?")
    )
    result = agent(model).invoke(ask())
    assert EXECUTED == []
    assert result["messages"][-1].text == "Do you mean this Friday or next Friday?"
    assert result["gate_log"][0]["decision"] == "ask"


def test_a_clarifying_reply_is_gated_again() -> None:
    model = scripted(
        call("move_event", FRI),
        call("move_event", SAT),
        call("move_event", SUN),  # turn 1: ask
        call("move_event", FRI),
        call("move_event", FRI),
        call("move_event", FRI),  # turn 2: act
        AIMessage("moved"),
    )
    app = create_agent(
        model,
        tools=TOOLS,
        middleware=[UncertaintyGateMiddleware(RISK)],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t"}}
    app.invoke(ask(), config)
    assert EXECUTED == []
    result = app.invoke(ask("friday the 9th"), config)
    assert EXECUTED == [("move_event", FRI)]
    assert [r["decision"] for r in result["gate_log"]] == ["ask", "act"]


async def test_async_path_matches_sync() -> None:
    model = scripted(call("move_event", FRI), call("move_event", SAT), call("move_event", SUN))
    result = await agent(model).ainvoke(ask())
    assert EXECUTED == []
    assert result["gate_log"][0]["decision"] == "ask"

    EXECUTED.clear()
    model = scripted(
        call("move_event", FRI), call("move_event", FRI), call("move_event", FRI), AIMessage("ok")
    )
    result = await agent(model).ainvoke(ask())
    assert EXECUTED == [("move_event", FRI)]


def test_needs_at_least_two_samples() -> None:
    with pytest.raises(ValueError):
        UncertaintyGateMiddleware(RISK, samples=1)
