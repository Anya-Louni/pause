from typing import Any

import pytest
from conftest import call as tool_call
from conftest import scripted as fake_model
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from pause.budget import BudgetMiddleware, DailyCap, DailyCapReached, summarize_progress


@tool
def noop() -> str:
    """Do nothing."""
    return "done"


def test_daily_cap_strict_and_lenient(tmp_path: Any) -> None:
    cap = DailyCap(tmp_path / "u.json", limit=2)
    assert cap.charge() == 1
    assert cap.charge() == 2
    with pytest.raises(DailyCapReached):
        cap.charge()
    assert cap.charge(strict=False) == 3


def test_summarize_progress_lists_finished_tools() -> None:
    messages = [
        HumanMessage("old"),
        ToolMessage("x", tool_call_id="1", name="old_tool"),
        HumanMessage("new"),
        ToolMessage("x", tool_call_id="2", name="get_claims"),
        ToolMessage("x", tool_call_id="3", name="probe_tool_calling"),
    ]
    expected = "Finished before the stop: get_claims, probe_tool_calling."
    assert summarize_progress(messages) == expected
    assert "No tool calls" in summarize_progress([HumanMessage("hi")])


def test_token_budget_stops_run_and_reports_progress(tmp_path: Any) -> None:
    heavy = tool_call("noop", {})
    heavy.usage_metadata = {"input_tokens": 900, "output_tokens": 200, "total_tokens": 1100}
    cap = DailyCap(tmp_path / "u.json", 100)
    middleware = BudgetMiddleware(max_tokens=1000, max_seconds=60, daily_cap=cap)
    agent = create_agent(
        fake_model(heavy, AIMessage("never")), tools=[noop], middleware=[middleware]
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    last = result["messages"][-1].text
    assert last.startswith("Stopped: token budget of 1000 used")
    assert "noop" in last


def test_daily_cap_blocks_before_model(tmp_path: Any) -> None:
    cap = DailyCap(tmp_path / "u.json", limit=1)
    cap.charge()
    middleware = BudgetMiddleware(max_tokens=10_000, max_seconds=60, daily_cap=cap)
    agent = create_agent(fake_model(AIMessage("hi")), tools=[], middleware=[middleware])
    result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    assert "daily cap" in result["messages"][-1].text


def test_resumed_run_starts_its_own_clock(tmp_path: Any) -> None:
    # A resumed run skips before_agent; before_model must start the clock itself.
    middleware = BudgetMiddleware(
        max_tokens=1000, max_seconds=60, daily_cap=DailyCap(tmp_path / "u.json", 10)
    )
    update = middleware.before_model({"messages": []}, None)  # type: ignore[arg-type]
    assert update is not None and update["run_tokens"] == 0 and update["run_started"] > 0
    assert (
        middleware.before_model({"messages": [], "run_started": update["run_started"]}, None)
        is None
    )  # type: ignore[arg-type]
