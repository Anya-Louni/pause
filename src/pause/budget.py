# Vendored from claimcheck (github.com/Anya-Louni/claimcheck) at commit 4dca7b7.
# Kept as a copy so this repo runs on its own.

"""Per-run budget middleware and a daily cap on provider calls.

The built-in call-limit middleware counts model and tool calls. This middleware
adds what they do not cover: estimated tokens, wall-clock time, and a daily cap
shared by the agent and the probes, so one busy day cannot drain the free tier.
"""

import json
import threading
import time
from datetime import date
from pathlib import Path
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime


class DailyCapReached(RuntimeError):
    """Raised when the daily provider call cap is used up."""


class DailyCap:
    """Counts provider calls per calendar day in a small JSON file."""

    def __init__(self, path: Path, limit: int) -> None:
        self.path = path
        self.limit = limit
        self._lock = threading.Lock()

    def _read(self) -> dict[str, int]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text())

    def used(self) -> int:
        """Return calls used today."""
        with self._lock:
            return self._read().get(date.today().isoformat(), 0)

    def charge(self, calls: int = 1, *, strict: bool = True) -> int:
        """Record calls for today and return the new total.

        With strict=True the calls are refused, and DailyCapReached raised, when
        they would pass the cap. Use strict=False to record a call that already ran.
        """
        with self._lock:
            data = self._read()
            today = date.today().isoformat()
            new_total = data.get(today, 0) + calls
            if strict and new_total > self.limit:
                raise DailyCapReached(f"daily cap of {self.limit} provider calls reached")
            self.path.write_text(json.dumps({today: new_total}))
            return new_total


class BudgetState(AgentState):
    """Private run counters. Untracked values reset on every new run."""

    run_tokens: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]
    run_started: NotRequired[Annotated[float, UntrackedValue, PrivateStateAttr]]


def summarize_progress(messages: list[Any]) -> str:
    """Describe which tools finished since the last user message."""
    done: list[str] = []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage):
            done.append(message.name or "tool")
    if not done:
        return "No tool calls finished before the stop."
    return "Finished before the stop: " + ", ".join(reversed(done)) + "."


class BudgetMiddleware(AgentMiddleware[BudgetState]):
    """Stops a run cleanly when tokens, time, or the daily cap run out."""

    state_schema = BudgetState

    def __init__(self, *, max_tokens: int, max_seconds: int, daily_cap: DailyCap) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.max_seconds = max_seconds
        self.daily_cap = daily_cap

    def before_agent(self, state: BudgetState, runtime: Runtime) -> dict[str, Any]:
        return {"run_tokens": 0, "run_started": time.monotonic()}

    def _limit_reason(self, state: BudgetState) -> str | None:
        elapsed = time.monotonic() - state.get("run_started", time.monotonic())
        if state.get("run_tokens", 0) >= self.max_tokens:
            return f"token budget of {self.max_tokens} used"
        if elapsed >= self.max_seconds:
            return f"time budget of {self.max_seconds}s used"
        if self.daily_cap.used() >= self.daily_cap.limit:
            return f"daily cap of {self.daily_cap.limit} provider calls reached"
        return None

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: BudgetState, runtime: Runtime) -> dict[str, Any] | None:
        reason = self._limit_reason(state)
        if reason is None:
            # A run resumed after an approval skips before_agent, so start its clock here.
            if "run_started" not in state:
                return {"run_started": time.monotonic(), "run_tokens": 0}
            return None
        text = f"Stopped: {reason}. {summarize_progress(state['messages'])}"
        return {"jump_to": "end", "messages": [AIMessage(content=text)]}

    def after_model(self, state: BudgetState, runtime: Runtime) -> dict[str, Any]:
        last = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
        usage = getattr(last, "usage_metadata", None) or {}
        tokens = state.get("run_tokens", 0) + usage.get("total_tokens", 0)
        daily = self.daily_cap.charge(1, strict=False)
        emit(
            {
                "type": "budget",
                "tokens": tokens,
                "max_tokens": self.max_tokens,
                "seconds": round(time.monotonic() - state.get("run_started", time.monotonic()), 1),
                "max_seconds": self.max_seconds,
                "daily_calls": daily,
                "daily_limit": self.daily_cap.limit,
            }
        )
        return {"run_tokens": tokens}


def emit(event: dict[str, Any]) -> None:
    """Send a custom stream event when running inside a graph, else do nothing."""
    try:
        get_stream_writer()(event)
    except RuntimeError:
        pass
