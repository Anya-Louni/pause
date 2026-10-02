"""Build the Pause agent with create_agent."""

from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from pause.budget import BudgetMiddleware, DailyCap
from pause.config import Limits, settings
from pause.gate import PauseGate
from pause.models import make_model, reliability_middleware
from pause.prompts import ASK_CLAUSE, BASE_PROMPT
from pause.tools import ALL_TOOLS, RISK, PauseContext

Setup = Literal["no_gate", "prompt_only", "gate"]

daily_cap = DailyCap(settings.data_dir / "daily_usage.json", settings.limits.provider_calls_per_day)
IRREVERSIBLE = [name for name, risk in RISK.items() if risk == "irreversible"]


def _tool_error(exc: Exception, request: Any) -> str:
    """Tell the model a tool failed without echoing internal details."""
    name = request.tool_call["name"]
    return f"{name} failed with {type(exc).__name__}. Check the arguments and try once more."


def build_middleware(model: BaseChatModel, limits: Limits, setup: Setup) -> list[Any]:
    """Limits first, then the gate, then approvals for anything that cannot be undone."""
    stack: list[Any] = [
        BudgetMiddleware(
            max_tokens=limits.tokens_per_run,
            max_seconds=limits.seconds_per_run,
            daily_cap=daily_cap,
        ),
        ModelCallLimitMiddleware(run_limit=limits.model_calls_per_run, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=limits.tool_calls_per_run, exit_behavior="end"),
    ]
    if setup == "gate":
        # Outside the retry middleware, so each sample gets its own retries.
        stack.append(
            PauseGate(
                samples=settings.gate_samples,
                temperature=settings.gate_temperature,
                on_extra_calls=lambda n: daily_cap.charge(n, strict=False),
            )
        )
    stack += [
        *reliability_middleware(),
        SummarizationMiddleware(model, trigger=("tokens", 40_000), keep=("messages", 20)),
        HumanInTheLoopMiddleware(
            interrupt_on={
                name: {"allowed_decisions": ["approve", "reject"]} for name in IRREVERSIBLE
            }
        ),
        ToolErrorMiddleware(on_error=_tool_error),
    ]
    return stack


def build_agent(
    *,
    model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    setup: Setup = "gate",
    limits: Limits | None = None,
    tools: list[Any] | None = None,
) -> Any:
    """Create the agent. `setup` picks the variant the eval compares."""
    chat_model = model or make_model()
    prompt = BASE_PROMPT + (ASK_CLAUSE if setup == "prompt_only" else "")
    return create_agent(
        chat_model,
        tools=tools if tools is not None else ALL_TOOLS,
        system_prompt=prompt,
        middleware=build_middleware(chat_model, limits or settings.limits, setup),
        context_schema=PauseContext,
        checkpointer=checkpointer,
        name="pause",
    )
