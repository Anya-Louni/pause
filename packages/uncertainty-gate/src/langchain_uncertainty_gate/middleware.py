"""The uncertainty gate as LangChain agent middleware.

Before a risky tool call runs, the gate asks the model for a few more proposals
for the same turn, measures how much they agree, and then lets the action
through, replaces it with one clarifying question, or hands the choice to a
human. It sits outside the model, so it works the same for models that never
ask on their own.
"""

from __future__ import annotations

import operator
from collections.abc import Awaitable, Callable, Mapping
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer
from langgraph.types import Command

from langchain_uncertainty_gate.decision import (
    DEFAULT_THRESHOLDS,
    Canonicalize,
    Describe,
    GateDecision,
    Proposal,
    Risk,
    Thresholds,
    build_proposal,
    decide,
    default_canonicalize,
    default_describe,
    default_question,
)

METADATA_KEY = "uncertainty_gate"


class GateState(AgentState):
    """Adds a running log of gate decisions to the agent state."""

    gate_log: NotRequired[Annotated[list[dict[str, Any]], operator.add]]


def _ai_message(response: ModelResponse) -> AIMessage | None:
    for message in reversed(response.result):
        if isinstance(message, AIMessage):
            return message
    return None


class UncertaintyGateMiddleware(AgentMiddleware[GateState]):
    """Sample risky proposals, score agreement, then act, ask or escalate.

    Args:
        risk: Risk level per tool name, as a mapping or a function. Tools not
            listed count as "none" and are never gated.
        samples: Proposals to compare for a risky turn, the first one included.
        temperature: Sampling temperature for the extra proposals.
        thresholds: Agreement needed per risk level. See `DEFAULT_THRESHOLDS`.
        canonicalize: Normalizes tool arguments so agreement is measured on
            meaning. Receives the tool name and its arguments.
        describe: Renders one canonical action as a short phrase for questions.
        write_question: Builds the clarifying or escalation text from a decision.
        on_extra_calls: Called with the number of extra model calls the gate
            made, so a budget can charge them.
    """

    state_schema = GateState

    def __init__(
        self,
        risk: Mapping[str, Risk] | Callable[[str], Risk],
        *,
        samples: int = 3,
        temperature: float = 0.7,
        thresholds: Mapping[str, Thresholds] | None = None,
        canonicalize: Canonicalize = default_canonicalize,
        describe: Describe = default_describe,
        write_question: Callable[[GateDecision], str] = default_question,
        on_extra_calls: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__()
        if samples < 2:
            raise ValueError("samples must be at least 2 to measure agreement")
        self._risk_of: Callable[[str], Risk] = (
            risk if callable(risk) else lambda name: risk.get(name, "none")
        )
        self.samples = samples
        self.temperature = temperature
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.canonicalize = canonicalize
        self.describe = describe
        self.write_question = write_question
        self.on_extra_calls = on_extra_calls

    # Shared steps -----------------------------------------------------------

    def _proposal(self, response: ModelResponse) -> Proposal:
        message = _ai_message(response)
        calls = [dict(c) for c in message.tool_calls] if message else []
        return build_proposal(calls, self._risk_of, self.canonicalize)

    def _is_risky(self, response: ModelResponse) -> bool:
        return bool(self._proposal(response).actions)

    def _sampling_request(self, request: ModelRequest) -> ModelRequest:
        settings = {**request.model_settings, "temperature": self.temperature}
        return request.override(model_settings=settings)

    def _finish(self, responses: list[ModelResponse]) -> ExtendedModelResponse:
        decision = decide(
            [self._proposal(r) for r in responses], self._risk_of, self.thresholds, self.describe
        )
        decision.extra_calls = len(responses) - 1
        chosen = responses[decision.majority_index]
        chosen_message = _ai_message(chosen)
        if decision.decision == "act":
            result = chosen
        elif chosen_message is not None and not chosen_message.tool_calls and chosen_message.text:
            # The majority already chose to ask in its own words. Keep its question.
            decision.question = chosen_message.text
            result = chosen
        else:
            decision.question = self.write_question(decision)
            result = ModelResponse(result=[AIMessage(content=decision.question)])
        record = decision.to_dict()
        message = _ai_message(result)
        if message is not None:
            message.additional_kwargs[METADATA_KEY] = record
        if self.on_extra_calls is not None:
            self.on_extra_calls(decision.extra_calls)
        _emit({"type": "gate", **record})
        return ExtendedModelResponse(
            model_response=result, command=Command(update={"gate_log": [record]})
        )

    # Hooks ------------------------------------------------------------------

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse | ExtendedModelResponse:
        first = handler(request)
        if not self._is_risky(first):
            return first
        sampling = self._sampling_request(request)
        responses = [first] + [handler(sampling) for _ in range(self.samples - 1)]
        return self._finish(responses)

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse | ExtendedModelResponse:
        first = await handler(request)
        if not self._is_risky(first):
            return first
        sampling = self._sampling_request(request)
        responses = [first]
        for _ in range(self.samples - 1):
            responses.append(await handler(sampling))
        return self._finish(responses)


def _emit(event: dict[str, Any]) -> None:
    """Send a custom stream event when running inside a graph, else do nothing."""
    try:
        get_stream_writer()(event)
    except RuntimeError:
        pass
