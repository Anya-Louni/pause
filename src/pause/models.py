# Vendored from claimcheck (github.com/Anya-Louni/claimcheck) at commit 4dca7b7.
# Kept as a copy so this repo runs on its own.

"""Model factory, retry and fallback. The only place a chat model is created."""

from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
)
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from pause.config import settings

# Free-tier friendly pace for the agent's own calls: about 15 requests a minute.
_agent_limiter = InMemoryRateLimiter(
    requests_per_second=0.25, check_every_n_seconds=0.1, max_bucket_size=2
)


def make_model(model: str | None = None, **kwargs: Any) -> BaseChatModel:
    """Create a chat model from a "provider:name" string. Defaults to settings.model.

    The client's own retries are turned off so ModelRetryMiddleware owns retry
    and backoff and every attempt shows up in the trace.
    """
    kwargs.setdefault("max_retries", 0)
    kwargs.setdefault("rate_limiter", _agent_limiter)
    return init_chat_model(model or settings.model, **kwargs)


def reliability_middleware() -> list[AgentMiddleware]:
    """Retry with exponential backoff on transient errors such as 429, then fall back."""
    stack: list[AgentMiddleware] = [
        ModelRetryMiddleware(max_retries=3, initial_delay=2.0, backoff_factor=2.0, max_delay=30.0)
    ]
    if settings.fallback_model:
        stack.append(ModelFallbackMiddleware(make_model(settings.fallback_model)))
    return stack
