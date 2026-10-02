"""Shared test setup: isolated data dir, a scripted fake model, and a seeded workspace."""

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="pause-test-"))
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ["FALLBACK_MODEL"] = ""
os.environ["LANGSMITH_TRACING"] = "false"

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from pause.tools import PauseContext, workspace_of  # noqa: E402
from pause.workspace import Workspace  # noqa: E402

NOW = "2026-10-06T09:00:00"  # a Tuesday. The seeded week runs Mon 5 to Fri 9 Oct.


class ScriptedModel(GenericFakeChatModel):
    """Replays messages in order. Streaming is off because the base fake drops tool calls."""

    disable_streaming: bool = True

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self


def scripted(*messages: AIMessage) -> ScriptedModel:
    return ScriptedModel(messages=iter(messages))


def call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture
def context(tmp_path: Path) -> PauseContext:
    return PauseContext(now=NOW, timezone="Europe/London", db_path=str(tmp_path / "ws.sqlite"))


@pytest.fixture
def workspace(context: PauseContext) -> Workspace:
    return workspace_of(context)
