import json
from typing import Any

import pytest
from conftest import call, scripted
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from pause.agent import build_agent
from pause.api import app as app_module

client = TestClient(app_module.app)


def events(text: str) -> list[dict[str, Any]]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def use_model(monkeypatch: pytest.MonkeyPatch, *messages: AIMessage) -> None:
    agent = build_agent(model=scripted(*messages), checkpointer=InMemorySaver(), setup="gate")
    monkeypatch.setattr(app_module, "_agent", agent)
    app_module._hits.clear()


def test_health_and_limits() -> None:
    assert client.get("/health").json()["ok"] is True
    assert client.get("/limits").json()["gate_samples"] == 3


def test_rejects_bad_input() -> None:
    assert client.post("/runs", data={"thread_id": "bad id!", "message": "x"}).status_code == 400
    assert client.post("/runs", data={"thread_id": "t1", "message": " "}).status_code == 400
    gif = {"images": ("a.gif", b"GIF89a", "image/gif")}
    assert client.post("/runs", data={"thread_id": "t1"}, files=gif).status_code == 400
    assert client.post("/runs/none/resume", json={"decisions": []}).status_code == 409


def test_each_thread_has_its_own_seeded_workspace() -> None:
    a = client.get("/workspace/apiA").json()
    assert len(a["events"]) == 6 and len(a["contacts"]) == 5
    assert client.post("/workspace/apiA/undo").status_code == 409


def test_run_streams_gate_decision_and_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    task = {"title": "Call the bank", "due": None}
    use_model(monkeypatch, *[call("add_task", task)] * 3, AIMessage("Added the task."))
    response = client.post(
        "/runs", data={"thread_id": "apiB", "message": "remind me to call the bank"}
    )
    stream = events(response.text)
    gate = next(e for e in stream if e.get("kind") == "gate")
    assert gate["decision"] == "act" and gate["agreement"] == 1.0
    snapshot = stream[-1]["snapshot"]
    assert snapshot["tasks"][-1]["title"] == "Call the bank"
    assert snapshot["undo"][0]["summary"] == "Added task 'Call the bank'"

    undone = client.post("/workspace/apiB/undo").json()
    assert undone["undone"] == "Added task 'Call the bank'"
    assert len(undone["snapshot"]["tasks"]) == 3


def test_delete_waits_for_approval_over_the_api(monkeypatch: pytest.MonkeyPatch) -> None:
    use_model(monkeypatch, *[call("delete_event", {"event_id": "e3"})] * 3, AIMessage("Deleted."))
    first = events(
        client.post("/runs", data={"thread_id": "apiC", "message": "delete dentist"}).text
    )
    assert any(e["type"] == "interrupt" for e in first)
    assert len(first[-1]["snapshot"]["events"]) == 6
    resumed = client.post("/runs/apiC/resume", json={"decisions": [{"type": "approve"}]})
    assert len(events(resumed.text)[-1]["snapshot"]["events"]) == 5


def test_rate_limit_per_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    use_model(monkeypatch, AIMessage("hi"), AIMessage("hi"))
    monkeypatch.setattr(app_module, "RUNS_PER_HOUR", 1)
    assert client.post("/runs", data={"thread_id": "rl1", "message": "a"}).status_code == 200
    assert client.post("/runs", data={"thread_id": "rl2", "message": "b"}).status_code == 429
