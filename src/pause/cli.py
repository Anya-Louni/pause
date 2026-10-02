"""Talk to Pause from the terminal and print the trace.

Usage:
    uv run python -m pause.cli "Move the dentist to next Friday at 3pm" --thread t1
    uv run python -m pause.cli "The 16th" --thread t1
    uv run python -m pause.cli --thread t1 --resume approve
"""

import argparse
import json
import sqlite3
from datetime import datetime
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from pause.agent import build_agent
from pause.config import settings
from pause.streaming import STREAM_MODES, to_events
from pause.tools import PauseContext, workspace_of


def _print(event: dict[str, Any]) -> None:
    kind = event.get("kind", "")
    if event["type"] == "trace" and kind == "tool_call":
        print(f"-> {event['name']}({json.dumps(event['args'])})")
    elif event["type"] == "trace" and kind == "tool_result":
        print(f"<- {event['name']} [{event['status']}] {event['detail'][:160]}")
    elif event["type"] == "trace" and kind == "gate":
        pairs = zip(event["votes"], event["options"], strict=True)
        votes = ", ".join(f"{v}x {o}" for v, o in pairs)
        head = f"{event['decision']} (agreement {event['agreement']}, {event['risk']})"
        print(f"   GATE {head}: {votes}")
    elif event["type"] == "trace" and kind in ("model", "limit"):
        print(f"   pause: {event['detail']}")
    elif event["type"] == "interrupt":
        names = [a["name"] for a in event["value"]["action_requests"]]
        print("APPROVAL NEEDED:", names)
    elif event["type"] == "error":
        print("ERROR:", event["message"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("message", nargs="?")
    parser.add_argument("--thread", required=True)
    parser.add_argument("--setup", default="gate", choices=["no_gate", "prompt_only", "gate"])
    parser.add_argument("--resume", choices=["approve", "reject"])
    parser.add_argument("--now", default=datetime.now().isoformat(timespec="seconds"))
    args = parser.parse_args()

    conn = sqlite3.connect(settings.data_dir / "checkpoints.sqlite", check_same_thread=False)
    agent = build_agent(checkpointer=SqliteSaver(conn), setup=args.setup)
    context = PauseContext(
        now=args.now,
        timezone=settings.timezone,
        db_path=str(settings.data_dir / f"workspace-{args.thread}.sqlite"),
    )
    config = {"configurable": {"thread_id": args.thread}, "tags": ["pause", args.setup]}
    payload: Any = (
        Command(resume={"decisions": [{"type": args.resume}]})
        if args.resume
        else {"messages": [{"role": "user", "content": args.message}]}
    )
    stream = agent.stream(payload, config, context=context, stream_mode=STREAM_MODES)
    for event in to_events(stream):
        _print(event)
    events = workspace_of(context).events()
    print("calendar:", [(e["id"], e["title"], e["start"]) for e in events])


if __name__ == "__main__":
    main()
