from langchain_uncertainty_gate import build_proposal, decide
from pause import gate
from pause.tools import RISK
from pause.workspace import Workspace


def risk_of(name: str) -> str:
    return RISK.get(name, "none")


def proposals(*calls: tuple[str, dict]):
    return [build_proposal([{"name": n, "args": a}], risk_of, gate.canonicalize) for n, a in calls]


def test_wording_and_offsets_do_not_count_as_disagreement() -> None:
    a = gate.canonicalize(
        "create_event",
        {
            "title": "Lunch with Sam",
            "start": "2026-10-09T12:30:00+01:00",
            "end": "2026-10-09T13:30",
            "attendee_ids": ["c2", "c1"],
        },
    )
    b = gate.canonicalize(
        "create_event",
        {
            "title": "Sam lunch",
            "start": "2026-10-09T12:30",
            "end": "2026-10-09T13:30:00",
            "attendee_ids": ["C1", "c2"],
            "location": "Canteen",
        },
    )
    assert (
        a
        == b
        == {"attendee_ids": ["c1", "c2"], "end": "2026-10-09T13:30", "start": "2026-10-09T12:30"}
    )


def test_due_dates_compare_as_dates() -> None:
    assert gate.canonicalize("add_task", {"title": "x", "due": "2026-10-09T00:00:00"}) == {
        "due": "2026-10-09"
    }
    assert gate.canonicalize("add_task", {"title": "x", "due": None}) == {}


def test_date_question_names_both_days(workspace: Workspace) -> None:
    token = gate._workspace.set(workspace)
    try:
        d = decide(
            proposals(
                ("move_event", {"event_id": "e3", "new_start": "2026-10-09T15:00"}),
                ("move_event", {"event_id": "e3", "new_start": "2026-10-16T15:00"}),
                ("move_event", {"event_id": "e3", "new_start": "2026-10-10T15:00"}),
            ),
            risk_of,
            describe=gate.describe,
        )
        assert d.decision == "ask"
        assert (
            gate.write_question(d)
            == "Which time do you mean: Fri 9 Oct 15:00 or Fri 16 Oct 15:00 or Sat 10 Oct 15:00?"
        )
    finally:
        gate._workspace.reset(token)


def test_event_question_shows_titles_and_days(workspace: Workspace) -> None:
    token = gate._workspace.set(workspace)
    try:
        d = decide(
            proposals(
                ("delete_event", {"event_id": "e1"}),
                ("delete_event", {"event_id": "e4"}),
                ("delete_event", {"event_id": "e1"}),
            ),
            risk_of,
            describe=gate.describe,
        )
        assert d.decision == "ask" and d.risk == "irreversible"
        assert gate.write_question(d) == (
            "Which event do you mean: 'Design review' on Mon 5 Oct 10:00 "
            "or 'Design review' on Thu 8 Oct 14:00?"
        )
    finally:
        gate._workspace.reset(token)


def test_contact_question_names_both_sams(workspace: Workspace) -> None:
    token = gate._workspace.set(workspace)
    try:
        base = {"title": "Coffee", "start": "2026-10-09T15:00", "end": "2026-10-09T16:00"}
        d = decide(
            proposals(
                ("create_event", {**base, "attendee_ids": ["c1"]}),
                ("create_event", {**base, "attendee_ids": ["c2"]}),
                ("create_event", {**base, "attendee_ids": ["c2"]}),
            ),
            risk_of,
            describe=gate.describe,
        )
        assert d.decision == "act"  # two of three agree, and it can be undone
        d.decision = "ask"
        assert (
            gate.write_question(d)
            == "Who do you mean: Sam Okafor (Design) or Sam Rivera (Finance)?"
        )
    finally:
        gate._workspace.reset(token)


def test_escalation_lists_described_options(workspace: Workspace) -> None:
    token = gate._workspace.set(workspace)
    try:
        d = decide(
            proposals(*[("delete_event", {"event_id": e}) for e in ("e1", "e3", "e4")]),
            risk_of,
            describe=gate.describe,
        )
        text = gate.write_question(d)
        assert d.decision == "escalate"
        assert "2. delete 'Dentist' on Wed 7 Oct 16:00" in text
    finally:
        gate._workspace.reset(token)
