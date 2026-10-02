from langchain_uncertainty_gate import (
    Thresholds,
    build_proposal,
    decide,
    default_canonicalize,
    default_question,
)

RISK = {"move_event": "reversible", "delete_event": "irreversible"}


def risk_of(name: str) -> str:
    return RISK.get(name, "none")


def proposal(*calls: tuple[str, dict]):
    tool_calls = [{"name": n, "args": a} for n, a in calls]
    return build_proposal(tool_calls, risk_of, default_canonicalize)


def test_canonicalize_ignores_case_and_spacing() -> None:
    a = proposal(("move_event", {"event_id": "E1", "new_start": " 2026-10-10T15:00 "}))
    b = proposal(("move_event", {"new_start": "2026-10-10t15:00", "event_id": "e1"}))
    assert a.key() == b.key()


def test_non_risky_calls_are_not_actions() -> None:
    p = proposal(("list_events", {}))
    assert p.actions == () and p.fallback == "list_events"
    assert proposal().fallback == "text"


def test_reversible_majority_acts() -> None:
    same = ("move_event", {"event_id": "e1", "new_start": "fri"})
    other = ("move_event", {"event_id": "e1", "new_start": "sat"})
    d = decide([proposal(same), proposal(other), proposal(same)], risk_of)
    assert (d.decision, d.risk, d.agreement) == ("act", "reversible", 0.667)
    assert d.majority_index == 0 and d.votes == [2, 1]


def test_reversible_three_way_split_asks_and_names_the_argument() -> None:
    props = [proposal(("move_event", {"event_id": "e1", "new_start": day})) for day in "abc"]
    d = decide(props, risk_of)
    assert d.decision == "ask"
    assert d.disagreements == [
        {"tool": "move_event", "argument": "new_start", "values": ["a", "b", "c"]}
    ]
    assert default_question(d) == "Before I do this I need one detail. new_start: a or b or c?"


def test_irreversible_needs_every_sample() -> None:
    a = ("delete_event", {"event_id": "e1"})
    b = ("delete_event", {"event_id": "e2"})
    assert decide([proposal(a)] * 3, risk_of).decision == "act"
    assert decide([proposal(a), proposal(a), proposal(b)], risk_of).decision == "ask"


def test_irreversible_with_no_majority_escalates() -> None:
    props = [proposal(("delete_event", {"event_id": e})) for e in ("e1", "e2", "e3")]
    d = decide(props, risk_of)
    assert d.decision == "escalate" and d.agreement == 0.333
    text = default_question(d)
    assert text.startswith("I will not choose here.") and "3. delete_event(event_id=e3)" in text


def test_one_irreversible_sample_raises_the_risk_level() -> None:
    move = ("move_event", {"event_id": "e1", "new_start": "fri"})
    d = decide(
        [proposal(move), proposal(move), proposal(("delete_event", {"event_id": "e1"}))], risk_of
    )
    assert d.risk == "irreversible" and d.decision == "ask"


def test_majority_that_takes_no_action_never_acts() -> None:
    move = ("move_event", {"event_id": "e1", "new_start": "fri"})
    d = decide([proposal(move), proposal(), proposal()], risk_of)
    assert d.decision == "ask" and d.options[1] == "take no action yet"


def test_custom_thresholds() -> None:
    props = [proposal(("move_event", {"event_id": "e1", "new_start": d})) for d in ("a", "a", "b")]
    strict = {"reversible": Thresholds(act=1.0, ask=0.0)}
    assert decide(props, risk_of, strict).decision == "ask"
