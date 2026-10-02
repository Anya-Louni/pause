"""Pure decision logic: canonicalize proposals, measure agreement, choose act, ask or escalate.

Nothing here touches a model, so every rule can be unit tested with plain data.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Risk = Literal["none", "reversible", "irreversible"]
Decision = Literal["act", "ask", "escalate"]
RISK_ORDER: dict[str, int] = {"none": 0, "reversible": 1, "irreversible": 2}

Canonicalize = Callable[[str, dict[str, Any]], dict[str, Any]]
Describe = Callable[[str, dict[str, Any]], str]


@dataclass(frozen=True)
class Thresholds:
    """Agreement share needed to act, and to ask instead of escalating."""

    act: float
    ask: float


DEFAULT_THRESHOLDS: dict[str, Thresholds] = {
    # Reversible actions can be undone, so a clear majority is enough to act.
    "reversible": Thresholds(act=0.66, ask=0.0),
    # Irreversible actions need every sample to agree. A split with no majority goes to a human.
    "irreversible": Thresholds(act=1.0, ask=0.5),
}


@dataclass(frozen=True)
class Action:
    """One tool call a proposal wants to make, after canonicalization."""

    tool: str
    args: dict[str, Any]

    def key(self) -> str:
        return json.dumps([self.tool, self.args], sort_keys=True, default=str)


@dataclass(frozen=True)
class Proposal:
    """What one sample proposed: its risky tool calls, or nothing risky."""

    actions: tuple[Action, ...]
    fallback: str = ""

    def key(self) -> str:
        if not self.actions:
            return f"no-risky-action:{self.fallback}"
        return "|".join(sorted(a.key() for a in self.actions))


@dataclass
class GateDecision:
    """The gate's verdict for one model turn. Serializable, so UIs and logs can show it."""

    decision: Decision
    risk: Risk
    agreement: float
    samples: int
    majority_index: int
    options: list[str]
    votes: list[int]
    disagreements: list[dict[str, Any]] = field(default_factory=list)
    question: str | None = None
    extra_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_canonicalize(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """Trim and lowercase strings so proposals differing only in case or spacing match."""

    def norm(value: Any) -> Any:
        if isinstance(value, str):
            return " ".join(value.split()).lower()
        if isinstance(value, list):
            return [norm(v) for v in value]
        if isinstance(value, dict):
            return {k: norm(v) for k, v in sorted(value.items())}
        return value

    return {k: norm(v) for k, v in sorted(args.items())}


def default_describe(tool: str, args: dict[str, Any]) -> str:
    inner = ", ".join(f"{k}={v}" for k, v in args.items())
    return f"{tool}({inner})"


def build_proposal(
    tool_calls: list[dict[str, Any]],
    risk_of: Callable[[str], Risk],
    canonicalize: Canonicalize,
) -> Proposal:
    """Keep only the risky calls of one sample, canonicalized."""
    risky = [c for c in tool_calls if risk_of(c["name"]) != "none"]
    if not risky:
        names = sorted(c["name"] for c in tool_calls)
        return Proposal(actions=(), fallback=",".join(names) or "text")
    return Proposal(
        tuple(Action(c["name"], canonicalize(c["name"], dict(c["args"]))) for c in risky)
    )


def proposal_risk(proposal: Proposal, risk_of: Callable[[str], Risk]) -> Risk:
    risks = [risk_of(a.tool) for a in proposal.actions] or ["none"]
    return max(risks, key=lambda r: RISK_ORDER[r])


def argument_disagreements(proposals: list[Proposal]) -> list[dict[str, Any]]:
    """For samples that picked the same single tool, list the arguments they disagree on."""
    single = [p.actions[0] for p in proposals if len(p.actions) == 1]
    if len(single) < 2 or len({a.tool for a in single}) != 1:
        return []
    out = []
    for name in sorted({k for a in single for k in a.args}):
        values: list[Any] = []
        for action in single:
            value = action.args.get(name)
            if value not in values:
                values.append(value)
        if len(values) > 1:
            out.append({"tool": single[0].tool, "argument": name, "values": values})
    return out


def decide(
    proposals: list[Proposal],
    risk_of: Callable[[str], Risk],
    thresholds: Mapping[str, Thresholds] = DEFAULT_THRESHOLDS,
    describe: Describe = default_describe,
) -> GateDecision:
    """Score agreement across proposals and choose act, ask or escalate.

    Agreement is the share of samples that match the most common proposal. The
    risk level is the highest any sample proposed, so one sample reaching for an
    irreversible tool is enough to apply the stricter thresholds.
    """
    keys = [p.key() for p in proposals]
    counts = Counter(keys)
    majority_key, majority_votes = max(counts.items(), key=lambda kv: (kv[1], -keys.index(kv[0])))
    majority_index = keys.index(majority_key)
    agreement = majority_votes / len(proposals)
    risk = max((proposal_risk(p, risk_of) for p in proposals), key=lambda r: RISK_ORDER[r])

    distinct: list[Proposal] = []
    for proposal in proposals:
        if proposal.key() not in [d.key() for d in distinct]:
            distinct.append(proposal)
    options = [
        " and ".join(describe(a.tool, a.args) for a in p.actions)
        if p.actions
        else "take no action yet"
        for p in distinct
    ]
    votes = [counts[p.key()] for p in distinct]

    rule = thresholds.get(risk, Thresholds(act=1.0, ask=0.0))
    majority_is_action = bool(proposals[majority_index].actions)
    if agreement >= rule.act and majority_is_action:
        decision: Decision = "act"
    elif agreement >= rule.ask:
        decision = "ask"
    else:
        decision = "escalate"
    return GateDecision(
        decision=decision,
        risk=risk,
        agreement=round(agreement, 3),
        samples=len(proposals),
        majority_index=majority_index,
        options=options,
        votes=votes,
        disagreements=argument_disagreements(proposals),
    )


def default_question(decision: GateDecision) -> str:
    """One specific question naming the disagreement, built without a model call."""
    if decision.decision == "escalate":
        lines = [f"{i + 1}. {option}" for i, option in enumerate(decision.options)]
        return (
            "I will not choose here. This cannot be undone and my readings of the request differ:\n"
            + "\n".join(lines)
            + "\nWhich one do you want, if any?"
        )
    if len(decision.disagreements) == 1:
        gap = decision.disagreements[0]
        values = " or ".join(str(v) for v in gap["values"])
        return f"Before I do this I need one detail. {gap['argument']}: {values}?"
    if len(decision.options) == 1:
        return f"I am not sure enough to do this yet: {decision.options[0]}. Should I go ahead?"
    return "I can read this more than one way: " + " or ".join(decision.options) + ". Which one?"
