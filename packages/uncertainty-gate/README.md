# langchain-uncertainty-gate

Agent middleware for LangChain that stops a model from confidently doing the wrong thing.

Before a risky tool call runs, the gate asks the model for a few more proposals for the same turn, measures how much they agree, and then does one of three things:

- **act**: the proposals agree, so the tool call goes through unchanged.
- **ask**: they disagree, so the tool call is replaced by one clarifying question that names the disagreement.
- **escalate**: there is no majority on an irreversible action, so the user gets every competing option and the agent does nothing.

It sits outside the model. It does not rely on the model choosing to ask, which matters because some models never do.

Status: not published to PyPI yet. Install from this folder.

## Use

```python
from langchain.agents import create_agent
from langchain_uncertainty_gate import UncertaintyGateMiddleware

gate = UncertaintyGateMiddleware(
    {"move_event": "reversible", "delete_event": "irreversible"},
)
agent = create_agent("google_genai:gemini-3.1-flash-lite", tools=tools, middleware=[gate])
```

Tools that are not listed count as `none` and are never gated, so reads cost nothing extra.

Every decision is recorded three ways:

- `state["gate_log"]`: a list with one entry per gated turn.
- `AIMessage.additional_kwargs["uncertainty_gate"]` on the message the gate returned.
- A custom stream event of type `gate`, for live UIs (`stream_mode="custom"`).

An entry looks like this:

```python
{
    "decision": "ask",
    "risk": "reversible",
    "agreement": 0.333,
    "samples": 3,
    "options": ["move_event(event_id=e3, new_start=2026-10-09t15:00)", "..."],
    "votes": [1, 1, 1],
    "disagreements": [{"tool": "move_event", "argument": "new_start", "values": ["...", "..."]}],
    "question": "Before I do this I need one detail. new_start: ... or ...?",
    "extra_calls": 2,
}
```

## Options

| Argument | Default | Meaning |
|---|---|---|
| `risk` | required | Mapping or function from tool name to `"none"`, `"reversible"` or `"irreversible"`. |
| `samples` | `3` | Proposals compared on a risky turn, the first one included. |
| `temperature` | `0.7` | Passed as a model setting for the extra proposals. Models with fixed sampling ignore it. |
| `thresholds` | see below | Agreement needed per risk level. |
| `canonicalize` | lowercase, trim | `(tool, args) -> args`. Put arguments in one form so agreement is measured on meaning, for example dates as ISO and names as ids. |
| `describe` | `tool(arg=value)` | `(tool, args) -> str`. How one option reads in a question. |
| `write_question` | built in | `(GateDecision) -> str`. The clarifying or escalation text. |
| `on_extra_calls` | `None` | Called with the number of extra model calls, so a budget can charge them. |

Default thresholds, as the share of proposals that match the most common one:

| Risk | Act when | Ask when | Escalate when |
|---|---|---|---|
| reversible | at least 0.66 | below 0.66 | never |
| irreversible | 1.0 | at least 0.5 | below 0.5 |

The risk level of a turn is the highest any proposal reached. One sample reaching for an irreversible tool is enough to apply the stricter row.

## How it works

The gate implements `wrap_model_call` and `awrap_model_call`. It calls the handler once. If the response has no risky tool call, it returns it untouched. Otherwise it calls the handler again `samples - 1` times, canonicalizes each proposal's risky calls, and counts matches. On `act` it returns one of the sampled responses as it is, so no proposal is wasted. On `ask` or `escalate` it returns an `AIMessage` with no tool calls, which ends the turn. If the majority of proposals already asked the user something in plain text, that text is kept instead of a generated question.

When the user replies, the next model call is gated again from scratch. A reply cannot widen what the agent is allowed to do.

## Cost

A gated turn costs `samples - 1` extra model calls. Turns that only read cost nothing. Place the gate before retry middleware in the list, so each sample gets its own retries.

## Known risks and limits

- **Agreement is not correctness.** Three samples can agree on the same wrong reading. The gate catches ambiguity the model is unsure about, not errors it is sure about.
- **Asking can be exploited.** A clarifying question invites more user-controlled text into the conversation. If tool output or a document contains an injected instruction, a question gives it another chance to steer the agent. The gate re-checks the turn after every reply, but it does not detect injection. Treat fetched content as untrusted in your prompt and tools as well.
- **Canonicalization decides what "same" means.** With the default, two phrasings of a free-text argument count as different. Supply a `canonicalize` that drops or normalizes free text.
- Samples run one after another, not in parallel.

## Tests

```bash
uv run pytest packages/uncertainty-gate/tests
```

The tests use a scripted fake chat model, so they need no API key.

## Listing description

Uncertainty gate: samples risky tool calls a few times, measures agreement, then acts, asks one clarifying question, or hands the choice to a human.

## License

MIT
