# Local Laya decisions

The optional `LayaClient` connects Laya-compatible `predict(state, questions)`
models to the same validated schema projection as Jev. It does not generate
prose, call a hosted service, or require `TYPESAFE_API_KEY`.

These packages are currently in the monorepo source checkout, not yet published.
From this repository run `uv sync --extra laya-mlx --extra langchain` on Apple
Silicon with Python 3.11+. Use `--extra laya` for the upstream PyTorch runtime.
Neither backend is a base dependency. Laya does not import the TypeSafe SDK.

## Python

```python
from typing import Literal
from pydantic import BaseModel
from pydantic_ai import Agent
from kedi_laya import LayaClient

class Route(BaseModel):
    team: Literal["billing", "technical"]

with LayaClient("aac6fef/laya-multilingual-mlx") as client:
    agent = Agent(client.as_pydantic_model(), output_type=Route)
    result = agent.run_sync("Please refund my duplicate charge.")
    print(result.output.team)
```

Use `client.as_langchain_model().with_structured_output(Route)` for LangChain.
Use `client.as_evaluator().evaluate_sync(state=..., schema=Route.model_json_schema())`
for a direct decision API; its result contains `values`, `metadata`, model name,
and reported token usage. An asynchronous `evaluate` method is also available.

The [complete example](../packages/kedi-laya/examples/local_laya.py) also captures a raw probability.
`ChoiceCriteria`, `Rubric`, `BooleanCriteria`, `Probability`, nested supported
models, and strict boolean thresholds share the existing schema implementation.
The default bool rule remains `probability > 0.85`, independently of an
application's choice-acceptance policy. Free-form strings remain unsupported.

## Other checkpoints and predictors

```python
client = LayaClient(
    "organization/checkpoint",
    backend="torch",
    load_options={"device": "cpu"},
)

# Supply an already-loaded Laya-compatible model instead of a loader.
client = LayaClient("my-checkpoint", backend="mlx", predictor=my_agent)
```

`backend` accepts `auto` (default), `mlx`, or `torch`. Auto selects installed MLX
on Apple Silicon, otherwise PyTorch. The checkpoint may be a Hub ID or a
local directory. `load_options` is forwarded unchanged to that backend's `load`;
only pass options supported by that runtime (MLX supports `revision`, for example).
No backend or hosted fallback is selected silently. Caller-supplied predictors
must implement Laya's result contract, including question types and probabilities.
Injection does not make unrelated models Laya-compatible.

The client loads weights lazily once and serializes predictions on that client.
Async calls move inference off the event loop. Cancelling the waiter does not
cancel an executing native kernel; subsequent predictions wait for it to finish.
Reuse a client/model for a session instead of constructing it per decision.
`close()` waits for active inference and drops the client's model reference.
It never calls `close` on a caller-owned predictor. Closing an evaluator does
not close the borrowed client: the client owner controls its lifetime.

## Context and evidence

- The built-in MLX loader checks the actual tokenizer, question/option budget,
  and checkpoint context window before prediction. It rejects silent truncation.
  Many Laya checkpoints have a 512-token window, including question/options.
- Injected predictors and upstream PyTorch loaders retain their own context
  handling. The MLX preflight is not a universal guarantee for those runtimes.
- Laya emits four-decimal probabilities. The evaluator accepts only the
  corresponding rounding error in their sum; raw probabilities are not rescaled.
  Invalid labels, missing answers, out-of-range values, and large sum errors fail.
- Choice/score confidence is supplied by Laya, not fabricated from its winner.
  Laya's entropy-based confidence is not calibrated to Jev's confidence. A shared
  numeric cutoff does not demonstrate equal reliability between providers.
- Laya's extra action head is not an approval decision and never authorizes tools.
- Metadata keeps `provider: laya` and the requested checkpoint name;
  the runtime's generic `laya-rl-agent` response label is not used as checkpoint ID.
  Laya uses the `decisions` metadata envelope. Jev keeps its existing `typesafe`
  envelope for compatibility; Kedi understands both.
- There is no generated-text token stream. Reported input counts come from Laya;
  missing usage stays unknown at the decision API, rather than being invented.

## Kedi

With the updated Kedi checkout and this package installed:

```kedi
> adapter: pydantic
> model: laya/aac6fef/laya-multilingual-mlx
> import: decisions

>> The team for "Please refund my duplicate charge" is [team: Literal["billing", "technical"]].
= <team>
```

`> adapter: langchain` uses the same surface. `laya-mlx/<checkpoint>` remains an
explicit MLX alias. These are decision models, not general-purpose agent harnesses.
Kedi does not attach its implicit artifact toolset to these models.

`> import: decisions` exposes the same shared criteria as `from kedi.decisions import ...`.
Laya uses `decision_threshold` (default `0.85`, strict greater-than) and
`decision_tool_call_threshold` (default `0.6`). Jev's existing `typesafe_*`
settings and the standalone provider package APIs are unchanged. Both providers
use the shared Kedi criteria import instead of provider-specific type modules.

Zero-argument handoffs can be selected and returned to the agent framework.
Parameterized tools raise `ToolCallProposed` for an explicit argument-producing
handler; the decision model never fabricates arbitrary arguments. The stream
interface delivers an already completed decision, not generated-text deltas.

The model wrappers declare `kedi_prompt_mode = "decision"`. The updated Kedi
runtime preserves template context and typed captures, but asks for decisions
instead of instructing the model to complete a grammatical sentence. Native
Pydantic AI and LangChain callers still control their own prompts; this marker
does not rewrite their input. It is not an accuracy guarantee.

Sources: [Laya MLX implementation](https://github.com/mizorewww/laya-mlx),
[multilingual checkpoint](https://huggingface.co/aac6fef/laya-multilingual-mlx),
[upstream Laya](https://pypi.org/project/laya/).
