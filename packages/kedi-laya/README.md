# Kedi Laya

Local Laya decision models without a TypeSafe dependency. Pydantic AI and
LangChain are optional integrations, not required to import the client.

Use `kedi-decisions[laya-mlx]` on Apple Silicon or `kedi-decisions[laya]` for
upstream PyTorch. Add `langchain` for that framework. The current packages are
development versions in this monorepo and have not been published yet.

```python
from kedi_laya import LayaClient

with LayaClient("path/to/checkpoint", backend="mlx") as client:
    evaluator = client.as_evaluator(threshold=0.85)
    result = evaluator.evaluate_sync(
        state="The duplicate charge has been refunded.",
        schema={
            "type": "object",
            "properties": {"refunded": {"type": "boolean", "description": "Has the refund been completed?"}},
            "required": ["refunded"],
        },
    )
    print(result.values)
```

`LayaModel` and `LayaChatModel` are available from `kedi_laya` or their respective
`integrations.pydantic` and `integrations.langchain` modules. They accept a
checkpoint, `backend`, `threshold`, and `load_options`. A borrowed `client` can
be shared between models; closing a wrapper never closes a borrowed client.

`backend="auto"` prefers an installed MLX runtime on Apple Silicon, otherwise selects
Torch. Explicit `mlx` or `torch` makes deployment deterministic. Missing or
failing selected runtimes raise an error; there is no inference-error fallback
to another backend. Weights load lazily once per client. The provider identifier
is always `laya`, and `model.backend` reports the selected runtime.

MLX-owned models are checked against the actual tokenizer and context budget
before inference. Injected predictors and Torch retain their own context
handling. Real inference has been verified on the multilingual MLX checkpoint;
Torch must not be considered verified by the mocked loader tests.

The optional tool selector only executes parameterless calls through the host
framework. Parameterized tool choices raise `ToolCallProposed`; the model does
not invent arguments. Raw generated text and unsupported schemas are rejected.
