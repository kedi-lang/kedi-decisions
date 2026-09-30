# Kedi Decisions

Provider-neutral typed decisions, with independent Jev and Laya integrations.

The root `kedi-decisions` package contains schema planning, criteria, validation,
decision evidence and optional framework bridges. It imports neither TypeSafe
SDK nor MLX/PyTorch. Provider packages live in `packages/` and can be installed
independently. Kedi itself is not required to use them.

| Install | Includes |
| --- | --- |
| `kedi-decisions` | Shared contracts and validation only |
| `kedi-decisions[typesafe]` | Jev and its native Pydantic AI integration |
| `kedi-decisions[laya]` | Laya, upstream PyTorch runtime, Pydantic AI |
| `kedi-decisions[laya-mlx]` | Laya, Apple Silicon MLX runtime, Pydantic AI |
| `kedi-decisions[langchain]` | Optional shared LangChain integration |

Jev's LangChain transport is installed with `kedi-typesafe[langchain]`.
The shared `langchain` extra alone does not install provider-specific SDKs.

These package versions are not yet published to PyPI. Clone the monorepo to
work from source:

```sh
git clone https://github.com/kedi-lang/kedi-decisions.git
cd kedi-decisions
uv sync --locked --group dev
```

Add `--extra laya-mlx` or `--extra laya` only on a host where that runtime is
needed. Installation does not download model weights.

## Layout

- `src/kedi_decisions/`: shared contract, schema, evidence and integrations.
- `packages/kedi-typesafe/`: TypeSafe SDK conversion and Jev transport.
- `packages/kedi-laya/`: local inference, MLX/Torch loading and model wrappers.
- `tests/`: package isolation and cross-provider contracts.

Kedi users keep `kedi.typesafe` and `> import: typesafe`. Laya uses `kedi.laya`
and `> import: laya`. These imports expose criteria but never select a model or
start inference. The provider is `laya`; its backend is a separate property.

## Compatibility

`kedi_typesafe` retains its public exports, evaluator/client ownership rules,
strict default boolean threshold of 0.85, SDK-native question plans and metadata
envelope. Criteria are aliases of the shared objects, not duplicate classes.
Existing serialized `x-typesafe` schema annotations and route identifiers are
retained for compatibility; they do not imply a dependency on the TypeSafe SDK.

Jev and Laya can return similar decision shapes without having interchangeable
confidence calibration. A finite-choice interface is not free-form extraction.
Unsupported schema types fail before model invocation.

## Checks

`make check`, `make tests` and `make build` validate the workspace. Live validation
programs reside with their provider packages and are excluded from normal CI.
`make coverage` enforces a 98% combined line/branch coverage floor.
