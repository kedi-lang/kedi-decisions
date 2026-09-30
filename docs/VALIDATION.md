# Decision Package Migration Validation

Verified on 2026-09-30 with Python 3.12 on Apple Silicon.

## Delivered Structure

| Distribution | Local version | Responsibility |
| --- | --- | --- |
| `kedi-decisions` | 0.1.0 | Shared contracts, schema planning, evidence, validation and framework bridges |
| `kedi-typesafe` | 0.3.0 | Jev SDK transport and compatibility with the existing public API |
| `kedi-laya` | 0.1.0 | Laya inference, explicit MLX/Torch backends and framework models |

The shared package is under `src/kedi_decisions/`; extensions are under
`packages/`. Both extensions use shared criteria objects. Laya does not import
or require TypeSafe SDK. All three packages are installed editable in the Kedi
development environment.

Kedi retains `kedi.typesafe` and `> import: typesafe`, and adds `kedi.laya` and
`> import: laya`. The canonical provider ID is `laya/<checkpoint>`. Backend
selection is separate: `auto`, `mlx`, or `torch`. The legacy `laya-mlx/` model
prefix remains accepted and explicitly selects MLX.

## Automated Verification

| Check | Result |
| --- | --- |
| Decision workspace tests | 230 passed; 1 hosted live test deselected |
| Existing TypeSafe tests included above | 167 passed |
| Focused Kedi regression suite | 514 passed |
| Workspace line/branch coverage | 98.97%; 98% gate passed |
| Ruff lint and format checks | Passed |
| Strict basedpyright, Python 3.10 target | 0 errors, 0 warnings |
| ty | Passed |
| Fresh workspace environment | `make check` and `make tests` passed |
| Root and workspace dependency checks | Passed |
| All three wheel and sdist builds | Passed |

The Kedi suite covers both adapters, captures, claims, decision metadata,
threshold overrides, profiles, public imports, examples and game integration.
Workspace tests cover response validation, tool selection and abstention,
structured output, streaming events, lifecycle ownership, backend selection,
inference serialization during cancellation and oversized MLX inputs.

The Python 3.10 type-check target is not an execution test on Python 3.10.
This validation did not run the complete Kedi test suite or a cross-OS matrix.

## Installed-Wheel Isolation

A separate virtual environment was populated from built wheels, not editable
source. Installing the core and Laya without extras required neither framework,
neither inference backend, nor `kedi-typesafe`/`typesafe_sdk`. An injected
predictor exercised the client successfully.

After adding Laya's Pydantic and LangChain extras, a Pydantic AI `Agent` and a
LangChain structured-output invocation both returned validated boolean outputs.
TypeSafe packages were still absent. This checks dependency isolation and
framework wiring, not model quality. Final wheel manifests include typing
markers; wheel dependencies and sdist membership were also inspected.

## Real Local Inference

The existing `laya-multilingual-mlx` checkpoint was used offline with
`laya-mlx==0.2.0`, `mlx==0.32.3`, `pydantic-ai-slim==2.45.0` and
`langchain==1.3.18`. No new model weights were downloaded.

Four triage cases ran once through each surface. Every surface produced the
expected department and outage decision in all four cases. Scores and raw
probabilities were also returned but were not independently accuracy-scored.

| Surface | Warm median |
| --- | --- |
| Shared decision API | 70 ms |
| Pydantic AI | 77 ms |
| LangChain | 89 ms |

One cold call took 1.87 seconds. These small sequential smoke measurements do
not establish a latency ranking between frameworks.

Kedi programs were then exercised with four cases, two adapters and two prompt
modes, for 16 calls. Decision-mode routing matched the expected team in 4/4
cases on each adapter, versus 3/4 with completion-mode prompts. Refund
threshold decisions remained weak: 0/4 on Pydantic and 3/4 on LangChain in
decision mode. Those differences are not an adapter accuracy claim; prompt
assembly differs and the sample is tiny. They show that correct integration
does not establish interchangeable model calibration or universal accuracy.

Evidence in the containing Kedi checkout:

- `tmp/decisions-monorepo-live.json`: direct API and framework runs.
- `tmp/decisions-monorepo-kedi-live.json`: Kedi program and prompt-mode runs.

These are checkout-local evidence files, not published benchmark artifacts.

## Limits and Release State

- Real MLX inference passed. The Torch loader contract is tested with injected
  runtimes; actual Torch checkpoint inference has not been verified here.
- TypeSafe compatibility was verified with its existing tests. No new hosted
  Jev request was made during this migration.
- Local Laya currently supports its decision question protocol, not arbitrary
  generative models or free-form text. Explicitly selected backends never
  silently switch to another runtime.
- The repository was moved from `typesafe/` to `decisions/` with its Git
  history preserved. The new monorepo repository is
  [kedi-lang/kedi-decisions](https://github.com/kedi-lang/kedi-decisions).
  The original TypeSafe repository is unchanged. No PyPI publication was performed.
- Published dependency resolution in Kedi remains compatible with TypeSafe
  0.2.1; local 0.3.0 is permitted by the updated dependency range. Laya's new
  distributions must be installed from this checkout until they are released.
