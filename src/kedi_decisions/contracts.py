"""Provider-neutral wire values for constrained decision requests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeAlias

JSONValue: TypeAlias = str | int | float | bool | None | Sequence[Any] | Mapping[str, Any]


MediaKind: TypeAlias = Literal["image", "audio", "video"]


@dataclass(frozen=True, kw_only=True)
class DecisionMedia:
    """One image, audio or video value from the prompt. The state refers to it by its label,
    such as `<image 1>`, in prompt order per kind."""

    kind: MediaKind
    data: bytes
    media_type: str | None = None


@dataclass(frozen=True, kw_only=True)
class BooleanQuestion:
    instructions: str = ""
    criteria: Mapping[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True, kw_only=True)
class ChoiceQuestion:
    instructions: str = ""
    criteria: Mapping[str, Any]


@dataclass(frozen=True, kw_only=True)
class ScoreQuestion:
    instructions: str = ""
    criteria: list[Any] | tuple[Any, ...]


Question: TypeAlias = BooleanQuestion | ChoiceQuestion | ScoreQuestion


@dataclass(frozen=True, kw_only=True)
class BooleanAnswer:
    probability: float


@dataclass(frozen=True, kw_only=True)
class ChoiceAnswer:
    choice: str
    confidence: float
    probabilities: Mapping[str, float]


@dataclass(frozen=True, kw_only=True)
class ScoreAnswer:
    score: float
    confidence: float
    probabilities: Mapping[int, float]
    legend: Mapping[int, Any]


Answer: TypeAlias = BooleanAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True, kw_only=True)
class DecisionUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, kw_only=True)
class DecisionResponse:
    model: str
    answers: Mapping[str, Answer]
    usage: DecisionUsage = field(default_factory=DecisionUsage)


class DecisionClient(Protocol):
    def decide(
        self, state: JSONValue, questions: Mapping[str, Question], *, model: str | None = None
    ) -> DecisionResponse: ...


class AsyncDecisionClient(Protocol):
    async def decide(
        self, state: JSONValue, questions: Mapping[str, Question], *, model: str | None = None
    ) -> DecisionResponse: ...
