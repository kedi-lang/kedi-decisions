"""Validate Laya's native response before it crosses the shared decision boundary."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StrictFloat, TypeAdapter, ValidationError

from kedi_decisions.contracts import (
    Answer,
    BooleanAnswer,
    ChoiceAnswer,
    DecisionResponse,
    DecisionUsage,
    ScoreAnswer,
)
from kedi_decisions.errors import DecisionResponseError


class BooleanResult(BaseModel):
    type: Literal["noul"]
    noul: Annotated[StrictFloat, Field(ge=0, le=1, allow_inf_nan=False)]


class ChoiceResult(BaseModel):
    type: Literal["choice"]
    choice: Annotated[str, Field(strict=True)]
    confidence: StrictFloat
    probabilities: dict[str, StrictFloat]


class ScoreResult(BaseModel):
    type: Literal["score"]
    score: StrictFloat
    confidence: StrictFloat
    probabilities: dict[int, StrictFloat]
    legend: dict[int, Any]


class Usage(BaseModel):
    input_tokens: Annotated[int, Field(strict=True, ge=0)] | None = None
    output_tokens: Annotated[int, Field(strict=True, ge=0)] | None = None


class Response(BaseModel):
    answers: dict[
        str, Annotated[BooleanResult | ChoiceResult | ScoreResult, Field(discriminator="type")]
    ]
    usage: Usage = Field(default_factory=Usage)


RESPONSE = TypeAdapter(Response)


def parse_response(response: Any, *, model: str, expected: set[str]) -> DecisionResponse:
    try:
        parsed = RESPONSE.validate_python(response)
    except ValidationError as exc:
        raise DecisionResponseError(f"Invalid Laya response: {exc}") from exc
    if set(parsed.answers) != expected:
        raise DecisionResponseError("Laya answer keys do not match the request")
    answers: dict[str, Answer] = {}
    for key, answer in parsed.answers.items():
        if isinstance(answer, BooleanResult):
            answers[key] = BooleanAnswer(probability=answer.noul)
        elif isinstance(answer, ChoiceResult):
            answers[key] = ChoiceAnswer(
                choice=answer.choice,
                confidence=answer.confidence,
                probabilities=answer.probabilities,
            )
        else:
            answers[key] = ScoreAnswer(
                score=answer.score,
                confidence=answer.confidence,
                probabilities=answer.probabilities,
                legend=answer.legend,
            )
    return DecisionResponse(
        model=model,
        answers=answers,
        usage=DecisionUsage(
            input_tokens=parsed.usage.input_tokens,
            output_tokens=parsed.usage.output_tokens,
        ),
    )
