from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from jsonschema import Draft202012Validator, ValidationError

from .contracts import (
    Answer,
    AsyncDecisionClient,
    BooleanAnswer,
    ChoiceAnswer,
    DecisionClient,
    DecisionResponse,
    DecisionUsage,
    JSONValue,
    ScoreAnswer,
)
from .errors import DecisionResponseError
from .extraction import CandidateExtractor
from .schema import EvaluationPlan, QuestionSpec, build_evaluation_plan

DEFAULT_THRESHOLD = 0.85


def _request_metadata(
    state: JSONValue,
    plan: EvaluationPlan,
    *,
    model: str,
    threshold: float,
) -> dict[str, str]:
    questions = [
        {
            "key": question.key,
            "kind": question.kind,
            "instructions": question.instructions,
            "options": list(question.native_options),
            "path": list(question.path),
            "probability": question.probability,
            "nullable": question.nullable,
            "local_none": question.local_none,
            "label": question.label,
            "criteria": question.criteria,
            "integer_score": question.integer_score,
        }
        for question in plan.questions
    ]
    state_fingerprint = _json_fingerprint(state)
    questions_fingerprint = _json_fingerprint(questions)
    config_fingerprint = _json_fingerprint(
        {
            "model": model,
            "boolean_threshold": threshold,
            "boolean_comparator": ">",
        }
    )
    return {
        "state_fingerprint": state_fingerprint,
        "questions_fingerprint": questions_fingerprint,
        "config_fingerprint": config_fingerprint,
        "request_fingerprint": _json_fingerprint(
            {
                "state_fingerprint": state_fingerprint,
                "questions_fingerprint": questions_fingerprint,
                "config_fingerprint": config_fingerprint,
            }
        ),
    }


def _json_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"kedi-decisions-request-v1:sha256:{hashlib.sha256(payload).hexdigest()}"


def validate_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("Decision boolean threshold must be a finite number")
    if not 0 <= value <= 1:
        raise ValueError("Decision boolean threshold must be between 0 and 1")
    return float(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationResult:
    values: dict[str, Any]
    model: str
    input_tokens: int | None
    output_tokens: int | None
    metadata: dict[str, Any]


class Evaluator(Protocol):
    """Shared evaluation surface; provider transports retain their own lifecycles."""

    model_name: str
    threshold: float

    @property
    def provider_name(self) -> str: ...

    @property
    def base_url(self) -> str: ...

    async def evaluate(
        self, *, state: Any, schema: Mapping[str, Any], threshold: float | None = None
    ) -> EvaluationResult: ...

    def evaluate_sync(
        self, *, state: Any, schema: Mapping[str, Any], threshold: float | None = None
    ) -> EvaluationResult: ...

    def close(self) -> None: ...
    async def aclose_current(self) -> None: ...
    async def aclose(self) -> None: ...


class ResultDecoder:
    """Validate normalized provider answers against the original output schema."""

    threshold: float
    _probability_decimals: int | None

    @property
    def provider_name(self) -> str:
        raise NotImplementedError

    def _decode_result(
        self,
        response: DecisionResponse,
        plan: EvaluationPlan,
        threshold: float | None = None,
        *,
        request_metadata: Mapping[str, str] | None = None,
    ) -> EvaluationResult:
        effective_threshold = self.threshold if threshold is None else threshold
        values, answer_metadata = self._decode(response, plan, effective_threshold)
        if plan.schema is not None:
            try:
                validator: Any = Draft202012Validator(plan.schema)
                validator.validate(values)
            except ValidationError as exc:
                raise DecisionResponseError(
                    f"Decision output failed schema validation at {list(exc.path)!r}: {exc.message}"
                ) from exc
        return EvaluationResult(
            values=values,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            metadata={
                "schema_version": 1,
                "answers": answer_metadata,
                "boolean_threshold": effective_threshold,
                "boolean_comparator": ">",
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                **dict(request_metadata or {}),
                "provider": self.provider_name,
            },
        )

    def _decode(
        self,
        response: DecisionResponse,
        plan: EvaluationPlan,
        threshold: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        expected_keys = {question.key for question in plan.questions if not question.local_none}
        actual_keys = set(response.answers)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(actual_keys - expected_keys)
            raise DecisionResponseError(
                f"Decision response answer keys do not match the request; "
                f"missing={missing}, unexpected={unexpected}"
            )

        values: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        for question in plan.questions:
            evidence: dict[str, Any]
            if question.local_none:
                value, evidence = None, {"type": "extraction", "source": "no_candidates"}
            elif question.kind == "noul":
                value, evidence = self._decode_noul(
                    question, response.answers[question.key], threshold
                )
            elif question.kind == "score":
                value, evidence = self._decode_score(
                    question, response.answers[question.key], self._probability_decimals
                )
            else:
                value, evidence = self._decode_choice(
                    question, response.answers[question.key], self._probability_decimals
                )
            path = question.path or (question.key,)
            target = values
            for segment in path[:-1]:
                target = target.setdefault(segment, {})
            if question.label is not None:
                labels = target.setdefault(path[-1], [])
                if value:
                    labels.append(question.label)
            else:
                target[path[-1]] = value
            answer_metadata: dict[str, Any] = {**evidence, "path": list(path)}
            if question.label is not None:
                answer_metadata["label"] = question.label
            metadata[question.key] = answer_metadata
        return values, metadata

    def _decode_noul(
        self,
        question: QuestionSpec,
        answer: Answer,
        threshold: float | None = None,
    ) -> tuple[bool | float, dict[str, Any]]:
        if not isinstance(answer, BooleanAnswer):
            raise DecisionResponseError(f"Decision answer {question.key!r} must be a Noul answer")
        probability = answer.probability
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise DecisionResponseError(
                f"Decision Noul answer {question.key!r} returned invalid probability {probability!r}"
            )
        effective_threshold = self.threshold if threshold is None else threshold
        return (probability if question.probability else probability > effective_threshold), {
            "type": "noul",
            "probability": probability,
            "output_kind": "probability" if question.probability else "boolean",
        }

    @staticmethod
    def _decode_choice(
        question: QuestionSpec,
        answer: Answer,
        probability_decimals: int | None = None,
    ) -> tuple[str | None, dict[str, Any]]:
        if not isinstance(answer, ChoiceAnswer):
            raise DecisionResponseError(f"Decision answer {question.key!r} must be a Choice answer")
        if answer.choice in question.rejected_options and not question.nullable:
            raise DecisionResponseError(
                f"Decision could not extract a matching candidate for {question.key!r}"
            )
        if answer.choice not in question.native_options:
            raise DecisionResponseError(
                f"Decision Choice answer {question.key!r} returned unsupported value "
                f"{answer.choice!r}"
            )
        expected = set(question.native_options)
        if set(answer.probabilities) != expected:
            raise DecisionResponseError(
                f"Decision Choice probabilities {question.key!r} do not match its options"
            )
        probabilities = dict(answer.probabilities)
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
            raise DecisionResponseError(
                f"Decision Choice answer {question.key!r} returned invalid probabilities"
            )
        tolerance = (
            len(probabilities) * 0.5 * 10**-probability_decimals + 1e-12
            if probability_decimals is not None
            else 1e-6
        )
        if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=1e-6, abs_tol=tolerance):
            raise DecisionResponseError(
                f"Decision Choice probabilities {question.key!r} must sum to 1"
            )
        if not math.isfinite(answer.confidence) or not 0 <= answer.confidence <= 1:
            raise DecisionResponseError(
                f"Decision Choice answer {question.key!r} returned invalid confidence"
            )
        return (None if answer.choice in question.rejected_options else answer.choice), {
            "type": "extraction" if question.kind == "extract" else "choice",
            "choice": answer.choice,
            "confidence": answer.confidence,
            "probabilities": probabilities,
        }

    @staticmethod
    def _decode_score(
        question: QuestionSpec, answer: Answer, probability_decimals: int | None = None
    ) -> tuple[float | int, dict[str, Any]]:
        if not isinstance(answer, ScoreAnswer):
            raise DecisionResponseError(f"Decision answer {question.key!r} must be a Score answer")
        expected = set(range(len(question.criteria)))
        probabilities = answer.probabilities
        tolerance = (
            len(probabilities) * 0.5 * 10**-probability_decimals + 1e-12
            if probability_decimals is not None
            else 1e-6
        )
        if set(probabilities) != expected or set(answer.legend) != expected:
            raise DecisionResponseError("Score probabilities and legend must match rubric levels")
        if any(
            not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()
        ) or not math.isclose(sum(probabilities.values()), 1, abs_tol=tolerance):
            raise DecisionResponseError("Invalid Score probabilities")
        if not math.isfinite(answer.score) or not 0 <= answer.score <= max(expected):
            raise DecisionResponseError("Invalid rubric score")
        if not math.isfinite(answer.confidence) or not 0 <= answer.confidence <= 1:
            raise DecisionResponseError("Invalid Score confidence")
        value = int(answer.score + 0.5) if question.integer_score else answer.score
        return value, {
            "type": "score",
            "score": answer.score,
            "confidence": answer.confidence,
            "probabilities": dict(probabilities),
            "legend": dict(answer.legend),
        }


class DecisionEvaluator(ResultDecoder):
    """Execute validated Decision question plans and normalize their typed answers."""

    def __init__(
        self,
        model_name: str,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        client: AsyncDecisionClient | None = None,
        sync_client: DecisionClient | None = None,
        provider_name: str = "decision",
        base_url: str = "",
        text_extractors: Mapping[str, CandidateExtractor] | None = None,
    ) -> None:
        model_name = model_name.strip()
        if not model_name:
            raise ValueError("Decision model name must not be empty")
        self.model_name = model_name
        self.threshold = validate_threshold(threshold)
        self.text_extractors = dict(text_extractors or {})
        self._borrowed_client = client
        self._sync_client = sync_client
        self._probability_decimals = getattr(client or sync_client, "probability_decimals", None)
        self._provider_name = provider_name
        self._base_url = base_url

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def base_url(self) -> str:
        return self._base_url

    async def evaluate(
        self,
        *,
        state: JSONValue,
        schema: Mapping[str, Any],
        threshold: float | None = None,
    ) -> EvaluationResult:
        plan = self._plan(state, schema)
        effective_threshold = self.threshold if threshold is None else validate_threshold(threshold)
        request_metadata = _request_metadata(
            state,
            plan,
            model=self.model_name,
            threshold=effective_threshold,
        )
        if not plan.decision_questions():
            return self._decode_result(
                DecisionResponse(
                    model=self.model_name,
                    answers={},
                    usage=DecisionUsage(input_tokens=0, output_tokens=0),
                ),
                plan,
                effective_threshold,
                request_metadata=request_metadata,
            )
        client = self.async_client()
        response = await client.decide(
            state,
            plan.decision_questions(),
            model=self.model_name,
        )
        return self._decode_result(
            response,
            plan,
            effective_threshold,
            request_metadata=request_metadata,
        )

    def async_client(self) -> AsyncDecisionClient:
        if self._borrowed_client is None:
            raise ValueError("An async decision client is required for evaluate()")
        return self._borrowed_client

    def evaluate_sync(
        self,
        *,
        state: JSONValue,
        schema: Mapping[str, Any],
        threshold: float | None = None,
    ) -> EvaluationResult:
        plan = self._plan(state, schema)
        effective_threshold = self.threshold if threshold is None else validate_threshold(threshold)
        request_metadata = _request_metadata(
            state,
            plan,
            model=self.model_name,
            threshold=effective_threshold,
        )
        if not plan.decision_questions():
            return self._decode_result(
                DecisionResponse(
                    model=self.model_name,
                    answers={},
                    usage=DecisionUsage(input_tokens=0, output_tokens=0),
                ),
                plan,
                effective_threshold,
                request_metadata=request_metadata,
            )
        client = self._sync_client
        if client is None:
            raise ValueError("A synchronous decision client is required for evaluate_sync()")
        response = client.decide(
            state,
            plan.decision_questions(),
            model=self.model_name,
        )
        return self._decode_result(
            response,
            plan,
            effective_threshold,
            request_metadata=request_metadata,
        )

    def _plan(
        self,
        state: JSONValue,
        schema: Mapping[str, Any],
    ) -> EvaluationPlan:
        return build_evaluation_plan(
            schema,
            text_extractors=self.text_extractors,
        ).resolve(state)

    def close(self) -> None:
        """Clients are borrowed; their owner controls their lifetime."""

    async def aclose_current(self) -> None:
        """The provider-neutral evaluator owns no event-loop resources."""

    async def aclose(self) -> None:
        self.close()


__all__ = [
    "EvaluationResult",
    "JSONValue",
    "DecisionEvaluator",
]
