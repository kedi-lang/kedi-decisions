"""Compatibility exports with TypeSafe-native question conversion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

from kedi_decisions import schema as shared
from typesafe_sdk import Choice, Noul, Score

from .extraction import CandidateExtractor
from .transport import question_to_sdk

MAX_CHOICE_OPTIONS = shared.MAX_CHOICE_OPTIONS
QuestionKind = shared.QuestionKind
TypeSafeQuestion = Noul | Choice | Score


class QuestionSpec(shared.QuestionSpec):
    def to_native(self) -> TypeSafeQuestion:
        return question_to_sdk(self.as_question())


class EvaluationPlan(shared.EvaluationPlan):
    def native_questions(self) -> dict[str, TypeSafeQuestion]:
        return {key: question_to_sdk(value) for key, value in self.decision_questions().items()}


def build_evaluation_plan(
    schema: Mapping[str, Any], *, text_extractors: Mapping[str, CandidateExtractor] | None = None
) -> EvaluationPlan:
    plan = shared.build_evaluation_plan(schema, text_extractors=text_extractors)
    return EvaluationPlan(
        questions=tuple(
            QuestionSpec(
                **{field.name: getattr(question, field.name) for field in fields(question)}
            )
            for question in plan.questions
        ),
        schema=plan.schema,
    )
