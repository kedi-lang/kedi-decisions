from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, TypeVar, cast

from ._json_types import is_list, is_mapping, is_sequence
from .contracts import BooleanQuestion, ChoiceQuestion, JSONValue, ScoreQuestion
from .errors import DecisionExtractionError, DecisionSchemaError
from .extraction import (
    BUILTIN_FORMAT_EXTRACTORS,
    CandidateExtractor,
    RegexExtractor,
    resolve_candidates,
)
from .metadata import SCHEMA_KEY

QuestionKind = Literal["noul", "choice", "extract", "score"]
DecisionQuestion = BooleanQuestion | ChoiceQuestion | ScoreQuestion
MAX_CHOICE_OPTIONS = 255
PlanT = TypeVar("PlanT", bound="EvaluationPlan")


@dataclass(frozen=True, slots=True, kw_only=True)
class QuestionSpec:
    """One validated mapping from an output property to a Decision question."""

    key: str
    kind: QuestionKind
    instructions: str
    options: tuple[str, ...] = ()
    extractor: CandidateExtractor | None = None
    rejected_options: tuple[str, ...] = ()
    path: tuple[str, ...] = ()
    probability: bool = False
    nullable: bool = False
    local_none: bool = False
    label: str | None = None
    criteria: Any = None
    integer_score: bool = False

    def resolve(self, state: JSONValue) -> QuestionSpec:
        if self.kind != "extract":
            return self
        extractor = cast(CandidateExtractor, self.extractor)
        candidates = resolve_candidates(
            extractor,
            state,
            field_name=self.key,
            limit=MAX_CHOICE_OPTIONS - 1,
            allow_empty=self.nullable,
        )
        no_match = _no_match_option(candidates)
        return replace(
            self,
            options=candidates,
            rejected_options=(no_match,),
            local_none=not candidates,
        )

    @property
    def native_options(self) -> tuple[str, ...]:
        return self.options + self.rejected_options

    def as_question(self) -> DecisionQuestion:
        if self.kind == "noul":
            return BooleanQuestion(
                instructions=self.instructions,
                criteria=self.criteria
                if self.criteria is not None
                else {
                    "true": "The proposition is established by the provided state.",
                    "false": "The proposition is false or is not established by the provided state.",
                },
            )
        if self.kind == "score":
            return ScoreQuestion(instructions=self.instructions, criteria=self.criteria)
        if self.kind == "extract" and not self.rejected_options:
            raise DecisionExtractionError(
                f"Output property {self.key!r} must resolve text candidates before evaluation"
            )
        criteria = dict(self.criteria) if self.criteria is not None else dict.fromkeys(self.options)
        criteria.update(
            dict.fromkeys(
                self.rejected_options, "None of the candidate values satisfies the requested field."
            )
        )
        return ChoiceQuestion(instructions=self.instructions, criteria=criteria)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationPlan:
    """A complete, provider-ready interpretation of one Pydantic output schema."""

    questions: tuple[QuestionSpec, ...]
    schema: Mapping[str, Any] | None = None

    def resolve(self: PlanT, state: JSONValue) -> PlanT:
        return type(self)(
            questions=tuple(question.resolve(state) for question in self.questions),
            schema=self.schema,
        )

    def decision_questions(self) -> dict[str, DecisionQuestion]:
        return {
            question.key: question.as_question()
            for question in self.questions
            if not question.local_none
        }


def build_evaluation_plan(
    schema: Mapping[str, Any],
    *,
    text_extractors: Mapping[str, CandidateExtractor] | None = None,
) -> EvaluationPlan:
    """Convert Pydantic AI's native output object schema into Decision questions."""

    if schema.get("type") != "object":
        raise DecisionSchemaError("Decision output schema must be an object")

    raw_properties: object = schema.get("properties")
    if not isinstance(raw_properties, Mapping) or not raw_properties:
        raise DecisionSchemaError("Decision output schema must define at least one property")
    properties = cast(Mapping[object, object], raw_properties)
    if not all(isinstance(key, str) for key in properties):
        raise DecisionSchemaError("Decision output property names must be strings")
    typed_properties: dict[str, Any] = {cast(str, key): value for key, value in properties.items()}

    raw_required: object = schema.get("required")
    if not isinstance(raw_required, list):
        raise DecisionSchemaError("Decision output schema must declare required properties")
    required = cast(list[object], raw_required)
    if not all(isinstance(key, str) for key in required):
        raise DecisionSchemaError("Decision output schema must declare required properties")
    typed_required = [cast(str, key) for key in required]
    required_names = set(typed_required)
    property_names = set(typed_properties)
    extractors = dict(text_extractors or {})
    unknown_roots = sorted(
        name
        for name in extractors
        if name.split(".", 1)[0].replace("~1", ".").replace("~0", "~") not in property_names
    )
    if unknown_roots:
        raise DecisionSchemaError(
            f"Text extractors reference unknown output properties: {', '.join(unknown_roots)}"
        )
    invalid_extractors = sorted(
        name for name, extractor in extractors.items() if not callable(extractor)
    )
    if invalid_extractors:
        raise DecisionSchemaError(
            f"Text extractors must be callable for: {', '.join(invalid_extractors)}"
        )
    if required_names != property_names:
        missing = sorted(property_names - required_names)
        unknown = sorted(required_names - property_names)
        details: list[str] = []
        if missing:
            details.append(f"optional properties are unsupported: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown required properties: {', '.join(unknown)}")
        raise DecisionSchemaError("; ".join(details))

    definitions = schema.get("$defs", {})
    if not isinstance(definitions, Mapping):
        raise DecisionSchemaError("Decision output schema `$defs` must be an object")
    typed_definitions = cast(Mapping[str, Any], definitions)

    questions: list[QuestionSpec] = []

    def visit(raw: Any, path: tuple[str, ...], references: frozenset[str]) -> None:
        key = ".".join(segment.replace("~", "~0").replace(".", "~1") for segment in path)
        reference = raw.get("$ref") if is_mapping(raw) else None
        if reference in references:
            raise DecisionSchemaError(
                f"Output property {key!r} contains a recursive JSON reference"
            )
        prop = _resolve_schema(raw, typed_definitions, path=key)
        next_references = references | {reference} if reference else references
        if prop.get("type") in {"object", "array"} and SCHEMA_KEY in prop:
            raise DecisionSchemaError(
                "Decision criteria belong on scalar fields or multi-label item types"
            )
        if prop.get("type") == "object":
            children = prop.get("properties", {})
            if not children or set(prop.get("required", [])) != set(children):
                raise DecisionSchemaError(f"Nested object {key!r} must declare required properties")
            for child, child_schema in children.items():
                visit(child_schema, (*path, child), next_references)
            return
        if prop.get("type") == "array":
            item = _resolve_schema(prop.get("items"), typed_definitions, path=key)
            spec = _question_for_property(key, item, extractor=None)
            if spec.kind != "choice" or spec.nullable:
                raise DecisionSchemaError(
                    "Multi-label outputs require a finite string Literal/Enum"
                )
            item_description = item.get("description")
            instructions = (
                prop.get("description")
                or (item_description.strip() if isinstance(item_description, str) else None)
                or f"Which labels apply to {' '.join(path)}?"
            )
            for index, option in enumerate(spec.options):
                meaning = spec.criteria.get(option) if spec.criteria else None
                questions.append(
                    QuestionSpec(
                        key=f"{key}.label{index}",
                        kind="noul",
                        path=path,
                        label=option,
                        instructions=f"{instructions}\nDoes the label {option!r} apply?",
                        criteria={
                            "true": meaning or f"The label {option!r} applies.",
                            "false": f"The label {option!r} does not apply or is not established.",
                        },
                    )
                )
            return
        questions.append(
            replace(_question_for_property(key, prop, extractor=extractors.get(key)), path=path)
        )

    for name, property_schema in typed_properties.items():
        visit(property_schema, (name,), frozenset())
    unknown_extractors = sorted(set(extractors) - {question.key for question in questions})
    if unknown_extractors:
        raise DecisionSchemaError(
            f"Text extractors reference unknown output properties: {', '.join(unknown_extractors)}"
        )
    return EvaluationPlan(questions=tuple(questions), schema=schema)


def _question_for_property(
    key: str,
    schema: Mapping[str, Any],
    *,
    extractor: CandidateExtractor | None,
) -> QuestionSpec:
    variants = schema.get("anyOf")
    nullable = is_list(variants) and len(variants) == 2 and {"type": "null"} in variants
    if nullable:
        assert is_list(variants)
        inner = next(item for item in variants if item != {"type": "null"})
        schema = {**inner, **{name: value for name, value in schema.items() if name != "anyOf"}}
        variants = schema.get("anyOf")
    extension = schema.get(SCHEMA_KEY, {})
    if not is_mapping(extension):
        raise DecisionSchemaError(f"Invalid Decision metadata for {key!r}")
    criteria = extension.get("criteria")
    kind = extension.get("kind")
    description = schema.get("description")
    if description is not None and not isinstance(description, str):
        raise DecisionSchemaError(f"Output property {key!r} has a non-string description")
    description = description.strip() if description else ""

    if (
        is_list(variants)
        and variants
        and all(is_mapping(item) and type(item.get("const")) is int for item in variants)
    ):
        levels = {item["const"]: item.get("description") for item in variants}
        if (
            nullable
            or kind is not None
            or set(levels) != set(range(len(variants)))
            or any(not value for value in levels.values())
        ):
            raise DecisionSchemaError(
                "Integer rubrics require consecutive described levels starting at zero"
            )
        schema = {**schema, "type": "integer"}
        kind, criteria = "score", [levels[index] for index in range(len(levels))]

    if kind == "score":
        if schema.get("type") not in {"number", "integer"} or nullable:
            raise DecisionSchemaError("Rubric metadata requires a non-null numeric output")
        if not is_sequence(criteria) or not 2 <= len(criteria) <= 10:
            raise DecisionSchemaError("A rubric requires 2-10 described levels")
        if any(level is None or level == "" for level in criteria):
            raise DecisionSchemaError("Every rubric level needs a description")
        minimum, maximum = schema.get("minimum", 0), schema.get("maximum", len(criteria) - 1)
        if not isinstance(minimum, (int, float)) or not isinstance(maximum, (int, float)):
            raise DecisionSchemaError("Rubric bounds must be numeric")
        if minimum > 0 or maximum < len(criteria) - 1:
            raise DecisionSchemaError("Numeric bounds must contain every rubric level")
        return QuestionSpec(
            key=key,
            kind="score",
            instructions=description or f"Evaluate {_humanize(key)} against the rubric.",
            criteria=list(criteria),
            integer_score=schema.get("type") == "integer",
        )
    if kind not in {None, "choice", "noul"}:
        raise DecisionSchemaError(f"Unknown Decision metadata kind {kind!r}")
    is_probability = (
        schema.get("type") == "number" and schema.get("minimum") == 0 and schema.get("maximum") == 1
    )

    if schema.get("type") == "boolean" or is_probability:
        if nullable or kind == "choice":
            raise DecisionSchemaError("Noul fields cannot be nullable or use Choice criteria")
        if extractor is not None:
            raise DecisionSchemaError(
                f"Text extractor for output property {key!r} requires a string field"
            )
        instructions = description or _fallback_bool_instructions(key)
        if criteria is not None and (
            not is_mapping(criteria) or set(criteria) != {"true", "false"}
        ):
            raise DecisionSchemaError("Boolean criteria require true and false descriptions")
        return QuestionSpec(
            key=key,
            kind="noul",
            instructions=instructions,
            probability=is_probability,
            criteria=criteria,
        )

    raw_enum: object = schema.get("enum")
    if (
        is_list(variants)
        and variants
        and all(is_mapping(item) and isinstance(item.get("const"), str) for item in variants)
    ):
        raw_enum = [item["const"] for item in variants]
        schema = {**schema, "type": "string"}
        criteria = (
            {item["const"]: item.get("description") for item in variants}
            if criteria is None
            else criteria
        )
    if schema.get("type") == "string" and isinstance(raw_enum, list):
        if extractor is not None:
            raise DecisionSchemaError(
                f"Output property {key!r} cannot combine enum values with a text extractor"
            )
        enum = cast(list[object], raw_enum)
        if not all(isinstance(option, str) and option for option in enum):
            raise DecisionSchemaError(
                f"Output property {key!r} must use non-empty string choice values"
            )
        options = tuple(cast(str, option) for option in enum)
        if len(options) < 2:
            raise DecisionSchemaError(
                f"Output property {key!r} must define at least two choice values"
            )
        if len(options) + int(nullable) > 255:
            raise DecisionSchemaError(
                f"Output property {key!r} exceeds Decision's 255-option Choice limit"
            )
        if len(set(options)) != len(options):
            raise DecisionSchemaError(f"Output property {key!r} has duplicate choice values")
        if kind == "noul" or (
            criteria is not None and (not is_mapping(criteria) or set(criteria) != set(options))
        ):
            raise DecisionSchemaError("Choice criteria must match the declared options exactly")
        instructions = description or _fallback_choice_instructions(key)
        return QuestionSpec(
            key=key,
            kind="choice",
            instructions=instructions,
            options=options,
            nullable=nullable,
            rejected_options=(_no_match_option(options),) if nullable else (),
            criteria=criteria,
        )

    if schema.get("type") == "string":
        if kind is not None:
            raise DecisionSchemaError("Extraction fields cannot use choice or boolean criteria")
        resolved_extractor = extractor if extractor is not None else _schema_extractor(key, schema)
        if resolved_extractor is not None:
            instructions = description or _fallback_extraction_instructions(key)
            return QuestionSpec(
                key=key,
                kind="extract",
                instructions=instructions,
                extractor=resolved_extractor,
                nullable=nullable,
            )

    raise DecisionSchemaError(
        f"Output property {key!r} is unsupported; expected bool, a finite string "
        "Literal/Enum, or a string with a candidate extractor"
    )


def _schema_extractor(
    key: str,
    schema: Mapping[str, Any],
) -> CandidateExtractor | None:
    pattern = schema.get("pattern")
    if pattern is not None:
        if not isinstance(pattern, str) or not pattern:
            raise DecisionSchemaError(
                f"Output property {key!r} must use a non-empty string extraction pattern"
            )
        try:
            return RegexExtractor(pattern=pattern)
        except re.error as exc:
            raise DecisionSchemaError(
                f"Output property {key!r} has an invalid extraction pattern: {exc}"
            ) from exc

    format_name = schema.get("format")
    if format_name is None:
        return None
    if not isinstance(format_name, str):
        raise DecisionSchemaError(f"Output property {key!r} has a non-string format")
    return BUILTIN_FORMAT_EXTRACTORS.get(format_name)


def _resolve_schema(
    value: Any,
    definitions: Mapping[str, Any],
    *,
    path: str,
    references: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DecisionSchemaError(f"Output property {path!r} must contain a JSON schema object")
    typed_value = cast(Mapping[str, Any], value)
    reference = typed_value.get("$ref")
    if reference is None:
        if isinstance(typed_value.get("anyOf"), list):
            return {
                **typed_value,
                "anyOf": [
                    _resolve_schema(item, definitions, path=path, references=references)
                    for item in typed_value["anyOf"]
                ],
            }
        return typed_value
    if reference in references:
        raise DecisionSchemaError(f"Output property {path!r} contains a recursive JSON reference")
    if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
        raise DecisionSchemaError(f"Output property {path!r} has an unsupported JSON reference")

    definition_name = reference.removeprefix("#/$defs/").replace("~1", "/").replace("~0", "~")
    target = definitions.get(definition_name)
    if not isinstance(target, Mapping):
        raise DecisionSchemaError(
            f"Output property {path!r} references missing definition {definition_name!r}"
        )
    typed_target = cast(Mapping[str, Any], target)
    resolved = dict(
        _resolve_schema(typed_target, definitions, path=path, references=references | {reference})
    )
    resolved.update({key: item for key, item in typed_value.items() if key != "$ref"})
    return resolved


def _fallback_bool_instructions(key: str) -> str:
    if key == "response":
        return "Is the proposition or yes/no request in the provided state true?"
    return f"Is {_humanize(key)!r} true based on the provided state?"


def _fallback_choice_instructions(key: str) -> str:
    if key == "response":
        return "Which allowed value best answers the request in the provided state?"
    return f"Which allowed value best describes {_humanize(key)!r} in the provided state?"


def _fallback_extraction_instructions(key: str) -> str:
    if key == "response":
        return "Which candidate value should be extracted from the provided state?"
    return f"Which candidate value should be extracted for {_humanize(key)!r}?"


def _no_match_option(candidates: tuple[str, ...]) -> str:
    option = "__typesafe_no_matching_candidate__"
    while option in candidates:
        option = f"_{option}"
    return option


def _humanize(value: str) -> str:
    return value.replace("_", " ").strip()


__all__ = ["EvaluationPlan", "QuestionSpec", "build_evaluation_plan"]
