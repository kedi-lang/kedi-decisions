from __future__ import annotations

# This pinned integration intentionally shares upstream projection and evaluator internals.
# pyright: reportPrivateUsage=false
import json
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from functools import cached_property
from types import TracebackType
from typing import Any, ClassVar, Literal

from kedi_decisions.routing import ToolCallProposed, prepare_routing, resolve_routing
from pydantic_ai import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, check_allow_model_requests
from pydantic_ai.models import decision as decision_model
from pydantic_ai.models import typesafe as upstream
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage
from typing_extensions import Self

from ..core import CandidateExtractor, TypeSafeEvaluator
from ..core.evaluation import (
    DEFAULT_THRESHOLD,
    AsyncSystemOneClient,
    JSONValue,
    validate_threshold,
)
from ._pydantic_errors import provider_errors
from ._pydantic_provider import EvaluatorProvider
from ._pydantic_stream import ExtendedTypeSafeStream

_PROFILE = ModelProfile(
    supports_tools=True,
    supports_json_schema_output=True,
    supports_json_object_output=False,
    default_structured_output_mode="tool",
    supports_inline_system_prompts=True,
)


class TypeSafeModelSettings(upstream.TypeSafeModelSettings, total=False):
    typesafe_threshold: float
    """Strict probability threshold for bools and label membership; default 0.85."""


def messages_to_state(messages: list[ModelMessage]) -> JSONValue:
    """Use the pinned upstream projection, including tool and retry history."""
    state = decision_model._map_messages(messages, turn=False)
    if not isinstance(state, (str, list, dict)):
        raise TypeError("Decision history must map to text or structured state")
    return state


class TypeSafeModel(upstream.TypeSafeModel):
    """Upstream Jev model with explicit decision thresholds and extended schemas."""

    supports_decision_threshold = True
    kedi_prompt_mode: ClassVar[Literal["decision"]] = "decision"

    @cached_property
    def profile(self) -> ModelProfile:
        # NativeOutput is serialized locally; free text is rejected in request().
        return _PROFILE

    def __init__(
        self,
        model_name: str = "jev-latest",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        timeout: float | None = None,
        client: AsyncSystemOneClient | None = None,
        settings: ModelSettings | None = None,
        text_extractors: Mapping[str, CandidateExtractor] | None = None,
    ) -> None:
        self._evaluator = TypeSafeEvaluator(
            model_name,
            api_key=api_key,
            base_url=base_url,
            threshold=threshold,
            timeout=timeout,
            client=client,
            text_extractors=text_extractors,
        )
        super().__init__(
            model_name,
            provider=EvaluatorProvider(self._evaluator),
            profile=_PROFILE,
            settings=settings,
        )

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        if model_request_parameters.native_tools:
            raise UserError("Native tools are unsupported by TypeSafeModel")
        if model_request_parameters.allow_image_output:
            raise UserError("Image output is not supported by TypeSafeModel")
        # Keep Kedi's strict thresholds instead of upstream's changed routing defaults.
        settings, parameters = Model.prepare_request(self, model_settings, model_request_parameters)
        settings = settings or {}
        unsupported = set(settings) - {"typesafe_threshold", "typesafe_tool_call_threshold"}
        if unsupported:
            raise UserError(
                f"TypeSafeModel does not support model settings: {', '.join(sorted(unsupported))}"
            )
        threshold = validate_threshold(
            settings.get("typesafe_threshold", self._evaluator.threshold)
        )
        tool_threshold = validate_threshold(settings.get("typesafe_tool_call_threshold", 0.6))
        native = parameters.output_mode == "native" and parameters.output_object is not None
        if parameters.allow_text_output and not native:
            raise UserError("Text output is not supported by TypeSafeModel")
        if native:
            output = parameters.output_object
            assert output is not None
            output_tool = ToolDefinition(
                name=output.name or "final_result",
                description=output.description,
                parameters_json_schema=output.json_schema,
            )
            hand_offs: list[ToolDefinition] = []
        else:
            outputs, hand_offs = decision_model._output_tools(parameters)
            if len(outputs) > 1:
                raise UserError("TypeSafeModel supports one structured output schema")
            output_tool = outputs[0] if outputs else None
        tools, _ = decision_model._tools_left(
            messages,
            [
                *hand_offs,
                *[
                    tool
                    for tool in parameters.function_tools
                    if parameters.visibility_of(tool.name) != "withheld"
                ],
            ],
        )
        state = messages_to_state(messages)
        instructions = "\n\n".join(
            part.content for part in (self._get_instruction_parts(messages, parameters) or [])
        )
        if instructions:
            state = {"state": state, "instructions": instructions}
        if tools and output_tool and not (decision_model._purpose(output_tool) or instructions):
            raise UserError("Give the output type a docstring or provide agent instructions")
        if output_tool is None and not tools:
            raise UserError("TypeSafeModel requires an output schema or a hand-off tool")
        route = prepare_routing(
            output_tool.parameters_json_schema if output_tool else {"properties": {}},
            [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_json_schema,
                }
                for tool in tools
            ],
            set(),
            allow_final=output_tool is not None,
        )
        with provider_errors(self.model_name):
            result = await self._evaluator.evaluate(
                state=state, schema=route.schema, threshold=threshold
            )
        try:
            result, call = resolve_routing(result, route, tool_threshold)
        except ToolCallProposed as exc:
            raise decision_model.UnfillableRoute(
                self.model_name, exc.tool_name, exc.probability
            ) from exc
        details: dict[str, Any] = {"typesafe": result.metadata}
        parts: list[ModelResponsePart]
        if call is not None:
            parts = [ToolCallPart(call["name"], {})]
        else:
            if output_tool is None:
                raise UserError("TypeSafeModel did not select a hand-off above the threshold")
            parts = (
                [TextPart(json.dumps(result.values, ensure_ascii=False))]
                if native
                else [ToolCallPart(output_tool.name, result.values)]
            )
        return ModelResponse(
            parts=parts,
            usage=RequestUsage(
                input_tokens=result.input_tokens or 0,
                output_tokens=result.output_tokens or 0,
            ),
            model_name=result.model,
            provider_name=self.system,
            provider_url=self.base_url,
            provider_details=details,
            metadata=details,
            finish_reason="stop" if native and call is None else "tool_call",
        )

    async def aclose(self) -> None:
        await self._evaluator.aclose()

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[ExtendedTypeSafeStream]:
        response = await self.request(messages, model_settings, model_request_parameters)
        yield ExtendedTypeSafeStream(model_request_parameters, response)

    async def aclose_current(self) -> None:
        await self._evaluator.aclose_current()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


__all__ = ["TypeSafeModel", "TypeSafeModelSettings", "messages_to_state"]
