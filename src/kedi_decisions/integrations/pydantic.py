"""Native Pydantic AI model for provider-neutral decision clients."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, ClassVar, Literal

from pydantic_ai import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    check_allow_model_requests,
)
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from ..evaluation import DecisionEvaluator, JSONValue, validate_threshold
from ..routing import prepare_routing, resolve_routing
from ._pydantic_stream import DecisionStream


def messages_to_state(messages: list[ModelMessage]) -> JSONValue:
    history: list[dict[str, Any]] = []
    text: list[str] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                if not isinstance(part.content, str):
                    raise UserError("Decision models require text-only user input")
                if message is messages[-1]:
                    text.append(part.content)
                else:
                    history.append({"role": "user", "content": part.content})
            elif isinstance(part, (SystemPromptPart, TextPart)):
                history.append(
                    {
                        "role": "system" if isinstance(part, SystemPromptPart) else "assistant",
                        "content": part.content,
                    }
                )
            elif isinstance(part, ToolCallPart):
                history.append(
                    {
                        "role": "assistant",
                        "tool": part.tool_name,
                        "arguments": part.args_as_dict(),
                        "tool_call_id": part.tool_call_id,
                    }
                )
            elif isinstance(part, ToolReturnPart):
                history.append(
                    {
                        "role": "tool",
                        "name": part.tool_name,
                        "content": part.model_response_str(),
                        "tool_call_id": part.tool_call_id,
                    }
                )
            elif isinstance(part, RetryPromptPart):
                history.append({"role": "retry", "content": part.model_response()})
            else:
                raise UserError(f"Unsupported decision input part: {part.part_kind}")
    joined = "\n\n".join(text)
    if not joined and not history:
        raise UserError("Decision models require input to evaluate")
    return {"history": history, "text": joined} if history else joined


class DecisionModel(Model):
    kedi_prompt_mode: ClassVar[Literal["decision"]] = "decision"
    supports_decision_threshold = True

    def __init__(self, evaluator: DecisionEvaluator, *, settings: ModelSettings | None = None):
        super().__init__(
            settings=settings,
            profile=ModelProfile(
                supports_tools=True,
                supports_json_schema_output=True,
                supports_json_object_output=False,
                default_structured_output_mode="tool",
            ),
        )
        self._evaluator = evaluator

    @property
    def model_name(self) -> str:
        return self._evaluator.model_name

    @property
    def system(self) -> str:
        return self._evaluator.provider_name

    @property
    def base_url(self) -> str:
        return self._evaluator.base_url

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        settings, parameters = self.prepare_request(model_settings, model_request_parameters)
        settings = settings or {}
        unsupported = set(settings) - {"decision_threshold", "decision_tool_call_threshold"}
        if unsupported:
            raise UserError(
                f"DecisionModel does not support model settings: {', '.join(sorted(unsupported))}"
            )
        if parameters.native_tools:
            raise UserError("Native tools are unsupported by decision models")
        threshold = validate_threshold(
            settings.get("decision_threshold", self._evaluator.threshold)
        )
        tool_threshold = validate_threshold(settings.get("decision_tool_call_threshold", 0.6))
        native = parameters.output_mode == "native" and parameters.output_object is not None
        output = None
        handoffs: list[ToolDefinition] = []
        if native and parameters.output_object is not None:
            obj = parameters.output_object
            output = ToolDefinition(
                name=obj.name or "final_result", parameters_json_schema=obj.json_schema
            )
        else:
            if parameters.allow_text_output:
                raise UserError("Decision models require structured output, not free-form text")
            for tool in parameters.output_tools:
                if tool.parameters_json_schema.get("properties"):
                    if output is not None:
                        raise UserError("Decision models support one structured output schema")
                    output = tool
                else:
                    handoffs.append(tool)
        tools = [
            *handoffs,
            *[
                tool
                for tool in parameters.function_tools
                if parameters.visibility_of(tool.name) != "withheld"
            ],
        ]
        if output is None and not tools:
            raise UserError("Decision models require an output schema or a hand-off tool")
        completed: set[str] = set()
        for message in reversed(messages):
            if isinstance(message, ModelRequest) and any(
                isinstance(part, UserPromptPart) for part in message.parts
            ):
                break
            completed.update(
                part.tool_name for part in message.parts if isinstance(part, ToolReturnPart)
            )
        state = messages_to_state(messages)
        instructions = "\n\n".join(
            part.content for part in (self._get_instruction_parts(messages, parameters) or [])
        )
        if instructions:
            state = {"state": state, "instructions": instructions}
        schema: dict[str, Any] = (
            output.parameters_json_schema
            if output
            else {"type": "object", "properties": {}, "required": []}
        )
        route = prepare_routing(
            schema,
            [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_json_schema,
                }
                for tool in tools
            ],
            completed,
            allow_final=output is not None,
        )
        result = await self._evaluator.evaluate(
            state=state, schema=route.schema, threshold=threshold
        )
        result, call = resolve_routing(result, route, tool_threshold)
        if call is not None:
            parts = [ToolCallPart(call["name"], {})]
        elif output is not None:
            parts = (
                [TextPart(json.dumps(result.values, ensure_ascii=False))]
                if native
                else [ToolCallPart(output.name, result.values)]
            )
        else:
            raise UserError("The decision model did not select a hand-off above the threshold")
        details = {"decisions": result.metadata}
        return ModelResponse(
            parts=parts,
            model_name=result.model,
            provider_name=self.system,
            provider_url=self.base_url,
            usage=RequestUsage(
                input_tokens=result.input_tokens or 0, output_tokens=result.output_tokens or 0
            ),
            provider_details=details,
            metadata=details,
            finish_reason="stop" if native and call is None else "tool_call",
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        yield DecisionStream(
            model_request_parameters,
            await self.request(messages, model_settings, model_request_parameters),
        )

    async def aclose_current(self) -> None:
        await self._evaluator.aclose_current()

    async def aclose(self) -> None:
        await self._evaluator.aclose()
