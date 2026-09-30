from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from types import TracebackType
from typing import Any, ClassVar, Literal, NoReturn, cast

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr, TypeAdapter, field_validator
from typing_extensions import Self

from ..evaluation import (
    DEFAULT_THRESHOLD,
    Evaluator,
    JSONValue,
    validate_threshold,
)
from ._langchain_routing import ToolCallProposed, prepare_routing, resolve_routing

_PROFILE = {
    "name": "Typed decision model",
    "text_inputs": True,
    "text_outputs": True,
    "tool_calling": True,
    "tool_choice": True,
    "structured_output": True,
}


def messages_to_state(messages: list[BaseMessage]) -> JSONValue:
    """Project supported LangChain message history into structured Decision state."""

    rendered: list[JSONValue] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.invalid_tool_calls:
            _unsupported_message("invalid assistant tool-call history")
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, ToolMessage):
            role = "tool"
        elif isinstance(message, ChatMessage) and message.role in {
            "system",
            "user",
            "assistant",
        }:
            role = message.role
        else:
            _unsupported_message(message.type)
        rendered.append({"role": role, "content": _text_content(message)})

    if not rendered:
        raise ValueError("DecisionChatModel requires at least one supported text message")
    return {"messages": convert_to_openai_messages(messages)}


def _text_content(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    chunks: list[str] = []
    for item in message.content:
        if isinstance(item, str):
            chunks.append(item)
            continue
        content_block = cast(dict[str, Any], item)
        if content_block.get("type") == "text":
            text = content_block.get("text")
            if not isinstance(text, str):
                _unsupported_message("non-string text content")
            chunks.append(text)
        else:
            kind = content_block.get("type", "unknown")
            _unsupported_message(f"content block {kind!r}")
    return "\n".join(chunks)


def _unsupported_message(kind: str) -> NoReturn:
    raise ValueError(
        f"DecisionChatModel does not support LangChain {kind}; "
        "provide text-only state without tool or media history"
    )


class DecisionChatModel(BaseChatModel):
    """LangChain projection over a provider-neutral typed-decision evaluator."""

    kedi_prompt_mode: ClassVar[Literal["decision"]] = "decision"
    threshold_setting: ClassVar[str] = "decision_threshold"
    tool_threshold_setting: ClassVar[str] = "decision_tool_call_threshold"
    metadata_key: ClassVar[str] = "decisions"

    model_name: str
    threshold: float = DEFAULT_THRESHOLD
    decision_threshold: float | None = None
    decision_tool_call_threshold: float = 0.6
    _evaluator: Evaluator = PrivateAttr()

    @field_validator(
        "threshold", "decision_threshold", "decision_tool_call_threshold", mode="before"
    )
    @classmethod
    def _validate_threshold(cls, value: Any) -> float:
        return validate_threshold(value)

    def __init__(
        self,
        model_name: str,
        *,
        evaluator: Evaluator,
        threshold: float = DEFAULT_THRESHOLD,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("profile", dict(_PROFILE))
        model_values: dict[str, Any] = {
            "model_name": model_name,
            "threshold": threshold,
            **kwargs,
        }
        super().__init__(**model_values)
        self._evaluator = evaluator

    @property
    def _llm_type(self) -> str:
        return self._evaluator.provider_name

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "threshold": getattr(self, self.threshold_setting)
            if getattr(self, self.threshold_setting) is not None
            else self.threshold,
            self.tool_threshold_setting: getattr(self, self.tool_threshold_setting),
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        threshold = validate_threshold(
            kwargs.pop(
                self.threshold_setting,
                getattr(self, self.threshold_setting)
                if getattr(self, self.threshold_setting) is not None
                else self.threshold,
            )
        )
        tools = kwargs.pop("tools", [])
        tool_threshold = validate_threshold(
            kwargs.pop(self.tool_threshold_setting, getattr(self, self.tool_threshold_setting))
        )
        schema = self._request_schema(stop=stop, kwargs=kwargs)
        request = prepare_routing(schema, tools, messages)
        result = self._evaluator.evaluate_sync(
            state=messages_to_state(messages),
            schema=request.schema,
            threshold=threshold,
        )
        result, call = resolve_routing(result, request, tool_threshold)
        return _chat_result(result, call, metadata_key=self.metadata_key)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        threshold = validate_threshold(
            kwargs.pop(
                self.threshold_setting,
                getattr(self, self.threshold_setting)
                if getattr(self, self.threshold_setting) is not None
                else self.threshold,
            )
        )
        tools = kwargs.pop("tools", [])
        tool_threshold = validate_threshold(
            kwargs.pop(self.tool_threshold_setting, getattr(self, self.tool_threshold_setting))
        )
        schema = self._request_schema(stop=stop, kwargs=kwargs)
        request = prepare_routing(schema, tools, messages)
        result = await self._evaluator.evaluate(
            state=messages_to_state(messages),
            schema=request.schema,
            threshold=threshold,
        )
        result, call = resolve_routing(result, request, tool_threshold)
        return _chat_result(result, call, metadata_key=self.metadata_key)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        if tool_choice not in (None, "auto", "none"):
            raise ValueError("DecisionChatModel supports only auto or none tool choice")
        definitions = (
            [convert_to_openai_tool(tool)["function"] for tool in tools]
            if tool_choice != "none"
            else []
        )
        response_format = kwargs.pop("response_format", None)
        settings = {
            name: kwargs.pop(name)
            for name in (self.threshold_setting, self.tool_threshold_setting)
            if name in kwargs
        }
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise ValueError(f"DecisionChatModel does not support model settings: {names}")
        if response_format is None:
            raise ValueError(
                "DecisionChatModel requires native structured output; provide a response schema"
            )
        return self.bind(response_format=response_format, tools=definitions, **settings)

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise ValueError(f"DecisionChatModel does not support model settings: {names}")
        response_format, validator = _structured_contract(schema)
        model = self.bind(response_format=response_format)

        def parse(message: AIMessage) -> Any:
            try:
                parsed = validator.validate_json(_message_text(message))
            except Exception as exc:
                if include_raw:
                    return {"raw": message, "parsed": None, "parsing_error": exc}
                raise
            if include_raw:
                return {"raw": message, "parsed": parsed, "parsing_error": None}
            return parsed

        return model | RunnableLambda(parse)

    def close(self) -> None:
        self._evaluator.close()

    async def aclose(self) -> None:
        await self.aclose_current()
        self.close()

    async def aclose_current(self) -> None:
        await self._evaluator.aclose_current()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        self.close()
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        await self.aclose()
        return None

    @staticmethod
    def _request_schema(
        *,
        stop: list[str] | None,
        kwargs: dict[str, Any],
    ) -> Mapping[str, Any]:
        if stop:
            raise ValueError("DecisionChatModel does not support stop sequences")
        response_format = kwargs.pop("response_format", None)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise ValueError(f"DecisionChatModel does not support model settings: {names}")
        if not isinstance(response_format, Mapping):
            raise ValueError(
                "DecisionChatModel requires native structured output; provide a response schema"
            )
        typed_response_format = cast(Mapping[str, Any], response_format)
        if typed_response_format.get("type") != "json_schema":
            raise ValueError("DecisionChatModel requires a JSON Schema response format")
        envelope = typed_response_format.get("json_schema")
        if not isinstance(envelope, Mapping):
            raise ValueError("DecisionChatModel received an invalid JSON Schema response format")
        typed_envelope = cast(Mapping[str, Any], envelope)
        schema = typed_envelope.get("schema")
        if not isinstance(schema, Mapping):
            raise ValueError("DecisionChatModel received an invalid JSON Schema response schema")
        return cast(Mapping[str, Any], schema)


def _structured_contract(
    schema: dict[str, Any] | type,
) -> tuple[dict[str, Any], TypeAdapter[Any]]:
    if isinstance(schema, dict):
        function = schema.get("function")
        if schema.get("type") == "function" and isinstance(function, Mapping):
            typed_function = cast(Mapping[str, Any], function)
            json_schema = typed_function.get("parameters")
            name: object = typed_function.get("name", "response")
        else:
            json_schema = schema
            name = schema.get("title", "response")
        if not isinstance(json_schema, Mapping):
            raise ValueError("DecisionChatModel received an invalid structured output schema")
        json_schema = cast(Mapping[str, Any], json_schema)
        validator = TypeAdapter(dict[str, Any])
    else:
        validator = cast(TypeAdapter[Any], TypeAdapter(schema))
        json_schema = validator.json_schema()
        name = getattr(schema, "__name__", "response")
    return (
        {
            "type": "json_schema",
            "json_schema": {"name": str(name), "schema": dict(json_schema)},
        },
        validator,
    )


def _message_text(message: AIMessage) -> str:
    return cast(str, message.content)


def _chat_result(
    result: Any, call: AIMessage | None = None, *, metadata_key: str = "decisions"
) -> ChatResult:
    usage_metadata = None
    if result.input_tokens is not None or result.output_tokens is not None:
        input_tokens = result.input_tokens or 0
        output_tokens = result.output_tokens or 0
        usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    message = AIMessage(
        content=json.dumps(result.values, ensure_ascii=False, separators=(",", ":")),
        response_metadata={
            "model_name": result.model,
            "provider_name": result.metadata.get("provider", "decisions"),
            metadata_key: result.metadata,
        },
        usage_metadata=usage_metadata,
    )
    if call is not None:
        message = message.model_copy(update={"content": "", "tool_calls": call.tool_calls})
    return ChatResult(
        generations=[ChatGeneration(message=message)],
        llm_output={"model_name": result.model, "provider_name": result.metadata["provider"]},
    )


__all__ = ["DecisionChatModel", "ToolCallProposed", "messages_to_state"]
