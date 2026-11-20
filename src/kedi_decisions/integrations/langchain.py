from __future__ import annotations

import base64
import json
from collections.abc import Callable, Mapping, Sequence
from types import TracebackType
from typing import Any, ClassVar, Literal, NoReturn, Self, cast

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

from ..contracts import DecisionMedia
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


def messages_to_state(
    messages: list[BaseMessage], *, media: list[DecisionMedia] | None = None
) -> JSONValue:
    """Project supported LangChain message history into structured Decision state.

    With a `media` list, base64 image, audio and video blocks in user messages are collected
    into it and the state refers to each by a label such as `<image 1>`.
    """

    if media is not None:
        messages = [_with_media_labels(message, media) for message in messages]
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


_MEDIA_BLOCKS = {"image": "image", "audio": "audio", "video": "video"}


def _with_media_labels(message: BaseMessage, media: list[DecisionMedia]) -> BaseMessage:
    if not isinstance(message, HumanMessage) or isinstance(message.content, str):
        return message
    content: list[Any] = []
    for item in message.content:
        block = cast(dict[str, Any], item) if isinstance(item, Mapping) else None
        kind = _MEDIA_BLOCKS.get(block.get("type", "")) if block is not None else None
        if block is None or kind is None:
            content.append(item)
            continue
        if not isinstance(block.get("base64"), str):
            _unsupported_message(f"{kind} block without base64 data; load the file first")
        media.append(
            DecisionMedia(
                kind=cast(Any, kind),
                data=base64.b64decode(block["base64"]),
                media_type=block.get("mime_type"),
            )
        )
        label = f"<{kind} {sum(m.kind == kind for m in media)}>"
        content.append({"type": "text", "text": label})
    return message.model_copy(update={"content": content})


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
    threshold_setting: ClassVar[str] = "decision_boolean_threshold"
    tool_threshold_setting: ClassVar[str] = "decision_tool_call_threshold"
    metadata_key: ClassVar[str] = "decisions"

    model_name: str
    threshold: float = DEFAULT_THRESHOLD
    decision_boolean_threshold: float | None = None
    decision_tool_call_threshold: float = 0.6
    _evaluator: Evaluator = PrivateAttr()

    @field_validator(
        "threshold", "decision_boolean_threshold", "decision_tool_call_threshold", mode="before"
    )
    @classmethod
    def _validate_threshold(cls, value: Any) -> float | None:
        # decision_boolean_threshold is optional: an explicit None means "use the
        # evaluator default", just like omitting it. The other fields still
        # reject None through their float type.
        if value is None:
            return None
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
            **self._evaluation_input(messages),
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
            **self._evaluation_input(messages),
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
        # The Evaluator protocol's aclose() is the full async teardown; the
        # shipped evaluators implement it, but an evaluator with extra async
        # teardown in aclose() was silently skipped before. Keep the partial
        # teardown calls for evaluators that only implement those.
        await self._evaluator.aclose()
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

    def _evaluation_input(self, messages: list[BaseMessage]) -> dict[str, Any]:
        """Evaluator arguments for the messages: the decision state, and its media when the
        evaluator reads media."""
        kinds: frozenset[str] = getattr(self._evaluator, "media_kinds", frozenset())
        if not kinds:
            return {"state": messages_to_state(messages)}
        media: list[DecisionMedia] = []
        state = messages_to_state(messages, media=media)
        unsupported = sorted({m.kind for m in media} - kinds)
        if unsupported:
            _unsupported_message(f"{', '.join(unsupported)} input")
        return {"state": state, "media": tuple(media)} if media else {"state": state}

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
