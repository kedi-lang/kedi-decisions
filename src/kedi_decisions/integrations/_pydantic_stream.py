from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, ModelResponseStreamEvent, TextPart, ToolCallPart
from pydantic_ai.models import StreamedResponse


@dataclass
class DecisionStream(StreamedResponse):
    _response: ModelResponse

    def __post_init__(self) -> None:
        self._usage = self._response.usage
        self.provider_details = self._response.provider_details
        self.finish_reason = self._response.finish_reason

    async def close_stream(self) -> None:
        """The complete decision is obtained before exposing stream events."""

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        for index, part in enumerate(self._response.parts):
            if isinstance(part, TextPart):
                for event in self._parts_manager.handle_text_delta(
                    vendor_part_id=index, content=part.content
                ):
                    yield event
            elif isinstance(part, ToolCallPart):
                yield self._parts_manager.handle_tool_call_part(
                    vendor_part_id=index,
                    tool_name=part.tool_name,
                    args=part.args_as_dict(),
                    tool_call_id=part.tool_call_id,
                )
            else:
                raise UnexpectedModelBehavior(
                    f"Unsupported decision response part: {part.part_kind}"
                )

    @property
    def model_name(self) -> str:
        return self._response.model_name or ""

    @property
    def provider_name(self) -> str | None:
        return self._response.provider_name

    @property
    def provider_url(self) -> str | None:
        return self._response.provider_url

    @property
    def timestamp(self) -> datetime:
        return self._response.timestamp
