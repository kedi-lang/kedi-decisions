from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from kedi_decisions.integrations.langchain import DecisionChatModel
from kedi_decisions.integrations.langchain import (
    ToolCallProposed as ToolCallProposed,
)
from kedi_decisions.integrations.langchain import (
    messages_to_state as messages_to_state,
)
from pydantic import PrivateAttr, field_validator

from ..core import CandidateExtractor, TypeSafeEvaluator
from ..core.evaluation import (
    DEFAULT_THRESHOLD,
    AsyncSystemOneClient,
    SystemOneClient,
    validate_threshold,
)
from ._langchain_transport import ClassifierTransport, SyncClassifierTransport


class TypeSafeChatModel(DecisionChatModel):
    """Jev transport with the established TypeSafe settings and metadata names."""

    threshold_setting: ClassVar[str] = "typesafe_threshold"
    tool_threshold_setting: ClassVar[str] = "typesafe_tool_call_threshold"
    metadata_key: ClassVar[str] = "typesafe"
    typesafe_threshold: float | None = None
    typesafe_tool_call_threshold: float = 0.6
    _transport: ClassifierTransport | None = PrivateAttr(default=None)

    @field_validator("typesafe_threshold", "typesafe_tool_call_threshold", mode="before")
    @classmethod
    def _validate_typesafe_threshold(cls, value: Any) -> float:
        return validate_threshold(value)

    def __init__(
        self,
        model_name: str = "jev-latest",
        *,
        api_key: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        timeout: float | None = None,
        client: AsyncSystemOneClient | None = None,
        sync_client: SystemOneClient | None = None,
        text_extractors: Mapping[str, CandidateExtractor] | None = None,
        **kwargs: Any,
    ) -> None:
        transport = None
        if client is None or sync_client is None:
            transport = ClassifierTransport(api_key=api_key, timeout=timeout)
            client = client or transport
            sync_client = sync_client or SyncClassifierTransport(transport)
        evaluator = TypeSafeEvaluator(
            model_name,
            api_key=api_key,
            threshold=threshold,
            timeout=timeout,
            client=client,
            sync_client=sync_client,
            text_extractors=text_extractors,
        )
        super().__init__(model_name, evaluator=evaluator, threshold=threshold, **kwargs)
        self._transport = transport

    @property
    def _llm_type(self) -> str:
        return "typesafe-jev"

    def close(self) -> None:
        self._evaluator.close()
        if self._transport is not None:
            self._transport.close()

    async def aclose_current(self) -> None:
        await self._evaluator.aclose_current()
        if self._transport is not None:
            await self._transport.aclose()


__all__ = ["TypeSafeChatModel", "ToolCallProposed", "messages_to_state"]
