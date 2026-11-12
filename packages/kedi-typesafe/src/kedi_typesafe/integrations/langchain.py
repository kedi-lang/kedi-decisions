from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any, ClassVar

from kedi_decisions.integrations.langchain import DecisionChatModel
from kedi_decisions.integrations.langchain import (
    ToolCallProposed as ToolCallProposed,
)
from kedi_decisions.integrations.langchain import (
    messages_to_state as messages_to_state,
)
from langchain_core.messages import BaseMessage
from pydantic import PrivateAttr

from ..core import CandidateExtractor, TypeSafeEvaluator
from ..core.evaluation import (
    DEFAULT_THRESHOLD,
    AsyncSystemOneClient,
    SystemOneClient,
)
from ..core.images import with_image_assets
from ._langchain_transport import ClassifierTransport, SyncClassifierTransport


class TypeSafeChatModel(DecisionChatModel):
    """Jev transport with the established TypeSafe settings and metadata names."""

    metadata_key: ClassVar[str] = "typesafe"
    _transport: ClassifierTransport | None = PrivateAttr(default=None)

    def _evaluation_input(self, messages: list[BaseMessage]) -> dict[str, Any]:
        """Evaluator arguments for the messages: the state, and its images as assets."""
        images: list[bytes] = []
        labeled = [_label_images(message, images) for message in messages]
        state, assets = with_image_assets(messages_to_state(labeled), images)
        return {"state": state, "assets": assets}

    def __init__(
        self,
        model_name: str = "jev-latest",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        timeout: float | None = None,
        client: AsyncSystemOneClient | None = None,
        sync_client: SystemOneClient | None = None,
        text_extractors: Mapping[str, CandidateExtractor] | None = None,
        **kwargs: Any,
    ) -> None:
        transport = None
        if client is None or sync_client is None:
            transport = ClassifierTransport(api_key=api_key, timeout=timeout, base_url=base_url)
            client = client or transport
            sync_client = sync_client or SyncClassifierTransport(transport)
        evaluator = TypeSafeEvaluator(
            model_name,
            api_key=api_key,
            base_url=base_url,
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


def _label_images(message: BaseMessage, images: list[bytes]) -> BaseMessage:
    """Replace each image block with an `<image N>` label and collect its data."""
    if isinstance(message.content, str):
        return message
    content: list[Any] = []
    for block in message.content:
        if isinstance(block, dict) and block.get("type") == "image":
            data = block.get("base64")
            if not isinstance(data, str):
                raise ValueError("TypeSafe reads image data, not URLs: download the image first")
            images.append(base64.b64decode(data, validate=True))
            content.append({"type": "text", "text": f"<image {len(images)}>"})
        else:
            content.append(block)
    return message.model_copy(update={"content": content})
