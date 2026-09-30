from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import PrivateAttr

from kedi_decisions.evaluation import DEFAULT_THRESHOLD
from kedi_decisions.integrations.langchain import DecisionChatModel

from ..client import LayaClient


class LayaChatModel(DecisionChatModel):
    _client: LayaClient = PrivateAttr()
    _owns_client: bool = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        *,
        backend: Literal["auto", "mlx", "torch"] = "auto",
        client: LayaClient | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        load_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if client is not None and client.model_name != model_name:
            raise ValueError("Laya model name does not match its client")
        if client is not None and (load_options or backend not in ("auto", client.backend)):
            raise ValueError("Configure backend and load_options on the existing Laya client")
        owned = client is None
        client = client or LayaClient(model_name, backend=backend, load_options=load_options)
        super().__init__(
            model_name,
            evaluator=client.as_evaluator(threshold=threshold),
            threshold=threshold,
            **kwargs,
        )
        self._client = client
        self._owns_client = owned

    @property
    def backend(self) -> str:
        return self._client.backend

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)
