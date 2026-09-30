from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic_ai.settings import ModelSettings

from kedi_decisions.evaluation import DEFAULT_THRESHOLD
from kedi_decisions.integrations.pydantic import DecisionModel

from ..client import LayaClient


class LayaModel(DecisionModel):
    def __init__(
        self,
        model_name: str,
        *,
        backend: Literal["auto", "mlx", "torch"] = "auto",
        client: LayaClient | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        load_options: dict[str, Any] | None = None,
        settings: ModelSettings | None = None,
    ) -> None:
        if client is not None and client.model_name != model_name:
            raise ValueError("Laya model name does not match its client")
        if client is not None and (load_options or backend not in ("auto", client.backend)):
            raise ValueError("Configure backend and load_options on the existing Laya client")
        self._owns_client = client is None
        self._client = client or LayaClient(model_name, backend=backend, load_options=load_options)
        super().__init__(self._client.as_evaluator(threshold=threshold), settings=settings)

    @property
    def backend(self) -> str:
        return self._client.backend

    async def aclose(self) -> None:
        if self._owns_client:
            await asyncio.to_thread(self._client.close)
