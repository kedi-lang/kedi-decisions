"""Optional local Laya transport for the shared typed-decision integrations."""

from __future__ import annotations

import asyncio
import platform
import sys
from collections.abc import Mapping
from importlib import import_module
from importlib.util import find_spec
from threading import RLock
from typing import Any, Literal, Protocol

from kedi_decisions import DecisionEvaluator
from kedi_decisions.contracts import (
    BooleanQuestion,
    ChoiceQuestion,
    DecisionResponse,
    DecisionUsage,
    JSONValue,
    Question,
)
from kedi_decisions.evaluation import DEFAULT_THRESHOLD

from ._response import parse_response


class LayaPredictor(Protocol):
    """Implemented by Laya, Laya-MLX, or a compatible user-supplied predictor."""

    def predict(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]: ...


class LayaClient:
    """Reuse one local predictor across synchronous and asynchronous decisions.

    Model loading is lazy. Closing this client drops its reference to the predictor,
    but never closes a caller-owned predictor. No hosted fallback is performed.
    """

    probability_decimals = 4

    def __init__(
        self,
        model_name: str,
        *,
        backend: Literal["auto", "mlx", "torch"] = "auto",
        predictor: LayaPredictor | None = None,
        load_options: Mapping[str, Any] | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("Laya model_name must not be empty")
        if backend not in {"auto", "mlx", "torch"}:
            raise ValueError("Laya backend must be 'auto', 'mlx' or 'torch'")
        if predictor is not None and load_options:
            raise ValueError("load_options cannot be used with an existing predictor")
        self.model_name = model_name
        self.backend = (
            (
                "mlx"
                if sys.platform == "darwin"
                and platform.machine() == "arm64"
                and find_spec("laya_mlx") is not None
                else "torch"
            )
            if backend == "auto"
            else backend
        )
        self.provider_name = "laya"
        self.base_url = "local://laya"
        self._predictor = predictor
        self._borrowed = predictor is not None
        self._load_options = dict(load_options or {})
        self._lock = RLock()
        self._closed = False

    def decide(
        self,
        state: JSONValue,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
    ) -> DecisionResponse:
        if model is not None and model != self.model_name:
            raise ValueError("The request model does not match the loaded Laya client")
        payload = {
            name: {
                "type": "noul"
                if isinstance(question, BooleanQuestion)
                else "choice"
                if isinstance(question, ChoiceQuestion)
                else "score",
                "instructions": question.instructions,
                "criteria": dict(question.criteria)
                if isinstance(question, (BooleanQuestion, ChoiceQuestion))
                else list(question.criteria),
            }
            for name, question in questions.items()
        }
        with self._lock:
            if self._closed:
                raise RuntimeError("Laya client is closed")
            if not payload:
                return DecisionResponse(
                    model=self.model_name,
                    answers={},
                    usage=DecisionUsage(input_tokens=0, output_tokens=0),
                )
            if self._predictor is None:
                module_name = "laya_mlx" if self.backend == "mlx" else "laya"
                try:
                    runtime = import_module(module_name)
                except ModuleNotFoundError as exc:
                    if exc.name != module_name:
                        raise
                    extra = "laya-mlx" if self.backend == "mlx" else "laya"
                    raise ModuleNotFoundError(
                        f"Install the optional local runtime: pip install 'kedi-decisions[{extra}]'"
                    ) from exc
                self._predictor = runtime.load(self.model_name, **self._load_options)
            if self.backend == "mlx" and not self._borrowed:
                from ._mlx_context import check_mlx_context

                check_mlx_context(self._predictor, state, payload)
            response = self._predictor.predict(state, payload)
        return parse_response(response, model=self.model_name, expected=set(payload))

    def as_evaluator(self, *, threshold: float = DEFAULT_THRESHOLD) -> DecisionEvaluator:
        return DecisionEvaluator(
            self.model_name,
            threshold=threshold,
            client=AsyncLayaClient(self),
            sync_client=self,
            provider_name=self.provider_name,
            base_url=self.base_url,
        )

    def as_pydantic_model(self, *, threshold: float = DEFAULT_THRESHOLD):
        from .integrations.pydantic import LayaModel

        return LayaModel(self.model_name, threshold=threshold, client=self)

    def as_langchain_model(self, *, threshold: float = DEFAULT_THRESHOLD):
        from .integrations.langchain import LayaChatModel

        return LayaChatModel(self.model_name, threshold=threshold, client=self)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._predictor = None

    def __enter__(self) -> LayaClient:
        if self._closed:
            raise RuntimeError("Laya client is closed")
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class AsyncLayaClient:
    """Offload local inference without blocking the agent's event loop."""

    probability_decimals = LayaClient.probability_decimals

    def __init__(self, client: LayaClient) -> None:
        self.client = client
        self.provider_name = client.provider_name
        self.base_url = client.base_url

    async def decide(
        self,
        state: JSONValue,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
    ) -> DecisionResponse:
        # Cancellation does not stop a running native kernel. The client's lock
        # remains held by the worker until inference really finishes.
        return await asyncio.to_thread(self.client.decide, state, questions, model=model)

    async def aclose(self) -> None:
        await asyncio.to_thread(self.client.close)
