from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping

from langchain_typesafe import TypeSafeClassifier
from langchain_typesafe.types import ClassifierRequest, State
from langchain_typesafe.types import Question as ClassifierQuestion
from pydantic import JsonValue, TypeAdapter
from typesafe_sdk import Question, SystemOneResponse

from ..core.evaluation import JSONValue

_QUESTIONS = TypeAdapter(dict[str, ClassifierQuestion])
_SDK_QUESTIONS = TypeAdapter(dict[str, Question])
# The bridge accepts only the JSON subset of the classifier's broader State type.
_STATE: TypeAdapter[State] = TypeAdapter(str | list[JsonValue] | dict[str, JsonValue])


class ClassifierTransport:
    """Reuse official transport without nesting another LLM invocation span."""

    def __init__(
        self, *, api_key: str | None, timeout: float | None, base_url: str | None = None
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = base_url or os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai")
        self._sync: TypeSafeClassifier | None = None
        self._async: dict[asyncio.AbstractEventLoop, TypeSafeClassifier] = {}

    def _request(
        self,
        model: str | None,
        *,
        asynchronous: bool,
    ) -> TypeSafeClassifier:
        loop = asyncio.get_running_loop() if asynchronous else None
        base = self._async.get(loop) if loop is not None else self._sync
        if base is None:
            base = TypeSafeClassifier(
                model=model or "jev-latest",
                base_url=self.base_url,
                timeout=self.timeout if self.timeout is not None else 30.0,
                api_key=self.api_key
                if self.api_key is not None
                else os.environ.get("TYPESAFE_API_KEY", ""),
            )
            if loop is None:
                self._sync = base
            else:
                self._async[loop] = base
        return base.model_copy(update={"model": model or "jev-latest"})

    @staticmethod
    def _payload(state: JSONValue, questions: Mapping[str, Question]) -> ClassifierRequest:
        return {
            "state": _STATE.validate_python(state),
            "questions": _QUESTIONS.validate_python(
                _SDK_QUESTIONS.dump_python(dict(questions), mode="json")
            ),
        }

    async def system_one(
        self, state: JSONValue, questions: Mapping[str, Question], *, model: str | None = None
    ) -> SystemOneResponse:
        request = self._request(model, asynchronous=True)
        # Pinned classifier transport avoids double-counting a nested LLM span.
        response = await request._aclassify(self._payload(state, questions))  # pyright: ignore[reportPrivateUsage]
        return SystemOneResponse.model_validate_json(
            response.model_dump_json(exclude={"request_id"})
        )

    def system_one_sync(
        self, state: JSONValue, questions: Mapping[str, Question], *, model: str | None = None
    ) -> SystemOneResponse:
        request = self._request(model, asynchronous=False)
        response = request._classify(self._payload(state, questions))  # pyright: ignore[reportPrivateUsage]
        return SystemOneResponse.model_validate_json(
            response.model_dump_json(exclude={"request_id"})
        )

    async def aclose(self) -> None:
        base = self._async.pop(asyncio.get_running_loop(), None)
        if base is not None:
            assert base.async_client is not None and base.client is not None
            await base.async_client.aclose()
            base.client.close()

    def close(self) -> None:
        if self._sync is not None:
            assert self._sync.client is not None
            self._sync.client.close()
            # The async client paired with the sync classifier was never used.
            self._sync = None


class SyncClassifierTransport:
    def __init__(self, transport: ClassifierTransport) -> None:
        self.transport = transport

    def system_one(
        self, state: JSONValue, questions: Mapping[str, Question], *, model: str | None = None
    ) -> SystemOneResponse:
        return self.transport.system_one_sync(state, questions, model=model)

    def close(self) -> None:
        self.transport.close()
