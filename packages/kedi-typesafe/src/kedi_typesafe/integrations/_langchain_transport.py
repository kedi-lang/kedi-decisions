from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from typing import Any
from weakref import WeakKeyDictionary

from langchain_typesafe import TypeSafeClassifier
from langchain_typesafe.types import ClassifierRequest, State
from langchain_typesafe.types import Question as ClassifierQuestion
from pydantic import JsonValue, TypeAdapter
from typesafe_sdk import Asset, AsyncTypeSafeClient, Question, SystemOneResponse, TypeSafeClient

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
        self._async: WeakKeyDictionary[asyncio.AbstractEventLoop, TypeSafeClassifier] = (
            WeakKeyDictionary()
        )
        """Async classifiers, keyed by event loop. The weak keys release a
        classifier as soon as its loop is collected instead of leaking it."""

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

    def _client_options(self, model: str | None) -> dict[str, Any]:
        return {
            "api_key": self.api_key
            if self.api_key is not None
            else os.environ.get("TYPESAFE_API_KEY", ""),
            "base_url": self.base_url,
            "model": model or "jev-latest",
            "timeout": self.timeout if self.timeout is not None else 30.0,
        }

    @staticmethod
    def _payload(state: JSONValue, questions: Mapping[str, Question]) -> ClassifierRequest:
        return {
            "state": _STATE.validate_python(state),
            "questions": _QUESTIONS.validate_python(
                _SDK_QUESTIONS.dump_python(dict(questions), mode="json")
            ),
        }

    async def system_one(
        self,
        state: JSONValue,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        assets: Mapping[str, Asset] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> SystemOneResponse:
        if assets:
            # The classifier takes no assets; images go through the SDK client directly.
            async with AsyncTypeSafeClient(**self._client_options(model)) as client:
                return await client.system_one(  # pyright: ignore[reportUnknownMemberType]
                    state, questions, model=model, assets=assets, extra_headers=extra_headers
                )
        request = self._request(model, asynchronous=True)
        # Pinned classifier transport avoids double-counting a nested LLM span.
        response = await request._aclassify(self._payload(state, questions))  # pyright: ignore[reportPrivateUsage]
        return SystemOneResponse.model_validate_json(
            response.model_dump_json(exclude={"request_id"})
        )

    def system_one_sync(
        self,
        state: JSONValue,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        assets: Mapping[str, Asset] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> SystemOneResponse:
        if assets:
            with TypeSafeClient(**self._client_options(model)) as client:
                return client.system_one(  # pyright: ignore[reportUnknownMemberType]
                    state, questions, model=model, assets=assets, extra_headers=extra_headers
                )
        request = self._request(model, asynchronous=False)
        response = request._classify(self._payload(state, questions))  # pyright: ignore[reportPrivateUsage]
        return SystemOneResponse.model_validate_json(
            response.model_dump_json(exclude={"request_id"})
        )

    async def aclose(self) -> None:
        """Close every cached async classifier, including ones bound to closed
        loops, and the async client paired with the sync classifier."""
        classifiers = list(self._async.values())
        self._async.clear()
        if self._sync is not None:
            classifiers.append(self._sync)
        await asyncio.gather(
            *(base.async_client.aclose() for base in classifiers if base.async_client is not None)
        )
        for base in classifiers:
            if base is self._sync:
                continue
            if base.client is not None:
                base.client.close()

    async def aclose_current(self) -> None:
        """Close the async classifier bound to the current event loop."""
        base = self._async.pop(asyncio.get_running_loop(), None)
        if base is not None:
            if base.async_client is not None:
                await base.async_client.aclose()
            if base.client is not None:
                base.client.close()

    def close(self) -> None:
        if self._sync is not None:
            if self._sync.client is not None:
                self._sync.client.close()
            if self._sync.async_client is not None:
                # The async client paired with the sync classifier is created
                # at construction even though the sync path never uses it; close
                # it instead of leaving it open.
                _close_async_client(self._sync.async_client)
            self._sync = None


def _close_async_client(client: Any) -> None:
    """Close an async httpx client from synchronous teardown. Runs a fresh loop
    when none is running; otherwise schedules the close on the running loop so
    `close()` never raises for being called from async code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(client.aclose())
    else:
        loop.create_task(client.aclose())


class SyncClassifierTransport:
    def __init__(self, transport: ClassifierTransport) -> None:
        self.transport = transport

    def system_one(
        self,
        state: JSONValue,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        assets: Mapping[str, Asset] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> SystemOneResponse:
        return self.transport.system_one_sync(
            state, questions, model=model, assets=assets, extra_headers=extra_headers
        )

    def close(self) -> None:
        self.transport.close()
