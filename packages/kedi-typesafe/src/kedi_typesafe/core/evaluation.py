from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeAlias

from kedi_decisions.evaluation import EvaluationResult, ResultDecoder
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Question,
    SystemOneResponse,
    TypeSafeClient,
    Usage,
)

from .extraction import CandidateExtractor
from .schema import EvaluationPlan, build_evaluation_plan
from .transport import response_from_sdk

_JSONData: TypeAlias = (
    str | int | float | bool | None | Mapping[str, "_JSONData"] | Sequence["_JSONData"]
)
JSONValue: TypeAlias = str | Mapping[str, _JSONData] | Sequence[_JSONData]
DEFAULT_THRESHOLD = 0.85


def _request_metadata(
    state: JSONValue,
    plan: EvaluationPlan,
    *,
    model: str,
    threshold: float,
) -> dict[str, str]:
    questions = [
        {
            "key": question.key,
            "kind": question.kind,
            "instructions": question.instructions,
            "options": list(question.native_options),
            "path": list(question.path),
            "probability": question.probability,
            "nullable": question.nullable,
            "local_none": question.local_none,
            "label": question.label,
            "criteria": question.criteria,
            "integer_score": question.integer_score,
        }
        for question in plan.questions
    ]
    state_fingerprint = _json_fingerprint(state)
    questions_fingerprint = _json_fingerprint(questions)
    config_fingerprint = _json_fingerprint(
        {
            "model": model,
            "boolean_threshold": threshold,
            "boolean_comparator": ">",
        }
    )
    return {
        "state_fingerprint": state_fingerprint,
        "questions_fingerprint": questions_fingerprint,
        "config_fingerprint": config_fingerprint,
        "request_fingerprint": _json_fingerprint(
            {
                "state_fingerprint": state_fingerprint,
                "questions_fingerprint": questions_fingerprint,
                "config_fingerprint": config_fingerprint,
            }
        ),
    }


def _json_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"kedi-typesafe-request-v1:sha256:{hashlib.sha256(payload).hexdigest()}"


def validate_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("TypeSafe boolean threshold must be a finite number")
    if not 0 <= value <= 1:
        raise ValueError("TypeSafe boolean threshold must be between 0 and 1")
    return float(value)


class AsyncSystemOneClient(Protocol):
    async def system_one(
        self,
        state: JSONValue,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
    ) -> SystemOneResponse: ...

    async def aclose(self) -> None: ...


class SystemOneClient(Protocol):
    def system_one(
        self,
        state: JSONValue,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
    ) -> SystemOneResponse: ...

    def close(self) -> None: ...


class TypeSafeEvaluator(ResultDecoder):
    """Execute validated Jev question plans and normalize their typed answers."""

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
    ) -> None:
        model_name = model_name.strip()
        if not model_name:
            raise ValueError("TypeSafe model name must not be empty")
        self.model_name = model_name
        self.threshold = validate_threshold(threshold)
        self.text_extractors = dict(text_extractors or {})
        self._api_key = api_key
        self._base_url = base_url or os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai")
        self._timeout = timeout
        self._borrowed_client = client
        self._clients: dict[asyncio.AbstractEventLoop, AsyncSystemOneClient] = {}
        self._owns_sync_client = sync_client is None
        self._sync_client = sync_client
        self._probability_decimals = getattr(client or sync_client, "probability_decimals", None)

    @property
    def provider_name(self) -> str:
        return getattr(self._borrowed_client or self._sync_client, "provider_name", "typesafe")

    @property
    def base_url(self) -> str:
        return getattr(self._borrowed_client or self._sync_client, "base_url", self._base_url)

    async def evaluate(
        self,
        *,
        state: JSONValue,
        schema: Mapping[str, Any],
        threshold: float | None = None,
    ) -> EvaluationResult:
        plan = self._plan(state, schema)
        effective_threshold = self.threshold if threshold is None else validate_threshold(threshold)
        request_metadata = _request_metadata(
            state,
            plan,
            model=self.model_name,
            threshold=effective_threshold,
        )
        if not plan.native_questions():
            return self._result(
                SystemOneResponse(
                    model=self.model_name, answers={}, usage=Usage(input_tokens=0, output_tokens=0)
                ),
                plan,
                effective_threshold,
                request_metadata=request_metadata,
            )
        client = self.async_client()
        response = await client.system_one(
            state,
            plan.native_questions(),
            model=self.model_name,
        )
        return self._result(
            response,
            plan,
            effective_threshold,
            request_metadata=request_metadata,
        )

    def async_client(self) -> AsyncSystemOneClient:
        """Return the borrowed client or the owned client for this event loop."""
        client = self._borrowed_client
        if client is None:
            loop = asyncio.get_running_loop()
            client = self._clients.get(loop)
            if client is None:
                client = AsyncTypeSafeClient(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    model=self.model_name,
                    timeout=self._timeout,
                )
                self._clients[loop] = client
        return client

    def evaluate_sync(
        self,
        *,
        state: JSONValue,
        schema: Mapping[str, Any],
        threshold: float | None = None,
    ) -> EvaluationResult:
        plan = self._plan(state, schema)
        effective_threshold = self.threshold if threshold is None else validate_threshold(threshold)
        request_metadata = _request_metadata(
            state,
            plan,
            model=self.model_name,
            threshold=effective_threshold,
        )
        if not plan.native_questions():
            return self._result(
                SystemOneResponse(
                    model=self.model_name, answers={}, usage=Usage(input_tokens=0, output_tokens=0)
                ),
                plan,
                effective_threshold,
                request_metadata=request_metadata,
            )
        client = self._sync_client
        if client is None:
            client = TypeSafeClient(
                api_key=self._api_key,
                base_url=self._base_url,
                model=self.model_name,
                timeout=self._timeout,
            )
            self._sync_client = client
        # TypeSafe SDK 0.7.2 exposes a partially unknown recursive JSON annotation.
        response = client.system_one(  # pyright: ignore[reportUnknownMemberType]
            state,
            plan.native_questions(),
            model=self.model_name,
        )
        return self._result(
            response,
            plan,
            effective_threshold,
            request_metadata=request_metadata,
        )

    def close(self) -> None:
        if self._owns_sync_client and self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None

    def _plan(
        self,
        state: JSONValue,
        schema: Mapping[str, Any],
    ) -> EvaluationPlan:
        return build_evaluation_plan(
            schema,
            text_extractors=self.text_extractors,
        ).resolve(state)

    async def aclose(self) -> None:
        await self.aclose_current()
        self.close()

    async def aclose_current(self) -> None:
        """Close the owned async client bound to the current event loop."""

        if self._borrowed_client is not None:
            return
        loop = asyncio.get_running_loop()
        client = self._clients.pop(loop, None)
        if client is not None:
            await client.aclose()

    def _result(
        self,
        response: SystemOneResponse,
        plan: EvaluationPlan,
        threshold: float | None = None,
        *,
        request_metadata: Mapping[str, str] | None = None,
    ) -> EvaluationResult:
        return self._decode_result(
            response_from_sdk(response), plan, threshold, request_metadata=request_metadata
        )


__all__ = [
    "AsyncSystemOneClient",
    "EvaluationResult",
    "JSONValue",
    "SystemOneClient",
    "TypeSafeEvaluator",
]
