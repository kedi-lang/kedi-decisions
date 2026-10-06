from __future__ import annotations

from typing import Annotated

import httpx2
import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from typesafe_sdk import AsyncTypeSafeClient

from kedi_typesafe import Rubric, TypeSafeModel
from kedi_typesafe.core import evaluation
from kedi_typesafe.integrations import _langchain_transport
from kedi_typesafe.integrations.langchain import TypeSafeChatModel


class Verdict(BaseModel):
    approved: bool


class ScoredVerdict(BaseModel):
    quality: Annotated[float, Rubric(["Incorrect", "Partly correct", "Correct"])]


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["pydantic", "langchain"])
@pytest.mark.parametrize("explicit", [False, True])
async def test_custom_endpoint_is_used_without_public_fallback(
    adapter: str, explicit: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    endpoint = "https://decision.invalid"
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://unused.invalid" if explicit else endpoint)
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        assert str(request.url) == endpoint + "/v1/systemone"
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx2.Response(
            200,
            json={
                "model": "jev-test",
                "usage": {"input_tokens": 10, "output_tokens": 1},
                "answers": {"approved": {"type": "noul", "noul": 0.95}},
            },
        )

    transport = httpx2.MockTransport(respond)
    if adapter == "pydantic":
        monkeypatch.setattr(
            evaluation,
            "AsyncTypeSafeClient",
            lambda **kwargs: AsyncTypeSafeClient(**kwargs, transport=transport),
        )
        model = TypeSafeModel(
            "jev-test", api_key="test-key", base_url=endpoint if explicit else None
        )
        try:
            result = await Agent(model, output_type=Verdict).run("The request is approved.")
            assert result.output.approved is True
        finally:
            await model.aclose()
    else:
        classifier = _langchain_transport.TypeSafeClassifier
        monkeypatch.setattr(
            _langchain_transport,
            "TypeSafeClassifier",
            lambda **kwargs: classifier(
                **kwargs,
                client=httpx2.Client(transport=transport),
                async_client=httpx2.AsyncClient(transport=transport),
            ),
        )
        model = TypeSafeChatModel(
            "jev-test", api_key="test-key", base_url=endpoint if explicit else None
        )
        try:
            result = await model.with_structured_output(Verdict).ainvoke("The request is approved.")
            assert result.approved is True
        finally:
            await model.aclose_current()
            model.close()
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_langchain_score_transport_preserves_numeric_json_keys(
    asynchronous: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = httpx2.MockTransport(
        lambda request: httpx2.Response(
            200,
            json={
                "model": "jev-test",
                "usage": {"input_tokens": 12, "output_tokens": 1},
                "answers": {
                    "quality": {
                        "type": "score",
                        "score": 1.8,
                        "confidence": 0.9,
                        "legend": {"0": "Incorrect", "1": "Partly correct", "2": "Correct"},
                        "probabilities": {"0": 0.0, "1": 0.2, "2": 0.8},
                    }
                },
            },
        )
    )
    classifier = _langchain_transport.TypeSafeClassifier
    monkeypatch.setattr(
        _langchain_transport,
        "TypeSafeClassifier",
        lambda **kwargs: classifier(
            **kwargs,
            client=httpx2.Client(transport=transport),
            async_client=httpx2.AsyncClient(transport=transport),
        ),
    )
    model = TypeSafeChatModel("jev-test", api_key="test-key", base_url="https://decision.invalid")
    try:
        structured = model.with_structured_output(ScoredVerdict)
        if asynchronous:
            result = await structured.ainvoke("Two plus two equals four.")
        else:
            result = structured.invoke("Two plus two equals four.")
        assert result.quality == pytest.approx(1.8)
    finally:
        await model.aclose_current()
        model.close()
