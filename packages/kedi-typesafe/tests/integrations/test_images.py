"""Prompt images reach TypeSafe as assets referenced from the decision state."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import BinaryContent, ImageUrl, ModelRequest, UserPromptPart
from typesafe_sdk import Image, NoulAnswer, Question, SystemOneResponse, Usage

from kedi_typesafe.core.evaluation import JSONValue
from kedi_typesafe.core.images import with_image_assets
from kedi_typesafe.integrations.langchain import TypeSafeChatModel
from kedi_typesafe.integrations.pydantic import TypeSafeModel, messages_to_request

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


class Check(BaseModel):
    readable: bool = Field(description="Is the receipt legible?")


class Client:
    def __init__(self) -> None:
        self.calls: list[tuple[JSONValue, Mapping[str, Question], dict[str, Any]]] = []

    def _answer(self, state: JSONValue, questions: Mapping[str, Question], **kwargs: Any):
        self.calls.append((state, questions, kwargs))
        return SystemOneResponse(
            model="jev-latest",
            usage=Usage(input_tokens=1, output_tokens=1),
            answers={"readable": NoulAnswer(noul=0.9)},
        )

    async def system_one(self, state: JSONValue, questions: Mapping[str, Question], **kwargs: Any):
        return self._answer(state, questions, **kwargs)

    async def aclose(self) -> None:
        pass


class SyncClient(Client):
    def system_one(self, state: JSONValue, questions: Mapping[str, Question], **kwargs: Any):  # type: ignore[override]
        return self._answer(state, questions, **kwargs)

    def close(self) -> None:
        pass


def _single_asset(kwargs: dict[str, Any]) -> tuple[str, Image]:
    [(reference, image)] = kwargs["assets"].items()
    return reference, image


def test_pydantic_agent_sends_prompt_images_as_assets() -> None:
    client = Client()
    agent = Agent(TypeSafeModel("jev-latest", client=client), output_type=Check)

    result = agent.run_sync(["The receipt:", BinaryContent(PNG, media_type="image/png")])

    assert result.output == Check(readable=True)
    [(state, _, kwargs)] = client.calls
    reference, image = _single_asset(kwargs)
    assert reference == str(image)
    assert state == f"The receipt:\n\n{reference}"


def test_text_only_requests_send_no_assets() -> None:
    client = Client()
    Agent(TypeSafeModel("jev-latest", client=client), output_type=Check).run_sync("Legible?")

    [(_, _, kwargs)] = client.calls
    assert "assets" not in kwargs


def test_image_urls_must_be_downloaded_first() -> None:
    request = ModelRequest(parts=[UserPromptPart(["See", ImageUrl("https://example.com/a.png")])])
    with pytest.raises(UserError, match="not URLs"):
        messages_to_request([request])


def test_langchain_model_sends_image_blocks_as_assets() -> None:
    client = SyncClient()
    model = TypeSafeChatModel(sync_client=client).with_structured_output(Check)
    encoded = base64.b64encode(PNG).decode()

    output = model.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "The receipt:"},
                    {"type": "image", "base64": encoded, "mime_type": "image/png"},
                ]
            )
        ]
    )

    assert output == Check(readable=True)
    [(state, _, kwargs)] = client.calls
    reference, _ = _single_asset(kwargs)
    assert reference in str(state)
    assert encoded not in str(state)


def test_langchain_image_urls_must_be_downloaded_first() -> None:
    model = TypeSafeChatModel(sync_client=SyncClient()).with_structured_output(Check)
    with pytest.raises(ValueError, match="not URLs"):
        model.invoke(
            [HumanMessage(content=[{"type": "image", "url": "https://example.com/a.png"}])]
        )


def test_labels_map_to_their_own_images_anywhere_in_the_state() -> None:
    state, assets = with_image_assets(
        {"history": [{"user": "<image 1>"}], "text": "<image 2> and <image 3>"}, [PNG, PNG]
    )

    first, second = assets
    assert state == {"history": [{"user": first}], "text": f"{second} and <image 3>"}
    assert with_image_assets("no images", []) == ("no images", {})


class _SDKClient(SyncClient):
    opened: list[dict[str, Any]] = []

    def __init__(self, **options: Any) -> None:
        super().__init__()
        self.opened.append(options)
        _SDKClient.instances.append(self)

    instances: list[_SDKClient] = []

    def __enter__(self) -> _SDKClient:
        return self

    def __exit__(self, *exc: object) -> None:
        pass


class _AsyncSDKClient(Client):
    instances: list[_AsyncSDKClient] = []

    def __init__(self, **options: Any) -> None:
        super().__init__()
        _AsyncSDKClient.instances.append(self)

    async def __aenter__(self) -> _AsyncSDKClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass


@pytest.mark.asyncio
async def test_default_langchain_transport_sends_images_through_the_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kedi_typesafe.integrations import _langchain_transport

    monkeypatch.setattr(_langchain_transport, "TypeSafeClient", _SDKClient)
    monkeypatch.setattr(_langchain_transport, "AsyncTypeSafeClient", _AsyncSDKClient)
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    model = TypeSafeChatModel().with_structured_output(Check)
    message = HumanMessage(
        content=[
            {"type": "text", "text": "The receipt:"},
            {"type": "image", "base64": base64.b64encode(PNG).decode(), "mime_type": "image/png"},
        ]
    )

    assert model.invoke([message]) == Check(readable=True)
    assert await model.ainvoke([message]) == Check(readable=True)

    [sync_client] = _SDKClient.instances
    [async_client] = _AsyncSDKClient.instances
    assert _SDKClient.opened == [
        {
            "api_key": "ts-test",
            "base_url": "https://api.typesafe.ai",
            "model": "jev-latest",
            "timeout": 30.0,
        }
    ]
    for client in (sync_client, async_client):
        [(_, _, kwargs)] = client.calls
        _single_asset(kwargs)
