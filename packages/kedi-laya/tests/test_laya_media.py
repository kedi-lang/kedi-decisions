"""Prompt images, audio and video reach Laya's engine through both integrations."""

import base64
import io
from typing import Literal

import pytest
from kedi_laya import LayaClient
from langchain_core.messages import HumanMessage
from PIL import Image
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent, ImageUrl
from pydantic_ai.exceptions import UserError

from kedi_decisions.contracts import ChoiceQuestion as Choice
from kedi_decisions.contracts import DecisionMedia


class Color(BaseModel):
    color: Literal["red", "blue"]


def _png(color: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


class EnginePredictor:
    """Like Laya's engine: `predict` takes `image=`, `audio=` and `video=`."""

    def __init__(self):
        self.calls = []

    def predict(self, state, questions, *, image=None, audio=None, video=None):
        given = [value for value in (state, image, audio, video) if value is not None]
        if len(given) != 1:
            raise ValueError(
                f"predict takes exactly one of state=, image=, audio=, video=; got {len(given)}"
            )
        self.calls.append(
            {"state": state, "image": image, "audio": audio, "video": video, "questions": questions}
        )
        color = "red" if image is not None and image.getpixel((0, 0)) == (255, 0, 0) else "blue"
        return {
            "answers": {
                key: {
                    "type": "choice",
                    "choice": color,
                    "confidence": 0.9,
                    "probabilities": {
                        label: 0.9 if label == color else 0.1 for label in question["criteria"]
                    },
                }
                for key, question in questions.items()
            },
            "usage": {"input_tokens": 1, "output_tokens": 0},
        }


class TextPredictor:
    def predict(self, state, questions):
        raise AssertionError("a text-only model must not be called with media")


QUESTION = {"color": Choice(criteria={"red": "red", "blue": "blue"})}


def test_client_sends_the_media_value_with_the_text_as_question_context():
    predictor = EnginePredictor()
    client = LayaClient("vega", predictor=predictor)
    image = DecisionMedia(kind="image", data=_png("red"), media_type="image/png")
    audio = DecisionMedia(kind="audio", data=b"RIFF", media_type="audio/wav")
    question = {"color": Choice(instructions="Pick one.", criteria={"red": "red", "blue": "blue"})}
    answer = client.decide("The shapes in <image 1>", question, media=(image,)).answers["color"]
    assert answer.choice == "red"
    call = predictor.calls[0]
    assert call["state"] is None
    assert isinstance(call["image"], Image.Image) and call["image"].mode == "RGB"
    assert call["questions"]["color"]["instructions"] == "The shapes in <image 1>\n\nPick one."
    client.decide({"text": "<audio 1>"}, QUESTION, media=(audio,))
    assert predictor.calls[1]["audio"] == b"RIFF"
    assert predictor.calls[1]["questions"]["color"]["instructions"] == '{"text": "<audio 1>"}'
    with pytest.raises(ValueError, match="one image, audio or video per decision"):
        client.decide("x", QUESTION, media=(image, audio))


def test_text_only_models_reject_media_with_a_hint():
    client = LayaClient("convaiinnovations/laya", predictor=TextPredictor())
    image = DecisionMedia(kind="image", data=_png("red"))
    with pytest.raises(ValueError, match=r"reads text only.*vega checkpoint"):
        client.decide("x", QUESTION, media=(image,))


@pytest.mark.asyncio
async def test_pydantic_agent_sends_prompt_images_to_laya():
    predictor = EnginePredictor()
    with LayaClient("vega", predictor=predictor) as client:
        agent = Agent(client.as_pydantic_model(), output_type=Color)
        result = await agent.run(
            ["What color is this?", BinaryContent(_png("red"), media_type="image/png")]
        )
        assert result.output.color == "red"
        instructions = predictor.calls[0]["questions"]["color"]["instructions"]
        assert instructions.startswith("What color is this?\n<image 1>")
        with pytest.raises(UserError, match="not URLs"):
            await agent.run(["What color?", ImageUrl("https://example.com/cat.png")])


@pytest.mark.asyncio
async def test_langchain_model_sends_base64_blocks_to_laya():
    predictor = EnginePredictor()
    with LayaClient("vega", predictor=predictor) as client:
        model = client.as_langchain_model().with_structured_output(Color)
        message = HumanMessage(
            content=[
                {"type": "text", "text": "What color is this?"},
                {
                    "type": "image",
                    "base64": base64.b64encode(_png("red")).decode(),
                    "mime_type": "image/png",
                },
            ]
        )
        assert (await model.ainvoke([message])).color == "red"
        context = predictor.calls[0]["questions"]["color"]["instructions"]
        assert "<image 1>" in context and "base64" not in context
        url = HumanMessage(content=[{"type": "image", "url": "https://example.com/cat.png"}])
        with pytest.raises(ValueError, match="without base64 data"):
            await model.ainvoke([url])


class ImageOnlyClient(LayaClient):
    media_kinds = frozenset({"image"})


@pytest.mark.asyncio
async def test_unsupported_media_is_rejected_before_a_call():
    predictor = EnginePredictor()
    with ImageOnlyClient("vega", predictor=predictor) as client:
        agent = Agent(client.as_pydantic_model(), output_type=Color)
        with pytest.raises(UserError, match="does not read audio input"):
            await agent.run(["Listen.", BinaryContent(b"RIFF", media_type="audio/wav")])
        with pytest.raises(UserError, match="do not read BinaryContent input"):
            await agent.run(["Read.", BinaryContent(b"%PDF", media_type="application/pdf")])
        model = client.as_langchain_model().with_structured_output(Color)
        audio = {"type": "audio", "base64": base64.b64encode(b"RIFF").decode()}
        with pytest.raises(ValueError, match="audio input"):
            await model.ainvoke([HumanMessage(content=[audio])])
    assert predictor.calls == []


def test_missing_pillow_has_an_install_hint(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def without_pillow(name, *args, **kwargs):
        if name == "PIL":
            raise ModuleNotFoundError(name, name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pillow)
    client = LayaClient("vega", predictor=EnginePredictor())
    image = DecisionMedia(kind="image", data=_png("red"))
    with pytest.raises(ModuleNotFoundError, match=r"laya\[vision\]"):
        client.decide("x", QUESTION, media=(image,))


def test_vega_loads_laya_engine(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    loader = Mock(return_value=EnginePredictor())
    monkeypatch.setattr("kedi_laya.client.import_module", lambda name: SimpleNamespace(load=loader))
    LayaClient("vega", backend="torch").decide("x", QUESTION)
    loader.assert_called_once_with(vegaml=True)
    loader.reset_mock()
    LayaClient("vega", backend="torch", load_options={"vegaml": "4b"}).decide("x", QUESTION)
    loader.assert_called_once_with(vegaml="4b")
