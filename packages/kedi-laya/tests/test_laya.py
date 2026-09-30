import asyncio
from threading import Event
from types import SimpleNamespace
from typing import Annotated, Literal
from unittest.mock import Mock

import pytest
from kedi_laya import AsyncLayaClient, LayaClient
from pydantic import BaseModel
from pydantic_ai import Agent

from kedi_decisions import Probability, Rubric
from kedi_decisions.contracts import BooleanQuestion as Noul
from kedi_decisions.contracts import ChoiceQuestion as Choice
from kedi_decisions.errors import DecisionResponseError


class Decision(BaseModel):
    route: Literal["billing", "technical"]
    urgent: bool
    likelihood: Probability
    priority: Annotated[float, Rubric(["low", "medium", "high"])]


class Predictor:
    def __init__(self):
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        answers = {}
        for key, question in questions.items():
            if question["type"] == "choice":
                labels = list(question["criteria"])
                answer = {
                    "choice": labels[0],
                    "confidence": 0.8,
                    "probabilities": {
                        label: 1.0 if i == 0 else 0.0 for i, label in enumerate(labels)
                    },
                }
            elif question["type"] == "score":
                answer = {
                    "score": 1.5,
                    "confidence": 0.4,
                    "legend": dict(enumerate(question["criteria"])),
                    "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
                }
            else:
                answer = {"noul": 0.9, "confidence": 0.9}
            answers[key] = {"type": question["type"], "action": {"act_probability": 0.7}, **answer}
        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {"input_tokens": 30 * len(questions), "output_tokens": 0},
        }


def test_evaluator_preserves_decisions_metadata_and_model_identity():
    predictor = Predictor()
    with LayaClient("custom/checkpoint", predictor=predictor) as client:
        result = client.as_evaluator().evaluate_sync(
            state="A duplicate charge", schema=Decision.model_json_schema()
        )
        assert Decision.model_validate(result.values).model_dump() == {
            "route": "billing",
            "urgent": True,
            "likelihood": 0.9,
            "priority": 1.5,
        }
        assert result.metadata["provider"] == "laya"
        assert result.metadata["answers"]["route"]["confidence"] == 0.8
        assert result.model == "custom/checkpoint"
        assert result.input_tokens == 120 and result.output_tokens == 0
        assert len(predictor.calls) == 1


@pytest.mark.asyncio
async def test_pydantic_agent_and_langchain_use_same_local_predictor(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with LayaClient("local-test", predictor=Predictor()) as client:
        model = client.as_pydantic_model(threshold=0.95)
        assert model.system == "laya" and model.base_url == "local://laya"
        result = await Agent(model, output_type=Decision).run("A duplicate charge")
        assert result.output.urgent is False
        assert result.usage.output_tokens == 0
        lc = client.as_langchain_model().with_structured_output(Decision, include_raw=True)
        answer = await lc.ainvoke("A duplicate charge")
        assert answer["parsed"].urgent is True
        assert answer["raw"].response_metadata["provider_name"] == "laya"


def test_lazy_load_once_and_forward_loader_options(monkeypatch):
    loader = Mock(return_value=Predictor())
    monkeypatch.setattr("kedi_laya.client.import_module", lambda name: SimpleNamespace(load=loader))
    client = LayaClient("checkpoint", backend="torch", load_options={"device": "cpu"})
    assert not loader.called
    for _ in range(2):
        client.decide("state", {"answer": Noul()})
    loader.assert_called_once_with("checkpoint", device="cpu")
    assert client.as_pydantic_model().system == "laya"
    client.close()
    client.close()
    with pytest.raises(RuntimeError, match="closed"):
        client.decide("state", {})
    with pytest.raises(RuntimeError, match="closed"):
        client.__enter__()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_name": " "},
        {"model_name": "x", "backend": "unknown"},
        {"model_name": "x", "predictor": Predictor(), "load_options": {"revision": "main"}},
    ],
)
def test_invalid_client_configuration(kwargs):
    with pytest.raises(ValueError):
        LayaClient(**kwargs)


def test_empty_request_never_loads_model_and_model_mismatch_fails():
    client = LayaClient("unloaded")
    assert client.decide("state", {}).usage.input_tokens == 0
    with pytest.raises(ValueError, match="does not match"):
        client.decide("state", {}, model="another")


def test_missing_optional_runtime_has_install_hint(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name, name=name)

    monkeypatch.setattr("kedi_laya.client.import_module", missing)
    with pytest.raises(ModuleNotFoundError, match=r"kedi-decisions\[laya-mlx\]"):
        LayaClient("x", backend="mlx").decide("state", {"a": Noul()})

    def broken_dependency(name):
        raise ModuleNotFoundError("broken dependency", name="numpy")

    monkeypatch.setattr("kedi_laya.client.import_module", broken_dependency)
    with pytest.raises(ModuleNotFoundError, match="broken dependency"):
        LayaClient("x").decide("state", {"a": Noul()})


def test_missing_answers_fail_instead_of_fabricating_a_decision():
    client = LayaClient("x", predictor=SimpleNamespace(predict=lambda *args: {"answers": {}}))
    with pytest.raises(DecisionResponseError, match="answer keys"):
        client.decide("state", {"a": Noul()})


@pytest.mark.parametrize("kind", ["choice", "score"])
@pytest.mark.parametrize(
    "probabilities,valid", [([0.3333, 0.3333, 0.3333], True), ([0.3, 0.3, 0.3], False)]
)
def test_laya_rounding_tolerance_preserves_raw_probabilities(kind, probabilities, valid):
    labels = ["a", "b", "c"] if kind == "choice" else ["0", "1", "2"]
    raw = dict(zip(labels, probabilities, strict=True))
    answer = {"type": kind, "confidence": 0.1, "probabilities": raw}
    if kind == "choice":
        answer["choice"] = "a"
        schema = {"type": "string", "enum": labels}
    else:
        answer.update(score=1.0, legend={"0": "low", "1": "medium", "2": "high"})
        schema = {
            "type": "number",
            "x-typesafe": {"kind": "score", "criteria": ["low", "medium", "high"]},
        }
    predictor = SimpleNamespace(predict=lambda *args: {"answers": {"x": answer}})
    evaluator = LayaClient("test", predictor=predictor).as_evaluator()
    contract = {
        "type": "object",
        "properties": {"x": schema},
        "required": ["x"],
        "additionalProperties": False,
    }
    if not valid:
        with pytest.raises(DecisionResponseError):
            evaluator.evaluate_sync(state="input", schema=contract)
    else:
        result = evaluator.evaluate_sync(state="input", schema=contract)
        assert list(result.metadata["answers"]["x"]["probabilities"].values()) == probabilities


@pytest.mark.asyncio
async def test_cancelled_call_retains_inference_lock_until_worker_finishes():
    started, release = Event(), Event()
    predictor = Predictor()
    original = predictor.predict

    def slow(state, questions):
        if state == "first":
            started.set()
            assert release.wait(3)
        return original(state, questions)

    predictor.predict = slow
    client = LayaClient("x", predictor=predictor)
    asynchronous = AsyncLayaClient(client)
    questions = {"a": Choice(criteria={"yes": None, "no": None})}
    first = asyncio.create_task(asynchronous.decide("first", questions))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(asynchronous.decide("second", questions))
        await asyncio.sleep(0.02)
        assert not predictor.calls and not second.done()
    finally:
        release.set()
    await second
    assert [state for state, _ in predictor.calls] == ["first", "second"]
    await asynchronous.aclose()


@pytest.mark.parametrize("surface", ["pydantic", "langchain"])
@pytest.mark.asyncio
async def test_borrowed_model_close_preserves_client_and_owned_model_close_releases_it(surface):
    from kedi_laya import LayaChatModel, LayaModel

    cls = LayaModel if surface == "pydantic" else LayaChatModel
    client = LayaClient("test", predictor=Predictor())
    borrowed = cls("test", client=client)
    await borrowed.aclose()
    assert client.decide("state", {"x": Noul()}).answers["x"].probability == 0.9
    owned = cls("unloaded")
    await owned.aclose_current()
    assert not owned._client._closed
    await owned.aclose()
    assert owned._client._closed
    client.close()


@pytest.mark.parametrize("surface", ["pydantic", "langchain"])
@pytest.mark.parametrize(
    "kwargs",
    [{"model_name": "different"}, {"load_options": {"device": "cpu"}}, {"backend": "torch"}],
)
def test_borrowed_client_rejects_conflicting_model_options(surface, kwargs):
    from kedi_laya import LayaChatModel, LayaModel

    cls = LayaModel if surface == "pydantic" else LayaChatModel
    with (
        LayaClient("test", backend="mlx", predictor=Predictor()) as client,
        pytest.raises(ValueError, match="client"),
    ):
        cls(**{"model_name": "test", "client": client, **kwargs})


@pytest.mark.parametrize(
    "system,machine,mlx_installed,expected",
    [
        ("darwin", "arm64", True, "mlx"),
        ("darwin", "arm64", False, "torch"),
        ("darwin", "x86_64", True, "torch"),
        ("linux", "aarch64", True, "torch"),
        ("win32", "AMD64", True, "torch"),
    ],
)
def test_auto_backend_is_local_and_platform_aware(
    monkeypatch, system, machine, mlx_installed, expected
):
    monkeypatch.setattr("kedi_laya.client.sys.platform", system)
    monkeypatch.setattr("kedi_laya.client.platform.machine", lambda: machine)
    monkeypatch.setattr(
        "kedi_laya.client.find_spec", lambda name: object() if mlx_installed else None
    )
    client = LayaClient("unloaded")
    assert client.backend == expected
    assert client.provider_name == "laya"
    assert client._predictor is None


@pytest.mark.parametrize("backend,module", [("mlx", "laya_mlx"), ("torch", "laya")])
def test_backend_load_failure_never_falls_back(monkeypatch, backend, module):
    requested = []

    def fail(name):
        requested.append(name)
        raise RuntimeError("checkpoint is incompatible")

    monkeypatch.setattr("kedi_laya.client.import_module", fail)
    with pytest.raises(RuntimeError, match="incompatible"):
        LayaClient("wrong/checkpoint", backend=backend).decide("state", {"x": Noul()})
    assert requested == [module]


@pytest.mark.parametrize(
    "answer",
    [
        None,
        {},
        {"type": "other"},
        {"type": "noul", "noul": "0.9"},
        {"type": "noul", "noul": True},
        {"type": "noul", "noul": float("nan")},
        {"type": "noul", "noul": 1.1},
    ],
)
def test_malformed_answers_fail_at_transport_boundary(answer):
    client = LayaClient(
        "test", predictor=SimpleNamespace(predict=lambda *args: {"answers": {"x": answer}})
    )
    with pytest.raises(DecisionResponseError, match="Invalid Laya response"):
        client.decide("state", {"x": Noul()})


@pytest.mark.parametrize("tokens", [-1, True, "4"])
def test_malformed_usage_is_not_reported_as_tokens(tokens):
    client = LayaClient(
        "test",
        predictor=SimpleNamespace(
            predict=lambda *args: {
                "answers": {"x": {"type": "noul", "noul": 0.9}},
                "usage": {"input_tokens": tokens},
            }
        ),
    )
    with pytest.raises(DecisionResponseError):
        client.decide("state", {"x": Noul()})
