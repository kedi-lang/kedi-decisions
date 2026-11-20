import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from kedi_laya import LayaClient
from kedi_laya._mlx_context import check_mlx_context

from kedi_decisions.contracts import BooleanQuestion as Noul
from kedi_decisions.contracts import DecisionMedia


class Tokenizer:
    mask_token = "[MASK]"

    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": text.split()}


@pytest.fixture
def local_runtime(monkeypatch):
    common = ModuleType("laya_mlx.common")
    common.serialize_state = lambda state: state if isinstance(state, str) else json.dumps(state)
    common.render_options = lambda q: list((q["crit"] or {}).values())
    common.build_prefix = lambda tok, q, budget: (q["ins"].split()[:budget], [])
    monkeypatch.setitem(sys.modules, "laya_mlx.common", common)
    return SimpleNamespace(tok=Tokenizer(), cfg={"max_len": 12, "head_max_len": 5})


def test_mlx_rejects_state_truncation_at_exact_boundary(local_runtime):
    questions = {"x": {"type": "choice", "instructions": "one two", "criteria": {"a": "a"}}}
    check_mlx_context(local_runtime, "word " * 9, questions)
    with pytest.raises(ValueError, match="context window"):
        check_mlx_context(local_runtime, "word " * 10, questions)


def test_mlx_rejects_question_and_option_truncation(local_runtime):
    questions = {"x": {"type": "choice", "instructions": "word " * 6, "criteria": {"a": "a"}}}
    with pytest.raises(ValueError, match="question token budget"):
        check_mlx_context(local_runtime, "state", questions)
    questions["x"]["instructions"] = {"intent": "select"}
    questions["x"]["criteria"] = {"a": "word " * 49}
    with pytest.raises(ValueError, match="option limit"):
        check_mlx_context(local_runtime, "state", questions)


def test_mlx_uses_runtime_default_limits_and_json_instructions(local_runtime):
    local_runtime.cfg = {}
    check_mlx_context(
        local_runtime,
        {"key": "value"},
        {"x": {"type": "choice", "instructions": {"intent": "select"}, "criteria": {"a": "a"}}},
    )


def test_owned_mlx_client_checks_context_before_prediction(local_runtime, monkeypatch):
    calls = []
    local_runtime.predict = lambda state, questions: (
        calls.append(state) or {"answers": {"x": {"type": "noul", "noul": 0.9}}}
    )
    monkeypatch.setattr(
        "kedi_laya.client.import_module",
        lambda name: SimpleNamespace(load=lambda *args, **kwargs: local_runtime),
    )
    client = LayaClient("test", backend="mlx")
    response = client.decide("valid", {"x": Noul(instructions="yes", criteria={})})
    assert response.usage.input_tokens is None
    assert calls == ["valid"]


def _mlx_engine(monkeypatch, local_runtime):
    """A fake MLX predictor behind the lazy loader, recording predict() calls."""
    calls = []

    def predict(state, questions, *, image=None, audio=None, video=None):
        calls.append({"state": state, "questions": questions, "audio": audio, "image": image})
        return {"answers": {"x": {"type": "noul", "noul": 0.9}}}

    local_runtime.predict = predict
    monkeypatch.setattr(
        "kedi_laya.client.import_module",
        lambda name: SimpleNamespace(load=lambda *args, **kwargs: local_runtime),
    )
    return calls


def test_mlx_check_measures_context_augmented_prompt_with_media(local_runtime, monkeypatch):
    """With media, the state is folded into each question's instructions, so a
    long state must trip the question budget even when the raw instructions are
    short (audit #215 D6)."""
    calls = _mlx_engine(monkeypatch, local_runtime)
    client = LayaClient("test", backend="mlx")
    audio = DecisionMedia(kind="audio", data=b"x" * 100, media_type="audio/wav")
    with pytest.raises(ValueError, match="question token budget"):
        client.decide(
            "word " * 9, {"x": Noul(instructions="one two", criteria={})}, media=[audio]
        )
    assert calls == []


def test_mlx_check_rejects_large_media(local_runtime, monkeypatch):
    """A 5 MB audio clip exceeds the context window on its media bytes alone
    (audit #215 D6)."""
    calls = _mlx_engine(monkeypatch, local_runtime)
    client = LayaClient("test", backend="mlx")
    big_audio = DecisionMedia(
        kind="audio", data=b"\xff" * (5 * 1024 * 1024), media_type="audio/wav"
    )
    with pytest.raises(ValueError, match="context window"):
        client.decide("hi", {"x": Noul(instructions="Is it a cat?", criteria={})}, media=[big_audio])
    assert calls == []


def test_mlx_context_counts_media_bytes(local_runtime):
    """Media bytes count against the window; the check also accepts small media
    (audit #215 D6)."""
    questions = {"x": {"type": "choice", "instructions": "one", "criteria": {"a": "a"}}}
    small = [DecisionMedia(kind="audio", data=b"x" * 100, media_type="audio/wav")]
    check_mlx_context(local_runtime, None, questions, media=small)
    big = [DecisionMedia(kind="audio", data=b"x" * (20 * 1024), media_type="audio/wav")]
    with pytest.raises(ValueError, match="context window"):
        check_mlx_context(local_runtime, None, questions, media=big)
