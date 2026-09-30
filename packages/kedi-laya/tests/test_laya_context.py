import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from kedi_laya import LayaClient
from kedi_laya._mlx_context import check_mlx_context

from kedi_decisions.contracts import BooleanQuestion as Noul


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
