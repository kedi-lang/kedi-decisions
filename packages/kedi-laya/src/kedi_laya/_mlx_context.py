"""Reject silent token truncation in the pinned optional MLX runtime."""

from __future__ import annotations

import json
from collections.abc import Sequence
from importlib import import_module
from typing import Any

from kedi_decisions.contracts import DecisionMedia

MEDIA_BYTES_PER_TOKEN = 1024
"""Conservative lower bound for media tokens: the MLX engine spends at least one
context token per KiB of media (audio runs about 50 tokens/s at 32 KiB/s; a
336 px image is about 100 KiB for 256+ tokens). The runtime exposes no
media-token count, so the check rejects when even this floor overflows the
window; the true media load is at least this large."""


def check_mlx_context(
    agent: Any,
    state: Any,
    questions: dict[str, Any],
    media: Sequence[DecisionMedia] = (),
) -> None:
    """Check the prompt exactly as `predict()` receives it.

    `state` and `questions` are the values passed to `predict()`: with media
    the state is `None` and the state text is already folded into each
    question's instructions, so it must not be counted a second time. Media
    bytes count against the window through `MEDIA_BYTES_PER_TOKEN`.
    """
    runtime = import_module("laya_mlx.common")

    tokenizer = agent.tok
    state_tokens: list[Any] = (
        tokenizer(
            runtime.serialize_state(state).replace(tokenizer.mask_token, " "),
            add_special_tokens=False,
        )["input_ids"]
        if state is not None
        else []
    )
    media_tokens = sum(max(1, len(item.data) // MEDIA_BYTES_PER_TOKEN) for item in media)
    for key, question in questions.items():
        instructions = question["instructions"]
        internal = {
            "t": question["type"],
            "ins": instructions if isinstance(instructions, str) else json.dumps(instructions),
            "crit": question.get("criteria"),
        }
        for option in runtime.render_options(internal):
            tokens = tokenizer(
                " " + option.replace(tokenizer.mask_token, " "), add_special_tokens=False
            )["input_ids"]
            if len(tokens) > 48:
                raise ValueError(f"Laya question {key!r} exceeds the 48-token option limit")
        prefix, _ = runtime.build_prefix(tokenizer, internal, agent.cfg.get("head_max_len", 192))
        full_prefix, _ = runtime.build_prefix(tokenizer, internal, 1_000_000)
        if prefix != full_prefix:
            raise ValueError(f"Laya question {key!r} exceeds the question token budget")
        maximum = agent.cfg.get("max_len", 512)
        if len(prefix) + len(state_tokens) + media_tokens + 1 > maximum:
            raise ValueError(
                f"Laya question {key!r} exceeds the {maximum}-token context window; "
                "shorten the input instead of silently truncating it"
            )
