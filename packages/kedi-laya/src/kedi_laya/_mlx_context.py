"""Reject silent token truncation in the pinned optional MLX runtime."""

from __future__ import annotations

import json
from importlib import import_module
from typing import Any


def check_mlx_context(agent: Any, state: Any, questions: dict[str, Any]) -> None:
    runtime = import_module("laya_mlx.common")

    tokenizer = agent.tok
    state_tokens = tokenizer(
        runtime.serialize_state(state).replace(tokenizer.mask_token, " "), add_special_tokens=False
    )["input_ids"]
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
        if len(prefix) + len(state_tokens) + 1 > maximum:
            raise ValueError(
                f"Laya question {key!r} exceeds the {maximum}-token context window; "
                "shorten the input instead of silently truncating it"
            )
