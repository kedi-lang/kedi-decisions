"""Laya decisions, without TypeSafe or agent-framework import dependencies."""

from typing import TYPE_CHECKING

from kedi_decisions import BooleanCriteria, ChoiceCriteria, Probability, Rubric

from .client import AsyncLayaClient, LayaClient, LayaPredictor

if TYPE_CHECKING:
    from .integrations.langchain import LayaChatModel
    from .integrations.pydantic import LayaModel

__all__ = [
    "AsyncLayaClient",
    "LayaClient",
    "LayaPredictor",
    "BooleanCriteria",
    "ChoiceCriteria",
    "Probability",
    "Rubric",
    "LayaModel",
    "LayaChatModel",
]


def __getattr__(name: str):
    if name == "LayaModel":
        from .integrations.pydantic import LayaModel

        return LayaModel
    if name == "LayaChatModel":
        from .integrations.langchain import LayaChatModel

        return LayaChatModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
