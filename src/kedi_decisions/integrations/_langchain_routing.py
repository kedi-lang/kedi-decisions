from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from ..evaluation import EvaluationResult
from ..routing import RoutingRequest, ToolCallProposed
from ..routing import prepare_routing as prepare_decision_routing
from ..routing import resolve_routing as resolve_decision_routing


def prepare_routing(
    schema: Mapping[str, Any], tools: Sequence[dict[str, Any]], messages: list[BaseMessage]
) -> RoutingRequest:
    completed: set[str] = set()
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage) and message.name:
            completed.add(message.name)
    return prepare_decision_routing(schema, tools, completed)


def resolve_routing(
    result: EvaluationResult, request: RoutingRequest, threshold: float
) -> tuple[EvaluationResult, AIMessage | None]:
    normalized, tool = resolve_decision_routing(result, request, threshold)
    if tool is None:
        return normalized, None
    return normalized, AIMessage(
        content="",
        tool_calls=[{"name": tool["name"], "args": {}, "id": str(uuid4()), "type": "tool_call"}],
    )


__all__ = ["RoutingRequest", "ToolCallProposed", "prepare_routing", "resolve_routing"]
