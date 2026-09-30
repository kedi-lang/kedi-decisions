"""Typed tool selection without generating arguments or executing tools."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .evaluation import EvaluationResult
from .metadata import SCHEMA_KEY


class ToolCallProposed(RuntimeError):  # noqa: N818
    def __init__(self, tool_name: str, probability: float) -> None:
        self.tool_name = tool_name
        self.probability = probability
        super().__init__(
            f"Decision model proposed {tool_name!r} (probability {probability:.3f}); "
            "an explicit argument-producing handler is required"
        )


@dataclass(frozen=True)
class RoutingRequest:
    schema: Mapping[str, Any]
    route_key: str | None
    tools: tuple[dict[str, Any], ...]


def prepare_routing(
    schema: Mapping[str, Any],
    tools: Sequence[dict[str, Any]],
    completed: set[str],
    *,
    allow_final: bool = True,
) -> RoutingRequest:
    available = tuple(tool for tool in tools if tool["name"] not in completed)
    if not available:
        return RoutingRequest(schema, None, ())
    properties = dict(schema.get("properties", {}))
    # Preserve existing serialized route keys for TypeSafe callers.
    route_key = "__typesafe_route"
    while route_key in properties:
        route_key += "_"
    final = "__final_output" if allow_final else "__no_action"
    while final in {tool["name"] for tool in available}:
        final += "_"
    criteria = (
        {
            final: schema.get("description")
            or "Return the requested structured assessment using the available evidence."
        }
        if allow_final
        else {final: "None of the available actions is appropriate for this request."}
    )
    criteria.update({tool["name"]: tool.get("description") or tool["name"] for tool in available})
    properties[route_key] = {
        "type": "string",
        "enum": list(criteria),
        "description": "Which action does this request call for?",
        SCHEMA_KEY: {"kind": "choice", "criteria": criteria},
    }
    return RoutingRequest(
        {**schema, "properties": properties, "required": [*schema.get("required", []), route_key]},
        route_key,
        available,
    )


def resolve_routing(
    result: EvaluationResult,
    request: RoutingRequest,
    threshold: float,
) -> tuple[EvaluationResult, dict[str, Any] | None]:
    key = request.route_key
    if key is None:
        return result, None
    selected = result.values[key]
    probability = result.metadata["answers"][key]["probabilities"][selected]
    normalized = replace(
        result, values={name: value for name, value in result.values.items() if name != key}
    )
    tool = next((tool for tool in request.tools if tool["name"] == selected), None)
    if tool is None or probability < threshold:
        return normalized, None
    if tool.get("parameters", {}).get("properties"):
        raise ToolCallProposed(selected, probability)
    return normalized, tool
