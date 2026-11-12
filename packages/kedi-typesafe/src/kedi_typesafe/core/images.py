"""Images in Jev requests: TypeSafe receives them as assets referenced from the state."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

from typesafe_sdk import Image

_LABEL = re.compile(r"<image (\d+)>")

IMAGE_HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}
"""TypeSafe image input is experimental; requests that carry images opt in with this header."""


def with_image_assets(state: Any, images: Sequence[bytes]) -> tuple[Any, dict[str, Image]]:
    """Replace each `<image N>` label in the state with the reference of the Nth image."""

    assets = [Image(data) for data in images]
    if not assets:
        return state, {}

    def reference(match: re.Match[str]) -> str:
        index = int(match[1])
        return str(assets[index - 1]) if 0 < index <= len(assets) else match[0]

    def replace(value: object) -> object:
        if isinstance(value, str):
            return _LABEL.sub(reference, value)
        if isinstance(value, Mapping):
            mapping = cast("Mapping[str, object]", value)
            return {key: replace(item) for key, item in mapping.items()}
        if isinstance(value, Sequence):
            return [replace(item) for item in cast("Sequence[object]", value)]
        return value

    return replace(state), {str(asset): asset for asset in assets}
