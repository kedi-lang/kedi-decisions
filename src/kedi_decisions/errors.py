from __future__ import annotations


class DecisionIntegrationError(Exception):
    """Base exception for local Decision integration failures."""


class DecisionSchemaError(DecisionIntegrationError, ValueError):
    """Raised when an output schema cannot be represented by Decision questions."""


class DecisionExtractionError(DecisionIntegrationError, ValueError):
    """Raised when constrained text extraction cannot produce valid candidates."""


class DecisionResponseError(DecisionIntegrationError, RuntimeError):
    """Raised when a Decision response violates the requested question contract."""


__all__ = [
    "DecisionExtractionError",
    "DecisionIntegrationError",
    "DecisionResponseError",
    "DecisionSchemaError",
]
