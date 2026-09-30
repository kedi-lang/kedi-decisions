"""Existing exception names remain aliases of the shared decision exceptions."""

from kedi_decisions.errors import (
    DecisionExtractionError as TypeSafeExtractionError,
)
from kedi_decisions.errors import (
    DecisionIntegrationError as TypeSafeIntegrationError,
)
from kedi_decisions.errors import (
    DecisionResponseError as TypeSafeResponseError,
)
from kedi_decisions.errors import (
    DecisionSchemaError as TypeSafeSchemaError,
)

__all__ = [
    "TypeSafeExtractionError",
    "TypeSafeIntegrationError",
    "TypeSafeResponseError",
    "TypeSafeSchemaError",
]
