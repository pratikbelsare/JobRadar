"""Safe structured diagnostics for AWS queue processing failures."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

LOGGER = logging.getLogger("job_intelligence.aws")
_MAX_ERROR_LENGTH = 500


def log_processing_failure(
    *,
    event: str,
    message_id: str,
    stage: str,
    error: Exception,
    identifiers: Mapping[str, Any] | None = None,
) -> None:
    """Emit one CloudWatch-friendly JSON error without logging payload contents."""

    fields: dict[str, Any] = {
        "event": event,
        "message_id": message_id,
        "stage": stage,
        "exception_type": type(error).__name__,
        "error_message": _concise_message(str(error)),
    }
    if identifiers:
        fields.update(
            {
                key: str(value)
                for key, value in identifiers.items()
                if value is not None and value != ""
            }
        )
    LOGGER.error(json.dumps(fields, sort_keys=True), exc_info=True)


def _concise_message(message: str) -> str:
    compact = " ".join(message.split())
    if len(compact) <= _MAX_ERROR_LENGTH:
        return compact
    return compact[: _MAX_ERROR_LENGTH - 3] + "..."


__all__ = ["LOGGER", "log_processing_failure"]
