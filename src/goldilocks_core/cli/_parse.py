"""Shared CLI parsing utilities."""

from __future__ import annotations

from typing import Any


def coerce_hint_value(value: str) -> Any:
    """Coerce a raw CLI string value to int, float, bool, or str."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value
