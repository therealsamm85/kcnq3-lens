"""Sanitization helpers for serializable findings output.

Ensures every value produced by `summarize_*()` functions is:
- A native Python type (no numpy scalars leaking)
- JSON-serializable (no NaN / Inf)
- Reasonable when the underlying computation degenerated (e.g. all-zero
  channels → NaN kurtosis → 0)

All `summarize_*` functions in this codebase should run their numeric
outputs through `safe_float()` / `safe_int()` to guarantee downstream
serialization (JSON, PDF, LLM) doesn't choke on bad floats.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def safe_float(x: Any, default: float = 0.0, ndigits: int | None = None) -> float:
    """Coerce to a finite Python float. Returns `default` if NaN/Inf/None/error.

    If `ndigits` is given, rounds to that number of decimals.
    """
    try:
        if x is None:
            return default
        if isinstance(x, (np.floating, np.integer)):
            x = float(x)
        elif isinstance(x, bool):
            x = float(int(x))
        else:
            x = float(x)
        if not math.isfinite(x):
            return default
        if ndigits is not None:
            return round(x, ndigits)
        return x
    except (TypeError, ValueError):
        return default


def safe_int(x: Any, default: int = 0) -> int:
    """Coerce to a Python int. Returns `default` on failure."""
    try:
        if x is None:
            return default
        if isinstance(x, (np.integer, np.floating, bool)):
            x = int(x)
        else:
            x = int(x)
        return x
    except (TypeError, ValueError):
        return default


def safe_round_dict(d: dict, ndigits: int = 2, default: float = 0.0) -> dict:
    """Recursively sanitize a dict for JSON output.

    Returns a new dict where every numeric leaf is a finite Python float/int,
    every numpy scalar is unboxed, and every NaN/Inf is replaced with `default`.
    Strings, bools, None, and nested structures are preserved.
    """
    out: dict = {}
    for k, v in d.items():
        out[k] = _sanitize_value(v, ndigits, default)
    return out


def _sanitize_value(v: Any, ndigits: int, default: float) -> Any:
    if isinstance(v, dict):
        return safe_round_dict(v, ndigits, default)
    if isinstance(v, list):
        return [_sanitize_value(item, ndigits, default) for item in v]
    if isinstance(v, tuple):
        return [_sanitize_value(item, ndigits, default) for item in v]
    if isinstance(v, (np.floating,)):
        return safe_float(v, default, ndigits)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, float):
        return safe_float(v, default, ndigits)
    return v
