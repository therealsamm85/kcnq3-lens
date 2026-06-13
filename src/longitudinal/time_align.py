"""Time alignment for longitudinal series — parse dates, split before/after an
intervention, and match the nearest observation in time.

EEG recordings and parent diary entries almost never fall on the same calendar
day, so both the treatment-response dashboard (before/after an intervention)
and the word-tracker correlation (pair each EEG with the nearest word count)
need a small, well-tested time layer. It lives here once so both features use
identical, audited date logic rather than two slightly different inline parsers.
"""

from __future__ import annotations

import datetime
from typing import Optional


def parse_date(s: str | None) -> Optional[datetime.date]:
    """Parse a 'YYYY-MM-DD' string to a date, or None if absent/malformed.

    Tolerant of a trailing time component ('YYYY-MM-DDTHH:MM:SS' or a space
    separator) since diary/storage timestamps sometimes carry one.
    """
    if not s:
        return None
    text = str(s).strip()
    if not text:
        return None
    head = text.replace("T", " ").split(" ", 1)[0]
    try:
        return datetime.date.fromisoformat(head)
    except (ValueError, TypeError):
        return None


def days_between(a: datetime.date, b: datetime.date) -> int:
    """Signed day count (b − a)."""
    return (b - a).days


def split_before_after(
    series: list[tuple[datetime.date, float]],
    pivot: datetime.date,
) -> tuple[Optional[tuple[datetime.date, float]],
           Optional[tuple[datetime.date, float]]]:
    """Split a dated series around ``pivot``.

    Returns (last_before, first_on_or_after) where each is a (date, value) pair
    or None. ``pivot`` day itself counts as "after" — a recording taken on the
    day a medication changed reflects the new regimen's starting point.
    The input need not be sorted.
    """
    before: Optional[tuple[datetime.date, float]] = None
    after: Optional[tuple[datetime.date, float]] = None
    for d, v in sorted(series, key=lambda t: t[0]):
        if d < pivot:
            before = (d, v)            # keep advancing → last one before pivot
        elif after is None:
            after = (d, v)             # first one on/after pivot
    return before, after


def nearest_within(
    series: list[tuple[datetime.date, float]],
    target: datetime.date,
    max_days: int,
) -> Optional[tuple[datetime.date, float, int]]:
    """Return the (date, value, signed_gap_days) in ``series`` closest in time
    to ``target``, or None if the closest is farther than ``max_days``.

    signed_gap_days = series_date − target (negative = the match precedes the
    target). Ties in absolute distance resolve to the earlier date.
    """
    best: Optional[tuple[datetime.date, float, int]] = None
    best_abs = max_days + 1
    for d, v in series:
        gap = days_between(target, d)
        if abs(gap) <= max_days and abs(gap) < best_abs:
            best = (d, v, gap)
            best_abs = abs(gap)
    return best
