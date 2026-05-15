"""Bucketing helpers — coarsen exact values into k-anonymity-friendly bins.

Exact age + exact duration are quasi-identifiers. We never store either
in a submission. These helpers map exact values to one of the allowed
buckets in `schema.AGE_BUCKETS` / `schema.DURATION_BUCKETS`.

Conventions
- Buckets are LOWER-INCLUSIVE, UPPER-EXCLUSIVE except the open-ended
  top bucket. E.g., "3-5" means 3.0 <= age < 5.0; "30+" means age >= 30.
- Edge case at exact upper bound: "3-5" includes 3.0 and 4.999, but
  5.0 belongs to "5-7". This avoids ambiguity at the boundary.
"""

from __future__ import annotations

from . import schema as _schema


def bucket_age_years(age_years: float | int | None) -> str | None:
    """Map an age in years to one of `schema.AGE_BUCKETS`. None → None."""
    if age_years is None:
        return None
    try:
        a = float(age_years)
    except (TypeError, ValueError):
        return None
    if a < 0 or a != a:  # negative or NaN
        return None
    # Ordered list of (lower, upper, label). Upper-exclusive except last.
    bins = [
        (0.0, 1.0, "0-1"),
        (1.0, 2.0, "1-2"),
        (2.0, 3.0, "2-3"),
        (3.0, 5.0, "3-5"),
        (5.0, 7.0, "5-7"),
        (7.0, 10.0, "7-10"),
        (10.0, 13.0, "10-13"),
        (13.0, 18.0, "13-18"),
        (18.0, 30.0, "18-30"),
    ]
    for lo, hi, label in bins:
        if lo <= a < hi:
            return label
    return "30+"


def bucket_duration_hours(hours: float | int | None) -> str | None:
    """Map a recording duration in hours to `schema.DURATION_BUCKETS`."""
    if hours is None:
        return None
    try:
        h = float(hours)
    except (TypeError, ValueError):
        return None
    if h < 0 or h != h:
        return None
    bins = [
        (0.0, 1.0, "<1"),
        (1.0, 4.0, "1-4"),
        (4.0, 12.0, "4-12"),
        (12.0, 24.0, "12-24"),
        (24.0, 48.0, "24-48"),
    ]
    for lo, hi, label in bins:
        if lo <= h < hi:
            return label
    return "48+"


def assert_valid_bucket(value: str, allowed: tuple[str, ...]) -> str:
    """Raise ValueError if `value` is not in `allowed`. Returns value."""
    if value not in allowed:
        raise ValueError(
            f"bucket value {value!r} not in allowed set {allowed!r}"
        )
    return value


# Re-exports for convenience.
AGE_BUCKETS = _schema.AGE_BUCKETS
DURATION_BUCKETS = _schema.DURATION_BUCKETS
