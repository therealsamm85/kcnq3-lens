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


# ─── Schema v2 bucket helpers ─────────────────────────────────────────────────


def bucket_plv(plv: float | None) -> str | None:
    """Map a PLV (0..1) to one of schema.PLV_BUCKETS. None → None."""
    if plv is None:
        return None
    try:
        v = float(plv)
    except (TypeError, ValueError):
        return None
    if v != v or v < 0:  # NaN or negative
        return None
    # Lower-inclusive, upper-exclusive; last bucket open-ended
    if v < 0.1:
        return "<0.1"
    if v < 0.2:
        return "0.1-0.2"
    if v < 0.35:
        return "0.2-0.35"
    if v < 0.5:
        return "0.35-0.5"
    return ">0.5"


def bucket_phase_deg(deg: float | None) -> str | None:
    """Map a preferred phase in degrees (-180..180) to one of schema.PHASE_OCTANTS.

    Convention: lower-inclusive, upper-exclusive, except the last octant
    [135,180] which is upper-inclusive (closed on both ends to capture 180°).
    Each octant covers exactly 45°:
      [-180,-135), [-135,-90), [-90,-45), [-45,0),
      [0,45), [45,90), [90,135), [135,180]
    """
    if deg is None:
        return None
    try:
        v = float(deg)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    # Clamp to [-180, 180] to handle floating-point edge cases
    v = max(-180.0, min(180.0, v))
    octants = [
        (-180.0, -135.0, "[-180,-135)"),
        (-135.0, -90.0,  "[-135,-90)"),
        (-90.0,  -45.0,  "[-90,-45)"),
        (-45.0,    0.0,  "[-45,0)"),
        (  0.0,   45.0,  "[0,45)"),
        ( 45.0,   90.0,  "[45,90)"),
        ( 90.0,  135.0,  "[90,135)"),
        (135.0,  180.0,  "[135,180]"),
    ]
    for lo, hi, label in octants[:-1]:
        if lo <= v < hi:
            return label
    # Last octant is closed: [135, 180]
    return "[135,180]"


def bucket_coupled_events(n: int | None) -> str | None:
    """Map a count of coupled events to one of schema.COUPLED_EVENTS_BUCKETS.

    Convention: lower-inclusive, upper-exclusive (50-200 includes 50, excludes 200).
    """
    if n is None:
        return None
    try:
        v = int(n)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    if v < 10:
        return "<10"
    if v < 50:
        return "10-50"
    if v < 200:
        return "50-200"
    return ">200"


def bucket_sw_density(density: float | None) -> str | None:
    """Map slow-wave density (per minute) to one of schema.SW_DENSITY_BUCKETS."""
    if density is None:
        return None
    try:
        v = float(density)
    except (TypeError, ValueError):
        return None
    if v != v or v < 0:
        return None
    if v < 5:
        return "<5"
    if v < 15:
        return "5-15"
    if v < 30:
        return "15-30"
    if v < 50:
        return "30-50"
    return ">50"


def bucket_sw_ptp_uv(ptp: float | None) -> str | None:
    """Map slow-wave peak-to-peak amplitude (µV) to one of schema.SW_PTP_BUCKETS."""
    if ptp is None:
        return None
    try:
        v = float(ptp)
    except (TypeError, ValueError):
        return None
    if v != v or v < 0:
        return None
    if v < 75:
        return "<75"
    if v < 150:
        return "75-150"
    if v < 250:
        return "150-250"
    return ">250"


def bucket_hfo_rate(rate: float | None) -> str | None:
    """Map HFO rate (per minute) to one of schema.HFO_RATE_BUCKETS.

    The "0" bucket is exact zero (no ripples detected).
    """
    if rate is None:
        return None
    try:
        v = float(rate)
    except (TypeError, ValueError):
        return None
    if v != v or v < 0:
        return None
    if v == 0.0:
        return "0"
    if v < 1.0:
        return "<1"
    if v < 5.0:
        return "1-5"
    if v < 15.0:
        return "5-15"
    return ">15"


# ─── v0.13.3 — IED bucket helpers ────────────────────────────────────────────


def bucket_ied_rate(rate: float | None) -> str | None:
    """Map IED rate per minute to one of schema.IED_RATE_BUCKETS.

    The "0" bucket is exact zero.
    """
    if rate is None:
        return None
    try:
        v = float(rate)
    except (TypeError, ValueError):
        return None
    if v != v or v < 0:
        return None
    if v == 0.0:
        return "0"
    if v < 1.0:
        return "<1"
    if v < 5.0:
        return "1-5"
    if v < 15.0:
        return "5-15"
    if v < 50.0:
        return "15-50"
    return ">50"


def bucket_ied_agreement(pct: float | None) -> str | None:
    """Map agreement-with-morphology percent to schema.IED_AGREEMENT_BUCKETS."""
    if pct is None:
        return None
    try:
        v = float(pct)
    except (TypeError, ValueError):
        return None
    if v != v or v < 0 or v > 100.5:
        return None
    if v < 50.0:
        return "<50"
    if v < 75.0:
        return "50-75"
    if v < 90.0:
        return "75-90"
    return ">90"


def bucket_ied_rolandic(n: int | None) -> str | None:
    """Map count of likely-Rolandic flagged events to a coarse bucket."""
    if n is None:
        return None
    try:
        v = int(n)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    if v == 0:
        return "0"
    if v < 10:
        return "small"
    if v < 50:
        return "medium"
    return "large"


def bucket_ied_nrem_rate(rate: float | None) -> str | None:
    """Map IED rate during NREM (per minute) to schema.IED_NREM_RATE_BUCKETS.

    Clinically important for CSWS/ESES-spectrum evaluation.
    Same thresholds as bucket_ied_rate; exposed separately for independent
    NREM-vs-overall comparison in the registry.
    """
    if rate is None:
        return None
    try:
        v = float(rate)
    except (TypeError, ValueError):
        return None
    if v != v or v < 0:
        return None
    if v == 0.0:
        return "0"
    if v < 1.0:
        return "<1"
    if v < 5.0:
        return "1-5"
    if v < 15.0:
        return "5-15"
    if v < 50.0:
        return "15-50"
    return ">50"


# ─── v0.16.0 bucket helpers ───────────────────────────────────────────────────


def bucket_aperiodic_chi_n2(chi: float | None) -> str | None:
    """Map aperiodic exponent χ (N2 state) to schema.APERIODIC_CHI_N2_BUCKETS."""
    if chi is None:
        return None
    try:
        v = float(chi)
    except (TypeError, ValueError):
        return None
    if v != v or v < 0:  # NaN or negative
        return None
    if v < 1.5:
        return "<1.5"
    if v < 2.0:
        return "1.5-2.0"
    if v < 2.5:
        return "2.0-2.5"
    return ">2.5"


def bucket_pdr_asymmetry(asymmetry_interpretation: str | None) -> str | None:
    """Map asymmetry_interpretation string to schema.PDR_ASYMMETRY_BUCKETS.

    Mapping:
    - "symmetric" / "not_computed" → "symmetric"
    - "lh_dominant" → "lh_dominant"
    - "rh_dominant" → "rh_dominant"
    - "marked_asymmetric" → "marked"
    """
    if asymmetry_interpretation is None:
        return None
    _MAP = {
        "symmetric": "symmetric",
        "not_computed": "symmetric",
        "lh_dominant": "lh_dominant",
        "rh_dominant": "rh_dominant",
        "marked_asymmetric": "marked",
    }
    return _MAP.get(str(asymmetry_interpretation))


# Re-exports for convenience.
AGE_BUCKETS = _schema.AGE_BUCKETS
DURATION_BUCKETS = _schema.DURATION_BUCKETS
