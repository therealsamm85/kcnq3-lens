"""D1 — Age-normative qEEG z-scores.  [BUILD engine; norms pluggable]

The tool reports band power / PDR / delta-alpha ratio / 1/f exponent as absolute
values. For a developing brain the same number is normal at 4 and pathological at
10, so this converts the existing spectral metrics into age-referenced z-scores
("PDR z = −4 for a 5-year-old").

BUILD: the z-score engine (nearest-age-bin lookup → z = (x − mean)/sd) is simple
and built here. The NORM DATA is the hard, unsolved part: qEEGt's CHBMP norms are
MATLAB data (not portable) and NeuroGuide is proprietary. So norms are PLUGGABLE,
and the bundled ``NORMS_PLACEHOLDER`` is an APPROXIMATE, explicitly UNVERIFIED
developmental table for demonstration only. Every z it produces is stamped
norm_verified=False and the renderer leads with a do-not-use-clinically banner.
Replace NORMS_PLACEHOLDER with sourced, age-appropriate pediatric norms before
any clinical use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..longitudinal.trends import METRICS

_LABEL_TO_PATH = {label: path for label, path in METRICS}

# ⚠ PLACEHOLDER, UNVERIFIED — approximate developmental values for DEMONSTRATION
# of the engine only. NOT sourced from a validated pediatric normative database.
# Each entry: metric → {source, verified, bins:[(age_lo, age_hi, mean, sd)]}.
NORMS_PLACEHOLDER: dict[str, dict] = {
    "pdr_hz": {
        "source": "PLACEHOLDER (unverified developmental approximation)",
        "verified": False,
        "bins": [(0, 2, 5.0, 1.0), (2, 4, 6.5, 1.0), (4, 6, 8.0, 1.0),
                 (6, 9, 9.0, 1.0), (9, 18, 9.5, 1.0)],
    },
    "delta_alpha_ratio": {
        "source": "PLACEHOLDER (unverified developmental approximation)",
        "verified": False,
        "bins": [(0, 2, 4.0, 1.5), (2, 4, 2.5, 1.0), (4, 6, 1.8, 0.8),
                 (6, 9, 1.2, 0.6), (9, 18, 0.9, 0.5)],
    },
}


@dataclass
class NormPoint:
    metric: str
    value: float
    z: float | None
    age_years: float
    norm_mean: float | None
    norm_sd: float | None
    norm_source: str
    norm_verified: bool
    note: str = ""


@dataclass
class NormativeResult:
    age_years: float
    any_verified: bool
    points: list[NormPoint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _extract(findings: dict, metric: str):
    path = _LABEL_TO_PATH.get(metric)
    if not path:
        return None
    cur = findings
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _lookup_bin(bins, age):
    for lo, hi, mean, sd in bins:
        if lo <= age < hi:
            return mean, sd
    return None


def compute_normative_z(
    findings: dict,
    age_years: float,
    norms: dict | None = None,
) -> NormativeResult:
    """Convert spectral findings into age-referenced z-scores.

    A z is emitted only when a norm entry covers both the metric and the
    patient's age. Bundled norms are UNVERIFIED placeholders (see module docs).
    """
    norms = norms if norms is not None else NORMS_PLACEHOLDER
    points: list[NormPoint] = []
    any_verified = False
    for metric, spec in norms.items():
        value = _extract(findings, metric)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        verified = bool(spec.get("verified", False))
        source = str(spec.get("source", "unknown"))
        found = _lookup_bin(spec.get("bins", []), age_years)
        if found is None:
            points.append(NormPoint(
                metric=metric, value=round(value, 3), z=None, age_years=age_years,
                norm_mean=None, norm_sd=None, norm_source=source,
                norm_verified=verified,
                note=f"age {age_years:.1f}y outside norm coverage — no z",
            ))
            continue
        mean, sd = found
        z = (value - mean) / sd if sd and sd > 0 else None
        any_verified = any_verified or verified
        points.append(NormPoint(
            metric=metric, value=round(value, 3),
            z=round(float(z), 2) if z is not None else None, age_years=age_years,
            norm_mean=mean, norm_sd=sd, norm_source=source, norm_verified=verified,
        ))

    notes: list[str] = []
    if not points:
        notes.append("no normed metrics present in findings.")
    if points and not any_verified:
        notes.append("ALL norms used are UNVERIFIED placeholders — z-scores are "
                     "illustrative only and MUST NOT be used clinically until "
                     "replaced with a sourced pediatric normative database.")
    return NormativeResult(age_years=age_years, any_verified=any_verified,
                           points=points, notes=notes)


def summarize_normative(result: NormativeResult) -> dict:
    return {
        "age_years": result.age_years,
        "any_verified": result.any_verified,
        "points": [
            {
                "metric": p.metric, "value": p.value, "z": p.z,
                "norm_mean": p.norm_mean, "norm_sd": p.norm_sd,
                "norm_source": p.norm_source, "norm_verified": p.norm_verified,
                "note": p.note,
            }
            for p in result.points
        ],
        "notes": result.notes,
    }


def render_normative_md(result: NormativeResult) -> str:
    lines = ["# Age-normative qEEG z-scores", ""]
    if not result.any_verified and result.points:
        lines.append("> ⚠️ **UNVERIFIED placeholder norms — illustrative only, "
                     "not for clinical use.** Replace with a sourced pediatric "
                     "normative database.")
        lines.append("")
    lines.append(f"_Age {result.age_years:.1f} years._")
    lines.append("")
    if not result.points:
        for n in result.notes:
            lines.append(f"> {n}")
        return "\n".join(lines)
    lines.append("| Metric | Value | Norm (mean±sd) | z | Source verified |")
    lines.append("|---|---|---|---|---|")
    for p in result.points:
        norm = (f"{p.norm_mean:g}±{p.norm_sd:g}" if p.norm_mean is not None else "—")
        z = "—" if p.z is None else f"{p.z:+.1f}"
        lines.append(f"| {p.metric} | {p.value:g} | {norm} | {z} | "
                     f"{'yes' if p.norm_verified else 'NO'} |")
    return "\n".join(lines)
