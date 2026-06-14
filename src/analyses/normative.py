"""D1 — Age-normative qEEG z-scores.  [BUILD engine; norms pluggable]

The tool reports band power / PDR / delta-alpha ratio / 1/f exponent as absolute
values. For a developing brain the same number is normal at 4 and pathological at
10, so this converts the existing spectral metrics into age-referenced z-scores
("PDR z = −3.1 for a 5-year-old").

BUILD: the z-score engine (z = (x − mean(age)) / sd(age), with age-regression or
nearest-age-bin lookup) is simple and built here. The NORM DATA is the hard part:
qEEGt's CHBMP norms are MATLAB data, not portable, and NeuroGuide is proprietary.
So norms are PLUGGABLE via a table loader, and the bundled defaults are clearly
flagged UNVERIFIED placeholders until real pediatric norms are sourced — a z-score
is only emitted when a norm entry actually covers the patient's age + metric.

SCAFFOLD — implemented in wave D1.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NormPoint:
    metric: str
    value: float
    z: float | None
    age_years: float
    norm_source: str
    norm_verified: bool
    note: str = ""


@dataclass
class NormativeResult:
    age_years: float
    norm_source: str
    points: list[NormPoint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def compute_normative_z(
    findings: dict,
    age_years: float,
    norms: dict | None = None,
) -> NormativeResult:
    """Convert spectral findings to age-referenced z-scores. SCAFFOLD — wave D1."""
    raise NotImplementedError("scaffold — implemented in wave D1")


def summarize_normative(result: NormativeResult) -> dict:
    raise NotImplementedError("scaffold — implemented in wave D1")
