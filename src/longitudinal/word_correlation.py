"""Word-tracker correlation — does vocabulary growth track the EEG biomarkers?

Why this exists
---------------
The single question this family lives with: is the reference patient's brain getting quiet
enough to learn language? The development diary records a vocabulary count over
time; the stored recordings record the spike burden and maturation markers. This
module pairs them in time and asks whether vocabulary moves *with* (or against)
each biomarker, in the clinically-expected direction (more spikes → fewer words,
faster PDR → more words).

The honesty problem, stated up front
------------------------------------
A family will have a handful of EEGs, not hundreds. A correlation on 5 points is
a *picture*, never a proof: at n = 5 even a perfect rank correlation is not
statistically significant (p ≈ 0.08), and the small-sample p-value is itself
unreliable. So this module:

* refuses to report a coefficient below MIN_PAIRS_FOR_RHO paired points,
* omits the p-value entirely below MIN_PAIRS_FOR_SIGNIFICANCE points (where it
  would be a misleadingly tiny number, not evidence),
* only ever calls a relationship "statistically conclusive" at or above that
  point count with p < 0.05,
* states the expected direction so a matching sign is read as "consistent with
  the hypothesis", not "demonstrated".

It is a hypothesis-generating lens for the family and clinician — explicitly not
a significance test that a tiny n cannot support.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.stats import spearmanr

from .storage import StoredEntry
from .diary import DiaryEntry
from .trends import get_metric_series
from .time_align import parse_date, nearest_within
from .metric_polarity import polarity_of, CONFOUNDED_BY_MATURATION

# Below this many paired points, report no coefficient at all.
MIN_PAIRS_FOR_RHO = 4
# Below this many, omit the p-value and never call a result conclusive — the
# small-sample p (especially near |rho|=1) is unreliable and reads as false proof.
MIN_PAIRS_FOR_SIGNIFICANCE = 8

# Cognition-relevant biomarkers worth correlating against vocabulary.
WORD_CORRELATION_METRICS: list[str] = [
    "spike_rate_per_min",
    "pdr_hz",
    "swi_n3_pct",
    "delta_alpha_ratio",
    "spindle_density_per_min",
]


@dataclass
class PairedPoint:
    metric_date: str
    metric_value: float
    word_date: str
    word_count: int
    gap_days: int


@dataclass
class MetricWordCorr:
    metric: str
    n_pairs: int
    spearman_rho: float | None
    p_value: float | None
    expected_sign: int               # +1 / -1 / 0 (0 = no clinical expectation)
    observed_sign: int | None        # sign(rho) or None
    matches_expected: bool | None
    statistically_conclusive: bool
    maturation_confounded: bool
    interpretation: str
    pairs: list[PairedPoint] = field(default_factory=list)


@dataclass
class WordCorrelation:
    metric_correlations: list[MetricWordCorr] = field(default_factory=list)
    n_word_observations: int = 0
    n_recordings: int = 0
    notes: list[str] = field(default_factory=list)


def compute_word_correlation(
    entries: list[StoredEntry],
    diary: list[DiaryEntry],
    metrics: list[str] | None = None,
    max_pair_gap_days: int = 45,
) -> WordCorrelation:
    """Correlate diary vocabulary counts against each EEG biomarker over time.

    Each EEG recording is paired with the nearest diary word-count within
    ``max_pair_gap_days``; the Spearman rank correlation is taken over those
    pairs. See the module docstring for the n-gating policy.
    """
    metrics = metrics or WORD_CORRELATION_METRICS
    result = WordCorrelation(n_recordings=len(entries))

    # Build the vocabulary series from the diary.
    word_series: list[tuple] = []  # (date, word_count)
    for e in diary:
        d = parse_date(e.date)
        if d is not None and e.word_count is not None:
            word_series.append((d, int(e.word_count)))
    word_series.sort(key=lambda t: t[0])
    result.n_word_observations = len(word_series)

    if not word_series:
        result.notes.append(
            "No vocabulary counts in the diary — add diary entries with a "
            "word_count to populate the language-correlation view."
        )
        return result

    for metric in metrics:
        dates, vals = get_metric_series(entries, metric)
        metric_series = []
        for ds, v in zip(dates, vals):
            d = parse_date(ds)
            if d is not None:
                metric_series.append((d, float(v)))

        pairs: list[PairedPoint] = []
        for md, mv in metric_series:
            match = nearest_within(word_series, md, max_pair_gap_days)
            if match is None:
                continue
            wd, wc, gap = match
            pairs.append(PairedPoint(
                metric_date=md.isoformat(), metric_value=round(mv, 3),
                word_date=wd.isoformat(), word_count=int(wc),
                gap_days=gap,
            ))

        result.metric_correlations.append(
            _assess_metric(metric, pairs)
        )

    result.notes.append(
        f"Each biomarker is paired with the nearest diary word-count within "
        f"±{max_pair_gap_days} days. Correlations here are hypothesis-generating: "
        f"with a handful of recordings they describe a pattern, not a proven "
        f"link, and cannot establish causation. Confirm any signal clinically."
    )
    return result


def _assess_metric(metric: str, pairs: list[PairedPoint]) -> MetricWordCorr:
    """Compute the gated Spearman assessment for one metric's paired points."""
    expected = polarity_of(metric)
    confounded = metric in CONFOUNDED_BY_MATURATION
    n = len(pairs)

    base = dict(
        metric=metric, n_pairs=n, spearman_rho=None, p_value=None,
        expected_sign=expected, observed_sign=None, matches_expected=None,
        statistically_conclusive=False, maturation_confounded=confounded,
        pairs=pairs,
    )

    if n < MIN_PAIRS_FOR_RHO:
        return MetricWordCorr(
            **base,
            interpretation=(
                f"Only {n} paired point(s) — too few to estimate a correlation "
                f"(need ≥ {MIN_PAIRS_FOR_RHO})."
            ),
        )

    mvals = np.array([p.metric_value for p in pairs], dtype=float)
    wvals = np.array([p.word_count for p in pairs], dtype=float)
    if np.unique(mvals).size < 2 or np.unique(wvals).size < 2:
        return MetricWordCorr(
            **base,
            interpretation=(
                "No variance in the biomarker or the word counts across the "
                "paired points — a correlation is undefined."
            ),
        )

    rho_raw, p_raw = spearmanr(mvals, wvals)
    if not np.isfinite(rho_raw):
        return MetricWordCorr(
            **base,
            interpretation="Correlation could not be computed (non-finite result).",
        )
    rho = round(float(rho_raw), 3)
    obs_sign = 0 if rho == 0 else (1 if rho > 0 else -1)
    matches = None
    if expected != 0 and obs_sign != 0:
        matches = (obs_sign == expected)

    # p-value only when n is large enough for it to mean anything.
    show_p = n >= MIN_PAIRS_FOR_SIGNIFICANCE
    p_value = round(float(p_raw), 4) if (show_p and np.isfinite(p_raw)) else None
    conclusive = bool(show_p and p_value is not None and p_value < 0.05)

    base["spearman_rho"] = rho
    base["p_value"] = p_value
    base["observed_sign"] = obs_sign
    base["matches_expected"] = matches
    base["statistically_conclusive"] = conclusive

    # Plain-language interpretation.
    if expected > 0:
        expect_phrase = f"higher {metric} ↔ more words (expected positive correlation)"
    elif expected < 0:
        expect_phrase = f"lower {metric} ↔ more words (expected negative correlation)"
    else:
        expect_phrase = "no clinically-expected direction for this metric"

    strength = (
        "strong" if abs(rho) >= 0.7 else
        "moderate" if abs(rho) >= 0.4 else "weak"
    )
    sign_word = "negative" if rho < 0 else ("positive" if rho > 0 else "flat")
    parts = [f"Spearman ρ = {rho:+.2f} ({strength} {sign_word}) over {n} paired timepoints."]
    if expected != 0:
        parts.append(
            f"Clinical expectation: {expect_phrase} — "
            + ("the sign matches." if matches else "the sign does NOT match.")
        )
    else:
        parts.append(f"({expect_phrase}.)")
    if not show_p:
        parts.append(
            f"n = {n} is below {MIN_PAIRS_FOR_SIGNIFICANCE}; the p-value is "
            "unreliable at this size (even ρ = ±1 is not significant), so it is "
            "omitted. Treat this as a picture, not proof."
        )
    elif conclusive:
        parts.append(f"p = {p_value} — statistically significant at this n.")
    else:
        parts.append(f"p = {p_value} — not statistically significant.")
    if confounded:
        parts.append("This biomarker also tracks normal maturation, which inflates "
                     "any apparent link with vocabulary growth.")

    return MetricWordCorr(**base, interpretation=" ".join(parts))


def summarize_word_correlation(result: WordCorrelation) -> dict:
    """JSON-serializable summary."""
    return {
        "n_word_observations": result.n_word_observations,
        "n_recordings": result.n_recordings,
        "metric_correlations": [asdict(mc) for mc in result.metric_correlations],
        "notes": result.notes,
    }


def render_word_correlation_md(result: WordCorrelation) -> str:
    """Human-readable Markdown for the family / clinician."""
    lines = ["# Vocabulary ↔ EEG correlation", ""]
    lines.append(
        f"_{result.n_recordings} recording(s), {result.n_word_observations} "
        f"vocabulary observation(s)._"
    )
    lines.append("")
    if not result.metric_correlations:
        for n in result.notes:
            lines.append(f"> {n}")
        return "\n".join(lines)

    lines.append("| Biomarker | ρ | n | Matches expectation | Conclusive |")
    lines.append("|---|---|---|---|---|")
    for mc in result.metric_correlations:
        rho = "—" if mc.spearman_rho is None else f"{mc.spearman_rho:+.2f}"
        match = ("n/a" if mc.matches_expected is None
                 else ("✅ yes" if mc.matches_expected else "❌ no"))
        conc = "yes" if mc.statistically_conclusive else "no"
        label = mc.metric + (" *" if mc.maturation_confounded else "")
        lines.append(f"| {label} | {rho} | {mc.n_pairs} | {match} | {conc} |")
    lines.append("")
    for mc in result.metric_correlations:
        lines.append(f"- **{mc.metric}** — {mc.interpretation}")
    lines.append("")
    if any(mc.maturation_confounded for mc in result.metric_correlations):
        lines.append("> \\* also tracks normal maturation — a matching correlation is expected even without a true language link.")
    lines.append("---")
    for n in result.notes:
        lines.append(f"> {n}")
    return "\n".join(lines)
