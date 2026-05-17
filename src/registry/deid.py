"""De-identifying submission builder — the only sanctioned path from
local findings to a registry-shaped submission JSON.

Threat model
------------
The local SQLite DB may contain anything: filenames with patient names,
free-text labels with dates, indication fields with narrative text,
embedded PHI from EDF headers, etc. Our job is to produce a JSON that
contains NONE of that — only the bucketed quantitative outputs we
actually want to aggregate.

Defense strategy: ALLOWLIST BY CONSTRUCTION
-------------------------------------------
We never copy a dict wholesale. We never traverse `findings` and pull
"whatever is there." Instead, the builder reads specific keys by name
and either:
  - copies the value if it passes type + range checks, or
  - sets the field to None (or omits it) if it doesn't.

Anything not listed in `_EXTRACTORS` below is structurally invisible to
the output. New fields are added by adding extractors, never by relaxing
the walker.

After construction we run a PHI scan (`phi_check.scan_for_phi`) as a
belt-and-suspenders check. If anything trips, the build fails.

Public API
----------
- `SubmissionInput`: typed dataclass for family-provided context
  (variant, age, sex, intervention, ...). All fields go through bucket
  + enum validation before any string lands in the submission.
- `build_submission(...)`: returns the submission dict or raises
  `BuildError` with the human-readable reason.

The output is JSON-serializable (`json.dumps(result)` always works)
and matches `schema.SCHEMA_VERSION`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from . import schema as _schema
from . import buckets as _buckets
from . import phi_check
from .consent import Consent, CURRENT_CONSENT_VERSION


class BuildError(ValueError):
    """Raised when a submission cannot be built safely. Message is
    human-readable and intended for direct display in the UI."""


@dataclass
class SubmissionInput:
    """Family-provided context for one submission.

    Every field has tight validation. Free text is NOT accepted except
    where explicitly allowed (intervention_name)."""

    variant_gene: str                          # e.g. "KCNQ3"
    variant_protein: str                       # e.g. "p.Arg230His"
    variant_type: str                          # one of schema.VARIANT_TYPES
    age_years: float | int                     # exact, bucketed internally
    sex: str                                   # one of schema.SEX_VALUES
    country_region: str | None = None          # ISO 3166 alpha-2, optional

    # Recording metadata (exact values, bucketed/normalized internally)
    duration_hours: float = 0.0
    had_sleep: bool = False
    montage: str = "unknown"                   # one of schema.MONTAGE_VALUES
    n_channels: int = 0

    # Intervention (all optional; either all set or all None)
    intervention_type: str | None = None       # schema.INTERVENTION_TYPES
    intervention_name: str | None = None
    intervention_record_kind: str | None = None  # schema.INTERVENTION_RECORD_KINDS
    linked_pre_submission_id: str | None = None


# ─── Per-finding extractors ────────────────────────────────────────────────
#
# Each extractor is a function (findings: dict) -> value-or-None.
# It KNOWS which key it cares about. It NEVER traverses or returns
# anything outside the value at that key. Failure to find or validate
# returns None — the output simply omits the field.

def _extract_pdr_hz(f: dict) -> float | None:
    bg = f.get("background") if isinstance(f, dict) else None
    if not isinstance(bg, dict):
        return None
    v = bg.get("pdr_hz")
    if isinstance(v, (int, float)) and 0.5 <= float(v) <= 20.0:
        return float(v)
    return None


def _extract_swi_by_stage(f: dict) -> dict[str, float] | None:
    swi = f.get("swi") if isinstance(f, dict) else None
    if not isinstance(swi, dict):
        return None
    raw = swi.get("swi_per_stage_pct")
    if not isinstance(raw, dict):
        return None
    out: dict[str, float] = {}
    for k, v in raw.items():
        if k in _schema.SLEEP_STAGE_KEYS and _schema._is_pct(v):
            out[k] = float(v)
    return out or None


def _extract_csws_met(f: dict) -> bool | None:
    swi = f.get("swi") if isinstance(f, dict) else None
    if not isinstance(swi, dict):
        return None
    v = swi.get("csws_criterion_met")
    return bool(v) if isinstance(v, bool) else None


def _extract_csws_threshold(f: dict) -> float | None:
    swi = f.get("swi") if isinstance(f, dict) else None
    if not isinstance(swi, dict):
        return None
    v = swi.get("csws_threshold_pct")
    if _schema._is_pct(v):
        return float(v)
    return None


def _extract_spindle_density(f: dict) -> float | None:
    sp = f.get("spindles") if isinstance(f, dict) else None
    if not isinstance(sp, dict):
        return None
    v = sp.get("density_per_minute")
    if _schema._is_nonneg_finite(v) and float(v) < 100.0:
        return float(v)
    return None


def _extract_spindle_norm_range(f: dict) -> list[float] | None:
    sp = f.get("spindles") if isinstance(f, dict) else None
    if not isinstance(sp, dict):
        return None
    r = sp.get("age_normative_range")
    if (
        isinstance(r, (list, tuple))
        and len(r) == 2
        and all(_schema._is_nonneg_finite(x) for x in r)
    ):
        return [float(r[0]), float(r[1])]
    return None


def _extract_spindle_interp(f: dict) -> str | None:
    sp = f.get("spindles") if isinstance(f, dict) else None
    if not isinstance(sp, dict):
        return None
    v = sp.get("interpretation")
    return v if v in _schema.SPINDLE_INTERPRETATIONS else None


def _extract_activation_factor(f: dict) -> float | None:
    st = f.get("state_split") if isinstance(f, dict) else None
    if not isinstance(st, dict):
        return None
    v = st.get("activation_factor")
    if _schema._is_nonneg_finite(v) and float(v) < 10_000.0:
        return float(v)
    return None


def _extract_activation_label(f: dict) -> str | None:
    st = f.get("state_split") if isinstance(f, dict) else None
    if not isinstance(st, dict):
        return None
    v = st.get("activation_label")
    return v if v in _schema.ACTIVATION_LABELS else None


def _extract_morphology_per_min(f: dict) -> float | None:
    m = f.get("morphology") if isinstance(f, dict) else None
    if not isinstance(m, dict):
        return None
    v = m.get("events_per_minute")
    if _schema._is_nonneg_finite(v) and float(v) < 10_000.0:
        return float(v)
    return None


def _extract_morphology_sw_pct(f: dict) -> float | None:
    m = f.get("morphology") if isinstance(f, dict) else None
    if not isinstance(m, dict):
        return None
    v = m.get("pct_complex_spike_wave")
    if _schema._is_pct(v):
        return float(v)
    return None


def _extract_sleep_stages_pct(f: dict) -> dict[str, float] | None:
    ss = f.get("sleep_stages") if isinstance(f, dict) else None
    if not isinstance(ss, dict):
        return None
    raw = ss.get("stage_pct") or ss.get("stages_pct")
    if not isinstance(raw, dict):
        return None
    out: dict[str, float] = {}
    for k, v in raw.items():
        if k in _schema.SLEEP_STAGE_KEYS and _schema._is_pct(v):
            out[k] = float(v)
    return out or None


def _extract_n_sleep_cycles(f: dict) -> int | None:
    arch = f.get("sleep_architecture") if isinstance(f, dict) else None
    if not isinstance(arch, dict):
        return None
    v = arch.get("n_cycles")
    if isinstance(v, int) and 0 <= v <= 100:
        return v
    return None


def _extract_quality_grade(f: dict) -> str | None:
    q = f.get("quality") if isinstance(f, dict) else None
    if not isinstance(q, dict):
        return None
    v = q.get("grade")
    return v if v in _schema.QUALITY_GRADES else None


# ─── Schema v2 extractors ─────────────────────────────────────────────────────


def _extract_coupling_plv_bucket(f: dict) -> str | None:
    c = f.get("coupling") if isinstance(f, dict) else None
    if not isinstance(c, dict):
        return None
    plv = c.get("plv")
    if plv is None or not _schema._is_nonneg_finite(plv):
        return None
    return _buckets.bucket_plv(float(plv))


def _extract_coupling_phase_octant(f: dict) -> str | None:
    c = f.get("coupling") if isinstance(f, dict) else None
    if not isinstance(c, dict):
        return None
    deg = c.get("preferred_phase_deg")
    if deg is None or not _schema._is_finite(deg):
        return None
    return _buckets.bucket_phase_deg(float(deg))


def _extract_coupling_n_bucket(f: dict) -> str | None:
    c = f.get("coupling") if isinstance(f, dict) else None
    if not isinstance(c, dict):
        return None
    n = c.get("n_spindles_in_so")
    if not isinstance(n, int) or n < 0:
        return None
    return _buckets.bucket_coupled_events(n)


def _extract_coupling_rayleigh_sig(f: dict) -> bool | None:
    c = f.get("coupling") if isinstance(f, dict) else None
    if not isinstance(c, dict):
        return None
    p = c.get("rayleigh_p")
    if p is None or not _schema._is_nonneg_finite(p):
        return None
    return bool(float(p) < 0.05)


def _extract_sw_density_bucket(f: dict) -> str | None:
    sw = f.get("slow_waves") if isinstance(f, dict) else None
    if not isinstance(sw, dict):
        return None
    d = sw.get("density_per_minute")
    if d is None or not _schema._is_nonneg_finite(d):
        return None
    return _buckets.bucket_sw_density(float(d))


def _extract_sw_ptp_bucket(f: dict) -> str | None:
    sw = f.get("slow_waves") if isinstance(f, dict) else None
    if not isinstance(sw, dict):
        return None
    ptp = sw.get("mean_ptp_uv")
    if ptp is None or not _schema._is_nonneg_finite(ptp):
        return None
    return _buckets.bucket_sw_ptp_uv(float(ptp))


def _extract_sw_method(f: dict) -> str | None:
    sw = f.get("slow_waves") if isinstance(f, dict) else None
    if not isinstance(sw, dict):
        return None
    v = sw.get("method")
    # Allow only known method strings
    if v in ("yasa", "heuristic"):
        return v
    return None


def _extract_hfo_rate_bucket(f: dict) -> str | None:
    hfo = f.get("hfo_ripples") if isinstance(f, dict) else None
    if not isinstance(hfo, dict):
        return None
    rate = hfo.get("rate_per_minute_nrem")
    if rate is None or not _schema._is_nonneg_finite(rate):
        return None
    return _buckets.bucket_hfo_rate(float(rate))


def _extract_hfo_available(f: dict) -> bool | None:
    hfo = f.get("hfo_ripples") if isinstance(f, dict) else None
    if not isinstance(hfo, dict):
        return None
    v = hfo.get("available")
    return bool(v) if isinstance(v, bool) else None


def _extract_ied_method(f: dict) -> str | None:
    ied = f.get("ied_ml") if isinstance(f, dict) else None
    if not isinstance(ied, dict):
        return None
    v = ied.get("method")
    return v if v in _schema.IED_METHODS else None


def _extract_ied_rate_bucket(f: dict) -> str | None:
    ied = f.get("ied_ml") if isinstance(f, dict) else None
    if not isinstance(ied, dict):
        return None
    rate = ied.get("rate_per_minute")
    if rate is None or not _schema._is_nonneg_finite(rate):
        return None
    return _buckets.bucket_ied_rate(float(rate))


def _extract_ied_age_flag(f: dict) -> str | None:
    ied = f.get("ied_ml") if isinstance(f, dict) else None
    if not isinstance(ied, dict):
        return None
    v = ied.get("age_appropriateness_flag")
    return v if v in _schema.IED_AGE_FLAGS else None


def _extract_ied_agreement_bucket(f: dict) -> str | None:
    ied = f.get("ied_ml") if isinstance(f, dict) else None
    if not isinstance(ied, dict):
        return None
    pct = ied.get("agreement_with_morphology_pct")
    if pct is None or not _schema._is_nonneg_finite(pct):
        return None
    return _buckets.bucket_ied_agreement(float(pct))


def _extract_ied_rolandic_bucket(f: dict) -> str | None:
    ied = f.get("ied_ml") if isinstance(f, dict) else None
    if not isinstance(ied, dict):
        return None
    n = ied.get("n_likely_rolandic_benign")
    if not isinstance(n, int) or n < 0:
        return None
    return _buckets.bucket_ied_rolandic(n)


def _extract_ied_nrem_rate_bucket(f: dict) -> str | None:
    """H5 (option a): expose IED NREM rate as a bucketed registry field.

    Clinically important for CSWS/ESES-spectrum evaluation. The raw
    nrem_rate_per_min float from IEDDetectionResult is bucketed identically
    to the overall IED rate, enabling registry-level NREM-vs-overall comparison.
    """
    ied = f.get("ied_ml") if isinstance(f, dict) else None
    if not isinstance(ied, dict):
        return None
    rate = ied.get("nrem_rate_per_min")
    if rate is None or not _schema._is_nonneg_finite(rate):
        return None
    return _buckets.bucket_ied_nrem_rate(float(rate))


def _extract_aperiodic_chi_n2_bucket(f: dict) -> str | None:
    """Extract aperiodic exponent χ for N2 state and bucket it. (v0.16.0)"""
    aperiodic = f.get("aperiodic") if isinstance(f, dict) else None
    if not isinstance(aperiodic, dict):
        return None
    chi_by_state = aperiodic.get("chi_by_state")
    if not isinstance(chi_by_state, dict):
        return None
    n2 = chi_by_state.get("n2")
    if not isinstance(n2, dict):
        return None
    median = n2.get("median")
    if median is None or not _schema._is_nonneg_finite(median):
        return None
    return _buckets.bucket_aperiodic_chi_n2(float(median))


def _extract_pdr_asymmetry_bucket(f: dict) -> str | None:
    """Extract PDR posterior asymmetry bucket. (v0.16.0)"""
    bg = f.get("background") if isinstance(f, dict) else None
    if not isinstance(bg, dict):
        return None
    interp = bg.get("asymmetry_interpretation")
    if not isinstance(interp, str):
        return None
    return _buckets.bucket_pdr_asymmetry(interp)


def _extract_microstate_dominant(f: dict) -> str | None:
    """Extract dominant microstate class (A/B/C/D). (v0.16.0)"""
    ms = f.get("microstates") if isinstance(f, dict) else None
    if not isinstance(ms, dict):
        return None
    dom = ms.get("dominant_microstate")
    if dom in _schema.MICROSTATE_DOMINANT_VALUES:
        return str(dom)
    return None


def _extract_spike_topography_pattern(f: dict) -> str | None:
    """Extract spike topography pattern. (v0.17.0)"""
    pr = f.get("pattern_recognition") if isinstance(f, dict) else None
    if not isinstance(pr, dict):
        return None
    v = pr.get("spike_topography_pattern")
    if v in _schema.SPIKE_TOPOGRAPHY_PATTERNS:
        return str(v)
    return None


def _extract_spike_polyspike_pct_bucket(f: dict) -> str | None:
    """Extract polyspike percentage bucket. (v0.17.0)"""
    pr = f.get("pattern_recognition") if isinstance(f, dict) else None
    if not isinstance(pr, dict):
        return None
    morph = pr.get("morphology_subtypes")
    if not isinstance(morph, dict):
        return None
    pct = morph.get("pct_polyspike")
    if pct is None or not _schema._is_nonneg_finite(pct):
        return None
    return _buckets.bucket_spike_polyspike_pct(float(pct))


def _extract_sleep_activation_classification(f: dict) -> str | None:
    """Extract sleep activation classification. (v0.17.0)"""
    pr = f.get("pattern_recognition") if isinstance(f, dict) else None
    if not isinstance(pr, dict):
        return None
    v = pr.get("sleep_activation_classification")
    if v in _schema.SLEEP_ACTIVATION_CLASSIFICATIONS:
        return str(v)
    return None


def _extract_csws_risk_score_bucket(f: dict) -> str | None:
    """Extract CSWS risk score bucket. (v0.17.0)"""
    pr = f.get("pattern_recognition") if isinstance(f, dict) else None
    if not isinstance(pr, dict):
        return None
    score = pr.get("csws_risk_score")
    if score is None or not _schema._is_nonneg_finite(score):
        return None
    return _buckets.bucket_csws_risk_score(float(score))


def _extract_hfo_pct_on_spike_bucket(f: dict) -> str | None:
    hfo = f.get("hfo_ripples") if isinstance(f, dict) else None
    if not isinstance(hfo, dict):
        return None
    total = hfo.get("n_ripples_total")
    on_spike = hfo.get("n_ripples_on_spike")
    if (
        not isinstance(total, int) or not isinstance(on_spike, int)
        or total <= 0
    ):
        return None
    pct = 100.0 * on_spike / total
    # Bucket: "<10", "10-50", "50-90", ">90"
    if pct < 10:
        return "<10"
    if pct < 50:
        return "10-50"
    if pct < 90:
        return "50-90"
    return ">90"


# Order matters only for diff readability; semantics are independent.
_EXTRACTORS_V1: dict[str, Callable[[dict], Any]] = {
    "background_pdr_hz": _extract_pdr_hz,
    "swi_pct_by_stage": _extract_swi_by_stage,
    "csws_criterion_met": _extract_csws_met,
    "csws_threshold_pct": _extract_csws_threshold,
    "spindle_density_per_min_central": _extract_spindle_density,
    "spindle_age_norm_range": _extract_spindle_norm_range,
    "spindle_interpretation": _extract_spindle_interp,
    "activation_factor": _extract_activation_factor,
    "activation_label": _extract_activation_label,
    "morphology_events_per_min": _extract_morphology_per_min,
    "morphology_spike_wave_pct": _extract_morphology_sw_pct,
    "sleep_stages_pct": _extract_sleep_stages_pct,
    "n_sleep_cycles": _extract_n_sleep_cycles,
    "quality_grade": _extract_quality_grade,
}

_EXTRACTORS_V2_ADDITIONAL: dict[str, Callable[[dict], Any]] = {
    # coupling
    "coupling_plv_bucket": _extract_coupling_plv_bucket,
    "coupling_preferred_phase_octant": _extract_coupling_phase_octant,
    "coupling_n_events_bucket": _extract_coupling_n_bucket,
    "coupling_rayleigh_significant": _extract_coupling_rayleigh_sig,
    # slow waves
    "sw_density_bucket": _extract_sw_density_bucket,
    "sw_mean_ptp_bucket": _extract_sw_ptp_bucket,
    "sw_method": _extract_sw_method,
    # HFO
    "hfo_rate_bucket": _extract_hfo_rate_bucket,
    "hfo_available": _extract_hfo_available,
    "hfo_pct_on_spike_bucket": _extract_hfo_pct_on_spike_bucket,
    # v0.13.3 — IED detection
    "ied_method": _extract_ied_method,
    "ied_rate_bucket": _extract_ied_rate_bucket,
    "ied_age_flag": _extract_ied_age_flag,
    "ied_agreement_bucket": _extract_ied_agreement_bucket,
    "ied_n_rolandic_benign_bucket": _extract_ied_rolandic_bucket,
    "ied_nrem_rate_bucket": _extract_ied_nrem_rate_bucket,
    # v0.16.0 — Tier 3 add-ons
    "aperiodic_chi_n2_bucket": _extract_aperiodic_chi_n2_bucket,
    "pdr_asymmetry_bucket": _extract_pdr_asymmetry_bucket,
    "microstate_dominant": _extract_microstate_dominant,
    # v0.17.0 — Pattern recognition (all optional, additive)
    "spike_topography_pattern": _extract_spike_topography_pattern,
    "spike_morphology_polyspike_pct_bucket": _extract_spike_polyspike_pct_bucket,
    "sleep_activation_classification": _extract_sleep_activation_classification,
    "csws_risk_score_bucket": _extract_csws_risk_score_bucket,
}

# Default extractors (v2)
_EXTRACTORS: dict[str, Callable[[dict], Any]] = {
    **_EXTRACTORS_V1,
    **_EXTRACTORS_V2_ADDITIONAL,
}


# ─── Builder ───────────────────────────────────────────────────────────────

def build_submission(
    *,
    findings: dict,
    user_input: SubmissionInput,
    consent: Consent,
    tool_version: str,
    submission_id: str | None = None,
    now: datetime | None = None,
    schema_version_target: int = 2,
) -> dict:
    """Build a registry-shaped submission JSON.

    Raises BuildError if any input is invalid, consent is missing,
    or the final submission trips the PHI scanner.

    All time-dependent values use `now` (default: datetime.now()) so
    tests can pin the clock.
    """
    if not consent or not isinstance(consent, Consent):
        raise BuildError("consent record is required")
    if not consent.given:
        raise BuildError("consent has not been given; refusing to build")
    if consent.version != CURRENT_CONSENT_VERSION:
        raise BuildError(
            f"consent version {consent.version} does not match current "
            f"version {CURRENT_CONSENT_VERSION}; please re-affirm"
        )

    # Validate user_input — every field gets reconstructed, never copied
    # through wholesale.
    ui = _validate_user_input(user_input)

    now = now or datetime.now()
    sid = submission_id or str(uuid.uuid4())
    if not _schema.UUID4_RE.match(sid):
        raise BuildError(f"submission_id is not a valid uuid4: {sid!r}")

    submission_month = now.strftime("%Y-%m")

    # Select extractors based on target schema version.
    # v1: only the v1 fields (for backward-compat testing / legacy mode).
    # v2: v1 + additional v2 fields (additive, all optional).
    if schema_version_target == 1:
        active_extractors = _EXTRACTORS_V1
        schema_ver_out = 1
    else:
        active_extractors = _EXTRACTORS
        schema_ver_out = _schema.SCHEMA_VERSION

    findings_out: dict[str, Any] = {}
    for field_name, extractor in active_extractors.items():
        try:
            value = extractor(findings)
        except Exception:
            value = None  # extractors must never crash the build
        if value is not None:
            findings_out[field_name] = value

    intervention_out: dict[str, Any] | None = None
    if ui["intervention_type"] is not None:
        intervention_out = {
            "type": ui["intervention_type"],
            "name": ui["intervention_name"],
            "record_kind": ui["intervention_record_kind"],
            "linked_pre_submission_id": ui["linked_pre_submission_id"],
        }

    submission = {
        "submission_id": sid,
        "schema_version": schema_ver_out,
        "submitted_at_month": submission_month,
        "consent": {
            "version": consent.version,
            "given": consent.given,
            "given_at_month": consent.given_at_month,
        },
        "subject": {
            "variant_gene": ui["variant_gene"],
            "variant_protein": ui["variant_protein"],
            "variant_type": ui["variant_type"],
            "age_years_bucket": ui["age_years_bucket"],
            "sex": ui["sex"],
            "country_region": ui["country_region"],
        },
        "recording": {
            "duration_hours_bucket": ui["duration_hours_bucket"],
            "had_sleep": ui["had_sleep"],
            "montage": ui["montage"],
            "n_channels": ui["n_channels"],
        },
        "findings": findings_out,
        "intervention": intervention_out,
        "tool_version": tool_version,
    }

    # Belt-and-suspenders: scan the OUTPUT for PHI patterns.
    phi_findings = phi_check.scan_for_phi(submission)
    if phi_findings:
        raise BuildError(
            "PHI scan flagged the constructed submission:\n  "
            + "\n  ".join(phi_findings)
        )

    return submission


# ─── Validation helpers ────────────────────────────────────────────────────

def _validate_user_input(ui: SubmissionInput) -> dict[str, Any]:
    """Reconstruct each user-input field with strict validation.

    Returns a dict of cleaned values. Raises BuildError on any rejection.
    """
    if not isinstance(ui, SubmissionInput):
        raise BuildError("user_input must be a SubmissionInput instance")

    # Gene
    g = ui.variant_gene or ""
    if not _schema.GENE_SYMBOL_RE.match(g):
        raise BuildError(
            f"variant_gene {g!r} does not match HGNC-style symbol "
            f"(2-16 uppercase chars)"
        )

    # Variant protein
    p = ui.variant_protein or ""
    if not _schema.VARIANT_PROTEIN_RE.match(p):
        raise BuildError(
            f"variant_protein {p!r} is not in p.RefXxxNNNAltYyy form"
        )

    # Variant type
    if ui.variant_type not in _schema.VARIANT_TYPES:
        raise BuildError(
            f"variant_type {ui.variant_type!r} not in "
            f"{sorted(_schema.VARIANT_TYPES)}"
        )

    # Age → bucket
    age_bucket = _buckets.bucket_age_years(ui.age_years)
    if age_bucket is None:
        raise BuildError(f"age_years {ui.age_years!r} could not be bucketed")

    # Sex
    if ui.sex not in _schema.SEX_VALUES:
        raise BuildError(f"sex {ui.sex!r} not in {sorted(_schema.SEX_VALUES)}")

    # Country
    country = ui.country_region
    if country is not None:
        if not _schema.COUNTRY_RE.match(country):
            raise BuildError(
                f"country_region {country!r} not ISO 3166-1 alpha-2"
            )

    # Duration → bucket
    dur_bucket = _buckets.bucket_duration_hours(ui.duration_hours)
    if dur_bucket is None:
        raise BuildError(
            f"duration_hours {ui.duration_hours!r} could not be bucketed"
        )

    # Montage
    if ui.montage not in _schema.MONTAGE_VALUES:
        raise BuildError(
            f"montage {ui.montage!r} not in {sorted(_schema.MONTAGE_VALUES)}"
        )

    # n_channels
    if not isinstance(ui.n_channels, int) or not (0 <= ui.n_channels <= 256):
        raise BuildError(
            f"n_channels {ui.n_channels!r} must be an int in [0, 256]"
        )

    had_sleep = bool(ui.had_sleep)

    # Intervention: all-or-nothing.
    itype = ui.intervention_type
    iname = ui.intervention_name
    ikind = ui.intervention_record_kind
    ilink = ui.linked_pre_submission_id
    if itype is not None:
        if itype not in _schema.INTERVENTION_TYPES:
            raise BuildError(
                f"intervention_type {itype!r} not in "
                f"{sorted(_schema.INTERVENTION_TYPES)}"
            )
        if not isinstance(iname, str) or not iname.strip():
            raise BuildError("intervention_name is required when type is set")
        if len(iname) > _schema.INTERVENTION_NAME_MAX_LEN:
            raise BuildError(
                f"intervention_name longer than "
                f"{_schema.INTERVENTION_NAME_MAX_LEN} chars"
            )
        # Strip surrounding whitespace; PHI scan will handle anything wild.
        iname = iname.strip()
        # B1: Block non-ASCII letters in intervention_name (free-text PHI
        # risk via Unicode homoglyphs — e.g. Cyrillic lookalikes).
        # Spaces and hyphens are allowed; everything else must be ASCII.
        if iname and not all(ord(c) < 128 or c in " -" for c in iname):
            raise BuildError(
                "intervention_name contains non-ASCII characters. "
                "For privacy reasons free-text fields must be ASCII-only. "
                "Use a transliterated form."
            )
        if ikind not in _schema.INTERVENTION_RECORD_KINDS:
            raise BuildError(
                f"intervention_record_kind {ikind!r} not in "
                f"{sorted(_schema.INTERVENTION_RECORD_KINDS)}"
            )
        if ilink is not None and not _schema.UUID4_RE.match(ilink):
            raise BuildError(
                f"linked_pre_submission_id {ilink!r} is not a uuid4"
            )
    else:
        # Force the dependents to None if the type isn't set
        iname = None
        ikind = None
        ilink = None

    return {
        "variant_gene": g,
        "variant_protein": p,
        "variant_type": ui.variant_type,
        "age_years_bucket": age_bucket,
        "sex": ui.sex,
        "country_region": country,
        "duration_hours_bucket": dur_bucket,
        "had_sleep": had_sleep,
        "montage": ui.montage,
        "n_channels": ui.n_channels,
        "intervention_type": itype,
        "intervention_name": iname,
        "intervention_record_kind": ikind,
        "linked_pre_submission_id": ilink,
    }
