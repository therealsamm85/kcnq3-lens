"""Longitudinal EEG comparison — compare two stored findings over time.

This module works on *stored findings dicts* (from the SQLite longitudinal
store), NOT on raw EEG data. It answers: did anything meaningfully change
between two recordings, and what caveats must we flag?

Key design constraints
----------------------
- Methodological fairness: a 24h ambulatory vs a 49min routine EEG cannot
  be compared at face value. We detect this and use only a fair window from
  the longer recording when spike-rate is normalised to /min (the findings
  already carry events_per_minute so this is mostly a flag).
- Topographic shift: we compare the *rank order* of top-5 channels, not raw
  values, because different setups may use slightly different electrode
  impedances.
- Spindle comparison: only attempted when both recordings have genuine sleep
  (sleep_duration_h >= 2).
- Honest framing: we produce machine-readable confound lists and
  interpretation-hint strings so the UI can show them prominently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ─── Main dataclass ──────────────────────────────────────────────────────────

@dataclass
class LongitudinalDelta:
    """Comparison between two EEG recordings over time."""

    # Metadata
    recording_a: dict  # {date, label, duration_h, condition, source_filename}
    recording_b: dict
    age_delta_years: float

    # Compatibility
    duration_compatible: bool     # both long-form OR both short routine
    methodology_warning: str      # human-readable confound description

    # Per-channel spike rate: ch -> (rate_a, rate_b) in events/min
    spike_rate_per_channel: dict[str, tuple[float, float]]
    mean_spike_rate_delta_pct: float  # (b - a) / a * 100, negative = improvement

    # Topographic shift
    topographic_shift: str        # "preserved" | "flattened" | "different_hotspots"
    top5_channels_a: list[str]    # rank-ordered channel names
    top5_channels_b: list[str]

    # Background
    pdr_a: float | None
    pdr_b: float | None
    pdr_delta_hz: float | None

    # Morphology
    complex_sw_pct_a: float
    complex_sw_pct_b: float

    # Spindles (only if both have sleep)
    spindle_density_a: float | None
    spindle_density_b: float | None
    spindle_delta_pct: float | None

    # Honest framing
    confounds: list[str]
    interpretation_hints: list[str]


# ─── Duration compatibility helpers ──────────────────────────────────────────

_LONG_FORM_H = 2.0           # threshold for "this has real sleep content"
_DURATION_MISMATCH_RATIO = 5  # flag if longer > shorter * this


def _get_duration_h(findings: dict[str, Any]) -> float:
    """Extract recording duration in hours from a findings dict.

    Tries multiple paths where duration might be stored.
    """
    # v0.12+ stores duration in metadata, but findings might carry it too
    # Check sleep_architecture first (most reliable for long recordings)
    arch = findings.get("sleep_architecture", {})
    if arch:
        total_h = arch.get("total_recording_time_h") or arch.get("total_sleep_time_h")
        if total_h and total_h > 0:
            return float(total_h)

    # Fall back to time_of_night which covers the analyzed window
    tn = findings.get("time_of_night", {})
    analyzed_min = tn.get("analyzed_duration_min")
    if analyzed_min and analyzed_min > 0:
        return float(analyzed_min) / 60.0

    # Fall back to morphology events_per_minute + total event count
    morph = findings.get("morphology", {})
    n_events = morph.get("n_events") or morph.get("total_events")
    epm = morph.get("events_per_minute")
    if n_events and epm and epm > 0:
        return (float(n_events) / float(epm)) / 60.0

    # Last resort: spindle density implies some sleep window was analyzed
    return 0.0


def _get_sleep_duration_h(findings: dict[str, Any]) -> float:
    """Return sleep duration hours, 0 if wake-only or unknown."""
    arch = findings.get("sleep_architecture", {})
    if arch:
        tst = arch.get("total_sleep_time_h")
        if tst is not None:
            return float(tst)
    # Check if sleep stages were detected at all
    ss = findings.get("sleep_stages", {})
    if ss:
        # sleep_minutes is a common summary key
        sm = ss.get("sleep_minutes") or ss.get("total_sleep_minutes")
        if sm:
            return float(sm) / 60.0
    return 0.0


# ─── Per-channel spike rate helpers ──────────────────────────────────────────

def _channel_spike_rates(findings: dict[str, Any]) -> dict[str, float]:
    """Build {channel_name: proportional spike-rate index} from findings.

    Strategy:
    1. If morphology has per-channel counts → use those normalized to /min.
    2. Otherwise use topography kurtosis as a relative index, then rescale
       so the *mean* equals the global ``events_per_minute`` from morphology.
       This makes mean_spike_rate_delta_pct track the global rate change
       (e.g. 17.4 → 2.2 /min = -87%) while preserving per-channel topology.
    """
    rates: dict[str, float] = {}

    # Try morphology per-channel counts first
    morph = findings.get("morphology", {})
    per_ch = morph.get("per_channel") or morph.get("channel_counts")
    if per_ch and isinstance(per_ch, dict):
        total_min = _get_duration_h(findings) * 60.0 or 1.0
        for ch, cnt in per_ch.items():
            rates[ch] = float(cnt) / total_min
        return rates

    # Fall back to topography kurtosis indices
    topo = findings.get("topography", {})
    all_ch = topo.get("all_channels", [])
    if all_ch:
        for entry in all_ch:
            name = entry.get("name") or entry.get("channel")
            median = entry.get("median") or entry.get("median_kurtosis")
            if name and median is not None:
                rates[name] = float(median)

        # Rescale so the mean equals global events_per_minute (when available).
        # This anchors the mean_spike_rate_delta_pct to the clinical event count
        # rather than the raw kurtosis value.
        global_epm = morph.get("events_per_minute")
        if global_epm is not None and global_epm >= 0:
            kurtosis_mean = (sum(rates.values()) / len(rates)) if rates else 0
            if kurtosis_mean > 0:
                scale = float(global_epm) / kurtosis_mean
                rates = {ch: v * scale for ch, v in rates.items()}

        return rates

    # Last resort: use scalar events_per_minute as a single pseudo-channel
    epm = morph.get("events_per_minute")
    if epm is not None:
        rates["__global__"] = float(epm)
    return rates


def _top5_channels(channel_rates: dict[str, float]) -> list[str]:
    """Return the top-5 channel names sorted by descending rate."""
    if not channel_rates:
        return []
    ranked = sorted(channel_rates.items(), key=lambda x: -x[1])
    return [ch for ch, _ in ranked[:5]]


def _mean_rate(rates: dict[str, float]) -> float:
    if not rates:
        return 0.0
    return sum(rates.values()) / len(rates)


# ─── Topographic shift detection ─────────────────────────────────────────────

def _detect_topographic_shift(
    rates_a: dict[str, float],
    rates_b: dict[str, float],
) -> tuple[str, list[str], list[str]]:
    """Classify topographic shift as 'preserved' | 'flattened' | 'different_hotspots'.

    Algorithm:
    1. "flattened": if rec_b has coefficient of variation < 0.15 (all channels
       approximately equal), classify as flattened regardless of overlap.
    2. "preserved": if >=3 of top-5 channels from A appear in top-5 of B.
    3. "different_hotspots": otherwise — new hotspots emerged.
    """
    top5_a = _top5_channels(rates_a)
    top5_b = _top5_channels(rates_b)

    # Check for flattening in B: low CV means homogeneous burden
    if rates_b:
        vals_b = list(rates_b.values())
        mean_b = sum(vals_b) / len(vals_b)
        if mean_b > 0:
            std_b = (sum((v - mean_b) ** 2 for v in vals_b) / len(vals_b)) ** 0.5
            cv_b = std_b / mean_b
            if cv_b < 0.15:
                return "flattened", top5_a, top5_b

    # Count overlap between top-5 sets
    set_a = set(top5_a)
    set_b = set(top5_b)
    overlap = len(set_a & set_b)
    if overlap >= 3:
        return "preserved", top5_a, top5_b
    return "different_hotspots", top5_a, top5_b


# ─── Confound detection ───────────────────────────────────────────────────────

def _detect_confounds(
    findings_a: dict[str, Any],
    findings_b: dict[str, Any],
    age_delta_years: float,
    dur_a: float,
    dur_b: float,
    condition_a: str,
    condition_b: str,
    mean_rate_a: float,
    mean_rate_b: float,
    metadata_a: dict[str, Any],
    metadata_b: dict[str, Any],
) -> list[str]:
    confounds: list[str] = []

    # Age delta > 6 months
    if age_delta_years > 0.5:
        months = round(age_delta_years * 12)
        confounds.append(
            f"age_delta: {months} months between recordings — "
            "brain maturation alone can change EEG patterns"
        )

    # Duration mismatch
    if dur_a > 0 and dur_b > 0:
        longer = max(dur_a, dur_b)
        shorter = min(dur_a, dur_b)
        if shorter > 0 and longer / shorter > _DURATION_MISMATCH_RATIO:
            ratio = round(longer / shorter, 1)
            dur_a_str = f"{dur_a:.1f}h" if dur_a >= 1.0 else f"{round(dur_a*60)}min"
            dur_b_str = f"{dur_b:.1f}h" if dur_b >= 1.0 else f"{round(dur_b*60)}min"
            confounds.append(
                f"duration_mismatch: {dur_a_str} vs {dur_b_str} ({ratio}× difference) — "
                "spike counts are normalized to /min but sampling bias differs"
            )

    # Different recording conditions
    if condition_a and condition_b and condition_a.lower() != condition_b.lower():
        confounds.append(
            f"recording_condition: '{condition_a}' vs '{condition_b}' — "
            "routine wake vs ambulatory sleep-containing recordings capture different states"
        )

    # Quality signal: if one recording had large delta in raw spike count, suspect artifact
    if mean_rate_a > 0 and mean_rate_b > 0:
        delta_pct = abs(mean_rate_b - mean_rate_a) / mean_rate_a * 100
        if delta_pct > 50:
            confounds.append(
                f"recording_quality: large spike-rate change ({round(delta_pct)}%) — "
                "verify neither recording has excessive artifact"
            )

    # Different time of recording day
    tod_a = metadata_a.get("time_of_day") or metadata_a.get("start_time")
    tod_b = metadata_b.get("time_of_day") or metadata_b.get("start_time")
    if tod_a and tod_b and tod_a != tod_b:
        confounds.append(
            f"time_of_day: recording A at {tod_a}, B at {tod_b} — "
            "circadian effects on spike burden are well-documented"
        )

    return confounds


# ─── Interpretation hints ─────────────────────────────────────────────────────

def _build_interpretation_hints(
    mean_spike_rate_delta_pct: float,
    topographic_shift: str,
    confounds: list[str],
    duration_compatible: bool,
    pdr_delta_hz: float | None,
    spindle_delta_pct: float | None,
) -> list[str]:
    hints: list[str] = []

    # Spike rate change
    if abs(mean_spike_rate_delta_pct) < 10:
        hints.append("Spike burden is essentially unchanged between recordings.")
    elif mean_spike_rate_delta_pct < -50:
        hints.append(
            f"Spike burden decreased by {abs(round(mean_spike_rate_delta_pct))}% — "
            "a substantial reduction, but confounds below should be reviewed before "
            "attributing this to treatment response."
        )
    elif mean_spike_rate_delta_pct < -20:
        hints.append(
            f"Spike burden decreased by {abs(round(mean_spike_rate_delta_pct))}% — "
            "a moderate reduction."
        )
    elif mean_spike_rate_delta_pct > 20:
        hints.append(
            f"Spike burden increased by {round(mean_spike_rate_delta_pct)}% — "
            "may indicate disease progression or recording artifact."
        )

    # Topographic findings
    if topographic_shift == "flattened":
        hints.append(
            "Topography is now homogeneous (no dominant hotspot) — "
            "this may reflect genuine improvement or a shorter recording with less "
            "focal sampling."
        )
    elif topographic_shift == "preserved":
        hints.append("Spike hotspots are in the same regions as before.")
    elif topographic_shift == "different_hotspots":
        hints.append(
            "Different channels dominate — new hotspots emerged. This is unusual "
            "with benign interventions; check electrode placement consistency."
        )

    # PDR change
    if pdr_delta_hz is not None:
        if pdr_delta_hz >= 0.5:
            hints.append(
                f"Posterior dominant rhythm increased by {round(pdr_delta_hz, 1)} Hz — "
                "a positive sign for background organization."
            )
        elif pdr_delta_hz <= -0.5:
            hints.append(
                f"Posterior dominant rhythm decreased by {abs(round(pdr_delta_hz, 1))} Hz — "
                "may indicate sedation, medication effect, or disease progression."
            )

    # Spindle comparison
    if spindle_delta_pct is not None:
        if spindle_delta_pct > 15:
            hints.append(
                f"Sleep spindle density increased {round(spindle_delta_pct)}% — "
                "spindles support sleep-dependent memory; improvement is encouraging."
            )
        elif spindle_delta_pct < -15:
            hints.append(
                f"Sleep spindle density decreased {abs(round(spindle_delta_pct))}% — "
                "may reflect sedative medication effect or sleep fragmentation."
            )

    # Duration incompatibility warning
    if not duration_compatible:
        hints.append(
            "These recordings are not directly comparable (routine vs long-form). "
            "Interpretation should focus on topographic patterns and PDR, "
            "not absolute spike counts."
        )

    # Confound count
    if len(confounds) >= 3:
        hints.append(
            f"{len(confounds)} methodological confounds detected. "
            "Treat any change as a hypothesis to be confirmed with matched recordings."
        )

    return hints


# ─── Main public function ─────────────────────────────────────────────────────

def compare_recordings(
    findings_a: dict[str, Any],
    findings_b: dict[str, Any],
    date_a: str = "",
    date_b: str = "",
    label_a: str = "",
    label_b: str = "",
    age_a_years: float | None = None,
    age_b_years: float | None = None,
    condition_a: str = "",
    condition_b: str = "",
    metadata_a: dict[str, Any] | None = None,
    metadata_b: dict[str, Any] | None = None,
) -> LongitudinalDelta:
    """Compare two EEG findings dicts and return a structured delta.

    Parameters
    ----------
    findings_a, findings_b
        Output of ``run_all_analyses`` (or loaded from the longitudinal store).
    date_a, date_b
        Recording dates as 'YYYY-MM-DD' strings.
    label_a, label_b
        Human-readable labels (e.g. 'baseline', 'post-supplements').
    age_a_years, age_b_years
        Child's age at each recording. Used for age-delta confound detection.
    condition_a, condition_b
        Recording conditions ('routine_wake', 'ambulatory_sleep', etc.).
    metadata_a, metadata_b
        Optional metadata dicts from StoredEntry.
    """
    meta_a = metadata_a or {}
    meta_b = metadata_b or {}

    # ── Ages ────────────────────────────────────────────────────────────────
    age_delta = 0.0
    if age_a_years is not None and age_b_years is not None:
        age_delta = abs(age_b_years - age_a_years)

    # ── Durations ───────────────────────────────────────────────────────────
    dur_a = _get_duration_h(findings_a) or meta_a.get("duration_h", 0.0)
    dur_b = _get_duration_h(findings_b) or meta_b.get("duration_h", 0.0)

    # If metadata carries explicit duration, prefer that
    if meta_a.get("duration_h"):
        dur_a = float(meta_a["duration_h"])
    if meta_b.get("duration_h"):
        dur_b = float(meta_b["duration_h"])

    # Duration compatibility: both should be either long-form or both short
    long_a = dur_a >= _LONG_FORM_H
    long_b = dur_b >= _LONG_FORM_H
    duration_compatible = (long_a == long_b)

    # Build methodology warning
    if not duration_compatible:
        dur_a_str = f"{dur_a:.1f}h" if dur_a >= 1.0 else f"{round(dur_a*60)}min"
        dur_b_str = f"{dur_b:.1f}h" if dur_b >= 1.0 else f"{round(dur_b*60)}min"
        methodology_warning = (
            f"Recording durations differ substantially: {dur_a_str} (A) vs "
            f"{dur_b_str} (B). Spike rates have been normalized to events/min, "
            f"but sampling window bias remains. For a fair comparison, repeat "
            f"with matched recording conditions."
        )
    else:
        methodology_warning = ""

    # ── Per-channel spike rates ──────────────────────────────────────────────
    rates_a = _channel_spike_rates(findings_a)
    rates_b = _channel_spike_rates(findings_b)

    # Build combined per-channel dict for all channels present in either
    all_channels = sorted(set(list(rates_a.keys()) + list(rates_b.keys())))
    spike_rate_per_channel: dict[str, tuple[float, float]] = {
        ch: (rates_a.get(ch, 0.0), rates_b.get(ch, 0.0))
        for ch in all_channels
    }

    mean_a = _mean_rate(rates_a)
    mean_b = _mean_rate(rates_b)
    if mean_a > 0:
        mean_spike_rate_delta_pct = (mean_b - mean_a) / mean_a * 100.0
    elif mean_b > 0:
        mean_spike_rate_delta_pct = 100.0  # went from 0 to something
    else:
        mean_spike_rate_delta_pct = 0.0

    # ── Topographic shift ────────────────────────────────────────────────────
    topographic_shift, top5_a, top5_b = _detect_topographic_shift(rates_a, rates_b)

    # ── Background / PDR ────────────────────────────────────────────────────
    bg_a = findings_a.get("background", {})
    bg_b = findings_b.get("background", {})
    pdr_a = bg_a.get("posterior_dominant_rhythm_hz") or bg_a.get("pdr_hz")
    pdr_b = bg_b.get("posterior_dominant_rhythm_hz") or bg_b.get("pdr_hz")
    if pdr_a is not None:
        pdr_a = float(pdr_a)
    if pdr_b is not None:
        pdr_b = float(pdr_b)
    pdr_delta_hz = (pdr_b - pdr_a) if (pdr_a is not None and pdr_b is not None) else None

    # ── Morphology ───────────────────────────────────────────────────────────
    morph_a = findings_a.get("morphology", {})
    morph_b = findings_b.get("morphology", {})
    complex_sw_pct_a = float(
        morph_a.get("pct_complex_spike_wave") or morph_a.get("complex_sw_pct") or 0.0
    )
    complex_sw_pct_b = float(
        morph_b.get("pct_complex_spike_wave") or morph_b.get("complex_sw_pct") or 0.0
    )

    # ── Spindles ─────────────────────────────────────────────────────────────
    sleep_dur_a = _get_sleep_duration_h(findings_a)
    sleep_dur_b = _get_sleep_duration_h(findings_b)
    both_have_sleep = (sleep_dur_a >= 2.0 and sleep_dur_b >= 2.0)

    spindle_density_a: float | None = None
    spindle_density_b: float | None = None
    spindle_delta_pct: float | None = None
    if both_have_sleep:
        sp_a = findings_a.get("spindles", {})
        sp_b = findings_b.get("spindles", {})
        d_a = sp_a.get("density_per_minute") or sp_a.get("spindle_density")
        d_b = sp_b.get("density_per_minute") or sp_b.get("spindle_density")
        if d_a is not None:
            spindle_density_a = float(d_a)
        if d_b is not None:
            spindle_density_b = float(d_b)
        if spindle_density_a is not None and spindle_density_b is not None:
            if spindle_density_a > 0:
                spindle_delta_pct = (
                    (spindle_density_b - spindle_density_a) / spindle_density_a * 100.0
                )

    # ── Confounds ────────────────────────────────────────────────────────────
    confounds = _detect_confounds(
        findings_a=findings_a,
        findings_b=findings_b,
        age_delta_years=age_delta,
        dur_a=dur_a,
        dur_b=dur_b,
        condition_a=condition_a,
        condition_b=condition_b,
        mean_rate_a=mean_a,
        mean_rate_b=mean_b,
        metadata_a=meta_a,
        metadata_b=meta_b,
    )

    # ── Interpretation hints ──────────────────────────────────────────────────
    hints = _build_interpretation_hints(
        mean_spike_rate_delta_pct=mean_spike_rate_delta_pct,
        topographic_shift=topographic_shift,
        confounds=confounds,
        duration_compatible=duration_compatible,
        pdr_delta_hz=pdr_delta_hz,
        spindle_delta_pct=spindle_delta_pct,
    )

    # ── Build output ──────────────────────────────────────────────────────────
    rec_a_meta = {
        "date": date_a,
        "label": label_a,
        "duration_h": round(dur_a, 2),
        "condition": condition_a,
        "age_years": age_a_years,
    }
    rec_b_meta = {
        "date": date_b,
        "label": label_b,
        "duration_h": round(dur_b, 2),
        "condition": condition_b,
        "age_years": age_b_years,
    }

    return LongitudinalDelta(
        recording_a=rec_a_meta,
        recording_b=rec_b_meta,
        age_delta_years=round(age_delta, 3),
        duration_compatible=duration_compatible,
        methodology_warning=methodology_warning,
        spike_rate_per_channel=spike_rate_per_channel,
        mean_spike_rate_delta_pct=round(mean_spike_rate_delta_pct, 1),
        topographic_shift=topographic_shift,
        top5_channels_a=top5_a,
        top5_channels_b=top5_b,
        pdr_a=pdr_a,
        pdr_b=pdr_b,
        pdr_delta_hz=round(pdr_delta_hz, 2) if pdr_delta_hz is not None else None,
        complex_sw_pct_a=round(complex_sw_pct_a, 2),
        complex_sw_pct_b=round(complex_sw_pct_b, 2),
        spindle_density_a=spindle_density_a,
        spindle_density_b=spindle_density_b,
        spindle_delta_pct=round(spindle_delta_pct, 1) if spindle_delta_pct is not None else None,
        confounds=confounds,
        interpretation_hints=hints,
    )
