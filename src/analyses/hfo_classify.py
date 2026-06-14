"""C2 — Two-stage HFO classification.  [BUILD on existing HFO events]

The HFO detector finds candidate ripples but does not say which are real vs
artifact, or which co-occur with a spike (spkHFO) — the distinction that cuts
the false-positive burden plaguing scalp HFOs. This adds a transparent
feature-based second stage on the events already in ``_hfo_ripples_events``.

BUILD (not borrow): PyHFO's deep classifiers need torch + a UCLA academic
licence — too heavy / wrong licence for a local family tool. The interpretable
features are computed directly from each event's metadata (oscillation-cycle
count from peak frequency × duration, amplitude sanity, in-band check, and
spike-time coincidence), so the decision is fully auditable.

It flags the spike-coupled subset (spkHFO) — the actionable one — but does NOT
claim "epileptogenic" (eHFO): that distinction is validated intracranially, not
on the scalp.
"""
from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field


@dataclass
class HfoClassifyResult:
    n_input: int
    n_artifact: int
    n_real: int                 # genuine HFOs (includes the spike-coupled subset)
    n_spike_coupled: int        # spkHFO subset of n_real
    per_event: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _event_peak(ev: dict) -> float | None:
    for k in ("peak_s", "time_s", "start_s"):
        if ev.get(k) is not None:
            try:
                return float(ev[k])
            except (TypeError, ValueError):
                return None
    return None


def _nearest_within(sorted_times: list[float], t: float, tol: float) -> bool:
    """True if any time in the sorted list is within tol seconds of t."""
    if not sorted_times:
        return False
    i = bisect_left(sorted_times, t)
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(sorted_times) and abs(sorted_times[j] - t) <= tol + 1e-9:
            return True
    return False


def classify_hfos(
    hfo_events: list[dict] | None,
    spike_events: list[dict] | None = None,
    min_cycles: float = 4.0,
    band_hz: tuple[float, float] = (80.0, 500.0),
    max_rms_z: float = 30.0,
    coupling_window_ms: float = 50.0,
) -> HfoClassifyResult:
    """Classify each HFO candidate as artifact / real / real-spike-coupled.

    A genuine HFO is an oscillation (≥ ``min_cycles`` cycles = peak_freq ×
    duration) within the ripple band and of sane amplitude; a too-short / too-
    sharp / out-of-band / extreme-amplitude candidate is an artifact. A real HFO
    that co-occurs with a spike (per the detector flag, or within
    ``coupling_window_ms`` of a supplied spike time) is a spkHFO.
    """
    events = hfo_events or []
    spikes = sorted(
        float(ev["time_s"]) for ev in (spike_events or [])
        if isinstance(ev, dict) and ev.get("time_s") is not None
    )
    tol = coupling_window_ms / 1000.0
    lo, hi = band_hz

    per_event: list[dict] = []
    n_art = n_real = n_spk = 0
    for ev in events:
        pf = ev.get("peak_freq_hz")
        dur = ev.get("duration_ms")
        rms_z = ev.get("rms_z", 0.0) or 0.0
        peak = _event_peak(ev)
        # Explicit None + finite checks (NaN is truthy in Python, so the old
        # `if (pf and dur)` and `n_cycles < min_cycles` let NaN/0 slip through
        # and be mislabeled a genuine HFO).
        pf_ok = pf is not None and math.isfinite(float(pf))
        dur_ok = dur is not None and math.isfinite(float(dur)) and float(dur) > 0
        n_cycles = (float(pf) * float(dur) / 1000.0) if (pf_ok and dur_ok) else None

        if not pf_ok or not (lo <= float(pf) <= hi):
            cls, reason = "artifact", "peak frequency outside the HFO band or invalid"
        elif not dur_ok:
            cls, reason = "artifact", "missing/invalid duration"
        elif n_cycles < min_cycles:
            cls, reason = "artifact", f"<{min_cycles:g} oscillation cycles (transient/blip)"
        elif (not math.isfinite(float(rms_z))) or abs(float(rms_z)) > max_rms_z:
            cls, reason = "artifact", "extreme/invalid amplitude (likely electrode pop)"
        else:
            cls, reason = "real", "oscillatory and in-band"

        coupled = False
        if cls == "real":
            coupled = bool(ev.get("co_occurs_with_spike"))
            if not coupled and peak is not None:
                coupled = _nearest_within(spikes, peak, tol)
            if coupled:
                cls = "real_spike_coupled"

        if cls == "artifact":
            n_art += 1
        else:
            n_real += 1
            if cls == "real_spike_coupled":
                n_spk += 1

        per_event.append({
            "peak_s": peak,
            "peak_freq_hz": pf,
            "n_cycles": round(n_cycles, 2) if n_cycles is not None else None,
            "classification": cls,
            "reason": reason,
        })

    notes: list[str] = []
    if events:
        notes.append(f"{n_art}/{len(events)} candidates rejected as artifact; "
                     f"{n_spk} of {n_real} genuine HFOs co-occur with a spike (spkHFO).")
    else:
        notes.append("no HFO candidates supplied.")
    notes.append("spkHFO is flagged as the actionable subset; 'epileptogenic' (eHFO) "
                 "is NOT claimed — that is validated intracranially, not on the scalp.")

    return HfoClassifyResult(
        n_input=len(events), n_artifact=n_art, n_real=n_real,
        n_spike_coupled=n_spk, per_event=per_event, notes=notes,
    )


def summarize_hfo_classify(result: HfoClassifyResult) -> dict:
    return {
        "n_input": result.n_input,
        "n_artifact": result.n_artifact,
        "n_real": result.n_real,
        "n_spike_coupled": result.n_spike_coupled,
        "notes": result.notes,
    }
