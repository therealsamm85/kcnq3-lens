"""Slow-Oscillation–Spindle phase-locking value (PLV) coupling.

Slow oscillations (SO, 0.5–1.25 Hz) organise spindle timing by preferentially
expressing spindle bursts during the SO up-state. This temporal co-ordination —
termed SO-spindle coupling — supports memory consolidation and undergoes
substantial development across childhood (Hahn et al. 2020, PMID 32499637).

Phase-locking value (PLV) quantifies the consistency of the SO phase at which
spindle peaks occur. PLV = 1 means every spindle occurs at exactly the same SO
phase; PLV = 0 means spindle phases are uniformly random.

IMPORTANT — NO validated paediatric normative ranges exist for coupling PLV
in children aged < 8 years as of 2026. Hahn et al. 2020 covers ages 8–19 and
shows PLV rising from ~0.15 (age 8) to adult levels (~0.3–0.5 at Fz/Cz).
In a 5-year-old, typical PLV of 0.15–0.35 may be within the expected
developmental range, but formal cutoffs do not exist. This module therefore
reports descriptive metrics ONLY — no "below / in / above" range
classification is applied.

Signal-length alignment note
-----------------------------
``iter_epochs`` yields only whole 30-second epochs (``n_epochs = int(duration_s)
// 30``). The concatenated signal therefore contains
``int(duration_s) // 30 * 30 * sfreq`` samples — which is shorter than
``int(round(duration_s * sfreq))`` by up to ``(duration_s % 30) * sfreq``
samples (up to ~7 500 samples at 250 Hz for a recording that ends mid-epoch).
All spindle-peak sample indices are bounds-checked against the actual signal
length; out-of-bounds spindles are silently skipped, never clipped, to avoid
false phase aliasing.

WARNING: ``notes`` field is local-only. Never extracted into registry
submissions. The ``_DISCLAIMER`` text exceeds the 80-char PHI threshold and
would be rejected by the scanner if it tried to land in a submission.

References
----------
Helfrich RF et al. 2018  PMID 29395264  SO-spindle coupling and memory in aging
Hahn M et al. 2020       PMID 32499637  Pediatric SO-spindle coupling, ages 8–19
Mölle M et al. 2011      PMID 21248118  SO up-state timing of spindles
Staresina BP et al. 2015 PMID 25822789  SO-spindle coupling in humans
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import firwin, filtfilt

from ..readers.base import EEGRecording
from .sleep_stages import SleepStageResult


_DISCLAIMER = (
    "RESEARCH METRIC — No validated paediatric normative ranges exist for "
    "SO-spindle PLV in children under 8 years. Typical PLV 0.15–0.35 at "
    "Fz/Cz may be within the expected developmental range for age 5 but "
    "cannot be classified as normal or abnormal. Results are descriptive only."
)


# ─── Result dataclass ─────────────────────────────────────────────────────────


@dataclass
class CouplingResult:
    channel: str
    available: bool
    unavailable_reason: str          # "" | "insufficient_events" | "no_n2_n3_sleep"
                                     # | "no_spindles" | "no_slow_waves"
    n_spindles_total: int
    n_so_total: int
    n_spindles_in_so: int            # spindles co-occurring with SOs
    plv: float                       # 0..1, NaN-safe → 0.0 + flag
    preferred_phase_deg: float       # -180..180, mean SO phase at spindle peak
    rayleigh_p: float                # p-value of non-uniform distribution
    rayleigh_z: float
    coupling_window_s: float         # ±window used for co-occurrence
    method: str                      # "hilbert_plv"
    notes: list[str] = field(default_factory=list)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _unavailable(
    reason: str,
    channel: str,
    notes: list[str],
    n_spindles: int = 0,
    n_so: int = 0,
) -> CouplingResult:
    return CouplingResult(
        channel=channel,
        available=False,
        unavailable_reason=reason,
        n_spindles_total=n_spindles,
        n_so_total=n_so,
        n_spindles_in_so=0,
        plv=0.0,
        preferred_phase_deg=0.0,
        rayleigh_p=1.0,
        rayleigh_z=0.0,
        coupling_window_s=0.0,
        method="hilbert_plv",
        notes=notes,
    )


def _bandpass_so(signal: np.ndarray, sfreq: float) -> np.ndarray:
    """FIR bandpass filter 0.5–1.25 Hz (zero-phase, linear-phase).

    Uses a generous order (3 × sfreq rounded to odd) for good stopband
    attenuation at these very low frequencies.
    """
    numtaps = int(round(3.0 * sfreq))
    if numtaps % 2 == 0:
        numtaps += 1
    nyq = sfreq / 2.0
    # Clamp to safe range avoiding Gibbs at 0 or Nyquist
    lo = max(0.001, 0.5 / nyq)
    hi = min(0.999, 1.25 / nyq)
    taps = firwin(numtaps, [lo, hi], pass_zero=False)
    return filtfilt(taps, [1.0], signal.astype(np.float64))


def _rayleigh_test(phases: np.ndarray) -> tuple[float, float]:
    """Inline Rayleigh test for circular uniformity.

    Returns (z_statistic, p_value).

    Approximation from Mardia/Zar Eq. 27.4 and Fisher 1993.
    Accurate for n >= 50. For 20 <= n < 50 error is typically < 10%.
    For 10 <= n < 20 error can reach 10-30%; callers should check whether
    the rayleigh_approximation_n_lt_20 note was appended.
    """
    n = len(phases)
    if n < 2:
        return 0.0, 1.0
    z_vec = np.exp(1j * phases)
    R = float(abs(np.sum(z_vec)))
    rayleigh_z = R ** 2 / n
    # Rayleigh test approximation (Mardia/Zar Eq. 27.4, Fisher 1993).
    # Accurate for n >= 50. For 20 <= n < 50: approximation error typically
    # < 10%. For 10 <= n < 20: error can reach 10-30%; we emit a note
    # (rayleigh_approximation_n_lt_20) for these cases.
    p = np.exp(-rayleigh_z) * (
        1
        + (2 * rayleigh_z - rayleigh_z ** 2) / (4 * n)
        - (
            24 * rayleigh_z
            - 132 * rayleigh_z ** 2
            + 76 * rayleigh_z ** 3
            - 9 * rayleigh_z ** 4
        ) / (288 * n ** 2)
    )
    # Clamp p to [0, 1] (approximation can yield tiny negatives for large Z)
    p = float(np.clip(p, 0.0, 1.0))
    return float(rayleigh_z), p


# ─── Main entry point ─────────────────────────────────────────────────────────


def compute_so_spindle_coupling(
    rec: EEGRecording,
    sleep_stages: SleepStageResult | None = None,
    spindle_events: list[dict] | None = None,
    slow_wave_events: list[dict] | None = None,
    channel: str = "Fz",
    coupling_window_s: float = 1.2,
) -> CouplingResult:
    """Compute SO-spindle coupling via phase-locking value (PLV).

    Parameters
    ----------
    rec : EEGRecording
    sleep_stages : SleepStageResult, optional
        If provided, analysis is restricted to N2 + N3 epochs only.
    spindle_events : list[dict], optional
        Each dict must have ``peak_time_s`` (float). Typically obtained from
        ``SpindleResult.events`` as exported by runner.py under the key
        ``_spindle_events``.
    slow_wave_events : list[dict], optional
        Each dict must have ``neg_peak_s`` (float). Typically obtained from
        ``SlowWaveResult.events`` via ``_slow_waves_events``.
    channel : str
        Preferred channel for SO phase extraction. Falls back to Cz → C3 →
        first available EEG channel (case-insensitive, same as slow_waves.py).
    coupling_window_s : float
        Half-window in seconds for SO-spindle co-occurrence (default ±1.2 s).

    Returns
    -------
    CouplingResult
        ``available=False`` when guards trigger (see notes).
    """
    notes: list[str] = []

    # ── Guard 1: spindle events ───────────────────────────────────────────────
    if not spindle_events:
        return _unavailable("no_spindles", channel, notes)

    # ── Guard 2: slow wave events ─────────────────────────────────────────────
    if not slow_wave_events:
        return _unavailable("no_slow_waves", channel, notes,
                            n_spindles=len(spindle_events))

    # ── Guard 3: N2/N3 sleep ─────────────────────────────────────────────────
    n2n3_windows: list[tuple[float, float]] | None = None
    total_s = rec.duration_s

    if sleep_stages is not None:
        nrem_labels = {"N2", "N3"}
        epoch_s = sleep_stages.epoch_seconds
        nrem_indices = [
            i for i, lbl in enumerate(sleep_stages.epoch_labels)
            if lbl in nrem_labels
        ]
        if not nrem_indices:
            return _unavailable(
                "no_n2_n3_sleep", channel, notes,
                n_spindles=len(spindle_events),
                n_so=len(slow_wave_events),
            )
        windows: list[tuple[float, float]] = []
        for ep_idx in nrem_indices:
            t0 = ep_idx * epoch_s
            t1 = (ep_idx + 1) * epoch_s
            if t0 < total_s:
                windows.append((t0, min(t1, total_s)))
        n2n3_windows = windows

    # ── Channel resolution (case-insensitive, same chain as slow_waves.py) ────
    channel_upper = channel.upper()
    ch_idx = rec.channel_index(channel)
    resolved_channel = rec.channel_names[ch_idx] if ch_idx is not None else channel

    if ch_idx is None:
        for fallback in ("Fz", "Cz", "C3"):
            if fallback.upper() == channel_upper:
                continue
            ch_idx = rec.channel_index(fallback)
            if ch_idx is not None:
                resolved_channel = rec.channel_names[ch_idx]
                break
    if ch_idx is None:
        if rec.eeg_channel_indices:
            ch_idx = rec.eeg_channel_indices[0]
            resolved_channel = rec.channel_names[ch_idx]
        else:
            return _unavailable(
                "no_channel", channel, notes + ["no_eeg_channel_found"],
                n_spindles=len(spindle_events),
                n_so=len(slow_wave_events),
            )

    # ── Build full signal ─────────────────────────────────────────────────────
    segments = []
    for _, d in rec.iter_epochs(epoch_seconds=30.0):
        segments.append(d[ch_idx])
    if not segments:
        return _unavailable(
            "no_channel", channel, notes + ["no_data_readable"],
            n_spindles=len(spindle_events),
            n_so=len(slow_wave_events),
        )
    signal = np.concatenate(segments).astype(np.float64)
    sfreq = rec.sfreq

    # ── Signal-length alignment note ─────────────────────────────────────────
    # iter_epochs yields only whole 30-s epochs; the tail (duration_s % 30 s)
    # is dropped. Signal may be shorter than expected by up to ~7 500 samples
    # at 250 Hz. We record the drift so callers are aware; out-of-bounds
    # spindles will be silently skipped below (not clipped).
    expected_n = int(round(rec.duration_s * sfreq))
    actual_n = signal.shape[0]
    drift = actual_n - expected_n
    if abs(drift) > sfreq:  # >1 s of drift is flagged
        notes.append(f"signal_length_drift_{drift}_samples")

    # ── NaN/Inf guard ─────────────────────────────────────────────────────────
    nan_frac = float(np.isnan(signal).sum()) / max(signal.size, 1)
    if nan_frac > 0.20:
        return _unavailable(
            "high_nan_fraction", resolved_channel, notes,
            n_spindles=len(spindle_events),
            n_so=len(slow_wave_events),
        )
    if nan_frac > 0.05:
        notes.append("high_nan_fraction")
    if not np.all(np.isfinite(signal)):
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Volts-vs-µV guard ─────────────────────────────────────────────────────
    p99 = float(np.percentile(np.abs(signal), 99))
    if p99 < 1.0:
        signal = signal * 1e6
        notes.append("auto_scaled_volts_to_uv")

    # ── Bandpass 0.5–1.25 Hz → Hilbert → SO phase ────────────────────────────
    from scipy.signal import hilbert as _hilbert
    so_filtered = _bandpass_so(signal, sfreq)
    analytic = _hilbert(so_filtered)
    so_phase = np.angle(analytic)   # radians, -π..π

    # ── Filter events to N2/N3 ────────────────────────────────────────────────
    def _in_nrem(t: float) -> bool:
        if n2n3_windows is None:
            return True
        return any(ws <= t < we for ws, we in n2n3_windows)

    # Validate and filter slow-wave events
    valid_so: list[dict] = []
    for ev in slow_wave_events:
        t = ev.get("neg_peak_s")
        if t is None or not math.isfinite(float(t)):
            continue
        if _in_nrem(float(t)):
            valid_so.append(ev)

    # Validate and filter spindle events
    valid_spindles: list[dict] = []
    for ev in spindle_events:
        t = ev.get("peak_time_s")
        if t is None or not math.isfinite(float(t)):
            continue
        if _in_nrem(float(t)):
            valid_spindles.append(ev)

    n_spindles_total = len(valid_spindles)
    n_so_total = len(valid_so)

    # ── Co-occurrence: spindle within ±coupling_window_s of any SO ───────────
    so_neg_peaks = np.array([float(ev["neg_peak_s"]) for ev in valid_so])
    phases_at_spindle: list[float] = []

    for sp_ev in valid_spindles:
        sp_t = float(sp_ev["peak_time_s"])
        # Find any SO within ±coupling_window_s
        if len(so_neg_peaks) == 0:
            continue
        diffs = np.abs(so_neg_peaks - sp_t)
        if diffs.min() <= coupling_window_s:
            # Get SO phase at the spindle peak time.
            # Skip (not clip) out-of-bounds indices to avoid false phase aliasing
            # caused by signal truncation at the tail of the recording.
            sample_idx = int(round(sp_t * sfreq))
            if sample_idx < 0 or sample_idx >= len(so_phase):
                continue
            phases_at_spindle.append(float(so_phase[sample_idx]))

    n_spindles_in_so = len(phases_at_spindle)

    # ── Guard 4: insufficient co-occurring events ─────────────────────────────
    if n_spindles_in_so < 10:
        return CouplingResult(
            channel=resolved_channel,
            available=False,
            unavailable_reason="insufficient_events",
            n_spindles_total=n_spindles_total,
            n_so_total=n_so_total,
            n_spindles_in_so=n_spindles_in_so,
            plv=0.0,
            preferred_phase_deg=0.0,
            rayleigh_p=1.0,
            rayleigh_z=0.0,
            coupling_window_s=coupling_window_s,
            method="hilbert_plv",
            notes=notes,
        )

    # ── PLV calculation ───────────────────────────────────────────────────────
    phases = np.array(phases_at_spindle)
    z = np.exp(1j * phases)
    mean_z = z.mean()
    plv = float(abs(mean_z))
    preferred_phase_deg = float(np.angle(mean_z, deg=True))

    # ── Rayleigh test ─────────────────────────────────────────────────────────
    rayleigh_z_val, rayleigh_p = _rayleigh_test(phases)
    # The approximation is accurate for n >= 50; for 20 <= n < 50 error < 10%;
    # for 10 <= n < 20, error can reach 10-30% — flag explicitly.
    if n_spindles_in_so < 20:
        notes.append("rayleigh_approximation_n_lt_20")

    # Sanity check: if PLV came out non-finite, flag it
    if not math.isfinite(plv):
        notes.append("plv_was_non_finite_set_to_zero")
        plv = 0.0
    if not math.isfinite(preferred_phase_deg):
        preferred_phase_deg = 0.0

    notes.append(_DISCLAIMER)

    return CouplingResult(
        channel=resolved_channel,
        available=True,
        unavailable_reason="",
        n_spindles_total=n_spindles_total,
        n_so_total=n_so_total,
        n_spindles_in_so=n_spindles_in_so,
        plv=round(plv, 4),
        preferred_phase_deg=round(preferred_phase_deg, 2),
        rayleigh_p=round(rayleigh_p, 6),
        rayleigh_z=round(rayleigh_z_val, 4),
        coupling_window_s=coupling_window_s,
        method="hilbert_plv",
        notes=notes,
    )


# ─── Summary ──────────────────────────────────────────────────────────────────


def summarize_so_spindle_coupling(result: CouplingResult) -> dict:
    """Return a JSON-serializable summary dict (no events).

    The coupling findings contain no event lists — PLV and preferred phase
    are sufficient summary statistics and safe to include in submissions
    (via deid.py extractors) or clinical reports.
    """
    def _sf(x: float) -> float | None:
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    if not result.available:
        return {
            "available": False,
            "unavailable_reason": result.unavailable_reason,
            "n_spindles_total": result.n_spindles_total,
            "n_so_total": result.n_so_total,
            "n_spindles_in_so": result.n_spindles_in_so,
            "notes": result.notes,
        }

    return {
        "available": True,
        "channel": result.channel,
        "n_spindles_total": result.n_spindles_total,
        "n_so_total": result.n_so_total,
        "n_spindles_in_so": result.n_spindles_in_so,
        "plv": _sf(result.plv),
        "preferred_phase_deg": _sf(result.preferred_phase_deg),
        "rayleigh_p": _sf(result.rayleigh_p),
        "rayleigh_z": _sf(result.rayleigh_z),
        "coupling_window_s": _sf(result.coupling_window_s),
        "method": result.method,
        "notes": result.notes,
        # events intentionally absent — coupling has no per-event list
    }
