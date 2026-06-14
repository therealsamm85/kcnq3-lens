"""C3 — Ictal (electrographic seizure) screener.  [BUILD, flag-for-review]

The tool is otherwise interictal-only; it would silently ignore an electrographic
seizure in an overnight recording. This is a sensitivity-first heuristic screener
that flags candidate evolving rhythmic runs for HUMAN review — explicitly not a
diagnosis.

BUILD (not borrow): DeepSOZ and the TUSZ/CHB-MIT DL detectors are adult-trained
on specific montages, unvalidated in pediatric ESES, and pull in torch. A
transparent screener — sustained line-length elevation (rhythmic high-amplitude
activity) combined with dominant-frequency drift (ictal evolution) over a
sustained window — is auditable, local, and appropriately humble. Confidence is
capped at "moderate": this never asserts a seizure, it asks a human to look.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..readers.base import EEGRecording

_MAX_CHANNELS = 4


@dataclass
class IctalCandidate:
    start_s: float
    duration_s: float
    channel: str
    peak_line_length_z: float
    freq_drift_hz: float
    confidence: str             # "low" | "moderate" (never "high" — screening only)


@dataclass
class IctalScreenResult:
    n_candidates: int
    candidates: list[IctalCandidate] = field(default_factory=list)
    minutes_screened: float = 0.0
    channels_screened: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    caveat: str = ""


def _dom_freq(w: np.ndarray, sf: float, lo: float = 2.0, hi: float = 25.0) -> float:
    w = w - w.mean()
    if np.allclose(w, 0.0):
        return 0.0
    spec = np.abs(np.fft.rfft(w * np.hanning(len(w))))
    freqs = np.fft.rfftfreq(len(w), 1.0 / sf)
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any():
        return 0.0
    return float(freqs[band][int(np.argmax(spec[band]))])


def _window_features(rec, ch_idx, sf, win_s, step_s):
    """Yield (abs_time_s, line_length, dom_freq) per sliding window."""
    nwin = max(2, int(win_s * sf))
    nstep = max(1, int(step_s * sf))
    out = []
    for ep, d in rec.iter_epochs(epoch_seconds=30.0):
        if not (0 <= ch_idx < d.shape[0]):
            continue
        sig = np.asarray(d[ch_idx], dtype=float)
        ep_t0 = ep * 30.0
        for s in range(0, sig.size - nwin + 1, nstep):
            w = sig[s: s + nwin]
            ll = float(np.sum(np.abs(np.diff(w))))
            out.append((ep_t0 + s / sf, ll, _dom_freq(w, sf)))
    return out


def screen_ictal(
    rec: EEGRecording,
    target_channels: list[str] | None = None,
    win_s: float = 2.0,
    step_s: float = 1.0,
    min_event_s: float = 10.0,
    ll_z_thresh: float = 3.0,
    freq_drift_min_hz: float = 1.0,
) -> IctalScreenResult:
    """Flag candidate electrographic seizures (rhythmic, evolving, sustained)."""
    # Resolve up to a few live channels to screen.
    ch_pairs: list[tuple[int, str]] = []
    if target_channels:
        for nm in target_channels:
            i = rec.channel_index(nm)
            if i is not None:
                ch_pairs.append((i, rec.channel_names[i]))
    else:
        for i in (rec.eeg_channel_indices or list(range(rec.n_channels_in_file))):
            if rec.is_channel_live(i) if hasattr(rec, "is_channel_live") else True:
                ch_pairs.append((i, rec.channel_names[i]))
            if len(ch_pairs) >= _MAX_CHANNELS:
                break
    if not ch_pairs:
        return IctalScreenResult(0, notes=["no usable channels to screen"])

    sf = float(rec.sfreq)
    all_candidates: list[IctalCandidate] = []
    for ch_idx, ch_name in ch_pairs:
        wins = _window_features(rec, ch_idx, sf, win_s, step_s)
        if len(wins) < 3:
            continue
        lls = np.array([w[1] for w in wins])
        med = float(np.median(lls))
        mad = float(np.median(np.abs(lls - med)))
        scale = 1.4826 * mad if mad > 0 else (float(np.std(lls)) or 1.0)
        z = (lls - med) / scale

        # Group consecutive above-threshold windows into runs.
        i = 0
        nstep_gap = step_s * 1.5
        while i < len(wins):
            if z[i] < ll_z_thresh:
                i += 1
                continue
            j = i
            while (j + 1 < len(wins) and z[j + 1] >= ll_z_thresh
                   and wins[j + 1][0] - wins[j][0] <= nstep_gap):
                j += 1
            t_start = wins[i][0]
            t_end = wins[j][0] + win_s
            duration = t_end - t_start
            if duration >= min_event_s:
                drift = abs(wins[j][2] - wins[i][2])
                peak_z = float(np.max(z[i:j + 1]))
                conf = ("moderate" if drift >= freq_drift_min_hz else "low")
                all_candidates.append(IctalCandidate(
                    start_s=round(t_start, 1), duration_s=round(duration, 1),
                    channel=ch_name, peak_line_length_z=round(peak_z, 1),
                    freq_drift_hz=round(drift, 1), confidence=conf,
                ))
            i = j + 1

    # Merge time-overlapping candidates across channels (keep the strongest).
    all_candidates.sort(key=lambda c: c.start_s)
    merged: list[IctalCandidate] = []
    for c in all_candidates:
        if merged and c.start_s <= merged[-1].start_s + merged[-1].duration_s:
            if c.peak_line_length_z > merged[-1].peak_line_length_z:
                merged[-1] = c
        else:
            merged.append(c)

    notes = [
        f"screened {len(ch_pairs)} channel(s); sliding {win_s:g}s/{step_s:g}s windows. "
        "Windows do not cross 30 s epoch seams, so a seizure straddling a seam may "
        "split into two flags.",
    ]
    caveat = (
        "SCREENING ONLY — sensitivity-first heuristic (line-length + frequency "
        "drift). It does not diagnose seizures; every candidate must be confirmed "
        "by a human reading the trace. Rhythmic artifact (chewing, movement) can "
        "trigger a flag. Confidence is capped at 'moderate' by design."
    )
    return IctalScreenResult(
        n_candidates=len(merged), candidates=merged,
        # Report time ACTUALLY screened (whole 30 s epochs processed), not the
        # nominal recording length — a sub-30 s clip or un-screened tail is not
        # claimed as covered.
        minutes_screened=round(rec.n_epochs * 30 / 60.0, 1),
        channels_screened=[nm for _i, nm in ch_pairs],
        notes=notes, caveat=caveat,
    )


def summarize_ictal(result: IctalScreenResult) -> dict:
    return {
        "n_candidates": result.n_candidates,
        "minutes_screened": result.minutes_screened,
        "channels_screened": result.channels_screened,
        "candidates": [
            {
                "start_s": c.start_s, "duration_s": c.duration_s,
                "channel": c.channel, "peak_line_length_z": c.peak_line_length_z,
                "freq_drift_hz": c.freq_drift_hz, "confidence": c.confidence,
            }
            for c in result.candidates
        ],
        "notes": result.notes,
        "caveat": result.caveat,
    }
