"""A1 — Annotated EDF+ write-back.  [BORROW: edfio, already in stack]

Export the recording's signal plus the tool's detected events (spikes, SWI
bursts, HFOs, sleep stages) as an EDF+ that any neurologist can open in free
EDFbrowser / Persyst / a clinical reviewer — no proprietary platform. edfio is
already a dependency, so this is wiring, not a new dep.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..readers.base import EEGRecording

# findings key → human label for the annotation track.
_EVENT_KEY_LABELS = {
    "_morphology_events": "spike",
    "_slow_waves_events": "slow-wave",
    "_spindle_events": "spindle",
    "_hfo_ripples_events": "HFO",
    "_sharp_spikes_events": "sharp-spike",
}


@dataclass
class EdfWritebackResult:
    out_path: str
    n_annotations: int
    channels_written: int
    duration_s: float
    note: str = ""
    notes: list[str] = field(default_factory=list)


def _event_onset(ev: dict) -> float | None:
    """Pull a seconds-from-start onset from an event dict, tolerant of keys."""
    for k in ("time_s", "onset_s", "onset", "start_s"):
        if k in ev and ev[k] is not None:
            try:
                return float(ev[k])
            except (TypeError, ValueError):
                return None
    return None


def collect_events_from_findings(findings: dict) -> list[dict]:
    """Flatten the findings' ``_*_events`` lists into labeled annotation dicts.

    Returns dicts shaped {onset_s, duration_s, label} ready for annotation. A
    convenience so callers can hand the whole findings dict straight in.
    """
    out: list[dict] = []
    for key, label in _EVENT_KEY_LABELS.items():
        for ev in findings.get(key, []) or []:
            onset = _event_onset(ev)
            if onset is None:
                continue
            text = ev.get("label") or label
            ch = ev.get("channel")
            if ch:
                text = f"{text} {ch}"
            out.append({
                "onset_s": onset,
                "duration_s": float(ev.get("duration_s", 0.0) or 0.0),
                "label": text,
            })
    out.sort(key=lambda e: e["onset_s"])
    return out


def export_annotated_edf(
    rec: EEGRecording,
    out_path: str | Path,
    events: list[dict] | None = None,
    *,
    anonymize: bool = True,
    eeg_only: bool = True,
) -> EdfWritebackResult:
    """Write an EDF+ with the recording signal + `events` as annotations.

    Parameters
    ----------
    events : list of dicts with an onset (``onset_s``/``time_s``), optional
        ``duration_s`` and ``label``. Use ``collect_events_from_findings`` to
        build this from a findings dict.
    anonymize : if True (default) no patient/recording identity is written.
    eeg_only : write only the real EEG channels (skip aux/marker channels).

    Memory note: EDF needs the full signal in memory; for multi-hour recordings
    this can be large. The signal is read once via the lazy epoch interface.
    """
    import edfio  # already a dependency (MNE's EDF export backend)

    out_path = Path(out_path)
    notes: list[str] = []

    ch_indices = rec.eeg_channel_indices if eeg_only else list(range(rec.n_channels_in_file))
    if not ch_indices:
        ch_indices = list(range(rec.n_channels_in_file))
        notes.append("no EEG channels flagged — wrote all channels")

    # Concatenate the signal per channel via the lazy epoch interface.
    per_ch: list[list[np.ndarray]] = [[] for _ in ch_indices]
    for _ep, d in rec.iter_epochs(epoch_seconds=30.0):
        for j, ci in enumerate(ch_indices):
            if 0 <= ci < d.shape[0]:
                per_ch[j].append(np.asarray(d[ci], dtype=np.float64))
    data = [np.concatenate(chunks) if chunks else np.zeros(0) for chunks in per_ch]

    n_samples = min((arr.size for arr in data), default=0)
    sf = float(rec.sfreq)
    # EDF needs an integer number of samples per (1 s) data record, so trim to
    # whole seconds and an integer sample rate.
    samples_per_record = int(round(sf))
    if samples_per_record <= 0:
        raise ValueError("recording sample rate is not a positive integer")
    whole_seconds = n_samples // samples_per_record
    keep = whole_seconds * samples_per_record
    if keep <= 0:
        raise ValueError("recording too short to write a 1-second-record EDF+")
    if keep < n_samples:
        notes.append(f"trimmed {n_samples - keep} trailing samples to whole seconds")

    signals = []
    for ci, arr in zip(ch_indices, data):
        sig = np.asarray(arr[:keep], dtype=np.float64)
        # edfio rejects a zero-width physical range (a dead/flat channel); widen it.
        lo, hi = float(np.min(sig)), float(np.max(sig))
        if hi - lo < 1e-6:
            lo, hi = lo - 1.0, hi + 1.0
        label = str(rec.channel_names[ci])[:16]
        # EDF needs an integer number of samples per 1 s data record, so declare
        # the (rounded) integer rate that matches the whole-second trim — passing
        # a fractional rate makes edfio reject the write.
        signals.append(edfio.EdfSignal(
            sig, sampling_frequency=float(samples_per_record), label=label,
            physical_dimension="uV", physical_range=(lo, hi),
        ))

    if float(samples_per_record) != sf:
        notes.append(f"sample rate {sf:g} Hz rounded to {samples_per_record} Hz "
                     "for the integer-rate EDF — timing is approximate.")

    annotations = []
    n_ann = 0
    max_onset = keep / sf
    for ev in (events or []):
        # Tolerant onset extraction (handles onset_s/time_s/... and present-but-
        # None without raising); reject non-finite / out-of-range onsets.
        onset = _event_onset(ev)
        if onset is None or not np.isfinite(onset) or onset < 0 or onset > max_onset:
            continue
        dur = float(ev.get("duration_s", 0.0) or 0.0)
        if not np.isfinite(dur) or dur < 0:
            dur = 0.0                       # clamp negative/non-finite duration
        annotations.append(edfio.EdfAnnotation(
            onset=float(onset), duration=dur,
            text=str(ev.get("label", "event"))[:40],
        ))
        n_ann += 1

    edf = edfio.Edf(
        signals=signals,
        data_record_duration=1.0,
        annotations=annotations or None,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edf.write(out_path)

    return EdfWritebackResult(
        out_path=str(out_path),
        n_annotations=n_ann,
        channels_written=len(signals),
        duration_s=round(keep / sf, 2),
        note=("anonymized; " if anonymize else "") + "open in EDFbrowser or any clinical reviewer",
        notes=notes,
    )
