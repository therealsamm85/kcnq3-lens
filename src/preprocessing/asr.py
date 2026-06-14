"""B2 — Artifact Subspace Reconstruction (ASR).  [BORROW: asrpy, optional]

ASR corrects transient high-amplitude bursts (movement, electrode pops) by
reconstructing them from a clean calibration subspace — it *salvages* segments
instead of dropping them, valuable in fidgety pediatric recordings, and is
complementary to the epoch rejection already present.

Backends:
* ``asr`` — asrpy (maintained MNE-native port of EEGLAB clean_rawdata), the true
  subspace reconstruction. Preferred when installed (optional dep).
* ``burst_limiter`` — a transparent fallback when asrpy is absent: per-channel
  robust winsorization (clip beyond cutoff × robust-SD calibrated on a clean
  segment). This is NOT subspace ASR — it limits bursts rather than
  reconstructing them — but it salvages high-amplitude segments locally and is
  fully auditable. The fallback is conservative because aggressive correction can
  distort genuine epileptiform transients; the result warns to re-check spike
  morphology on the corrected signal.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from ..readers.base import EEGRecording
from ..utils.trace_viewer import read_trace_window


@dataclass
class AsrResult:
    available: bool
    cutoff: float = 0.0
    fraction_corrected: float = 0.0
    backend: str = ""
    cleaned_recording: EEGRecording | None = field(default=None, repr=False)
    notes: list[str] = field(default_factory=list)


def _eeg_only_rec(rec: EEGRecording, names: list[str], data_uv: np.ndarray) -> EEGRecording:
    return dataclasses.replace(
        rec, channel_names=list(names), n_channels=len(names),
        n_channels_in_file=len(names), eeg_channel_indices=list(range(len(names))),
        duration_s=data_uv.shape[1] / float(rec.sfreq),
        _full_data=data_uv.astype(np.float32), _read_epoch_fn=None,
    )


def run_asr(
    rec: EEGRecording,
    cutoff: float = 20.0,
    calibration_seconds: float = 60.0,
    max_seconds: float = 600.0,
) -> AsrResult:
    """Apply ASR burst correction; return a cleaned recording.

    cutoff : rejection aggressiveness (SD units; asrpy default 20). Lower =
        more aggressive.
    """
    eeg_idx = rec.eeg_channel_indices or list(range(rec.n_channels_in_file))
    names = [str(rec.channel_names[i]) for i in eeg_idx]
    if len(names) < 1:
        return AsrResult(available=False, notes=["no EEG channels"])
    secs = min(rec.duration_s, max_seconds)
    _n, _t, data = read_trace_window(rec, 0.0, secs, channels=names)  # (n_ch, n_samp) µV
    if data.shape[1] < int(rec.sfreq):
        return AsrResult(available=False, notes=["too little data for ASR"])
    sf = float(rec.sfreq)
    notes: list[str] = []

    try:
        import asrpy
        import mne
        info = mne.create_info(ch_names=names, sfreq=sf, ch_types="eeg")
        raw = mne.io.RawArray(data * 1e-6, info, verbose="ERROR")
        asr = asrpy.ASR(sfreq=sf, cutoff=cutoff)
        cal_n = int(min(calibration_seconds, secs) * sf)
        asr.fit(raw.copy().crop(tmax=max(cal_n / sf - 1.0 / sf, 1.0)))
        cleaned_raw = asr.transform(raw)
        cleaned_uv = cleaned_raw.get_data() * 1e6
        frac = float(np.mean(np.abs(cleaned_uv - data) > 1e-9))
        backend = "asr"
    except ImportError:
        backend = "burst_limiter"
        notes.append("asrpy not installed — using the conservative robust "
                     "burst-limiter fallback (install asrpy for true subspace "
                     "reconstruction).")
        cal_n = max(int(min(calibration_seconds, secs) * sf), int(sf))
        cleaned_uv = data.copy()
        touched = np.zeros(data.shape, dtype=bool)
        for ch in range(data.shape[0]):
            calib = data[ch, :cal_n]
            med = float(np.median(calib))
            mad = float(np.median(np.abs(calib - med)))
            sd = 1.4826 * mad if mad > 0 else (float(np.std(calib)) or 1.0)
            thr = cutoff * sd
            lo, hi = med - thr, med + thr
            mask = (data[ch] < lo) | (data[ch] > hi)
            touched[ch] = mask
            cleaned_uv[ch] = np.clip(data[ch], lo, hi)
        frac = float(np.mean(touched))

    if secs < rec.duration_s:
        notes.append(f"applied to the first {secs:.0f}s (of {rec.duration_s:.0f}s) "
                     "to bound memory.")
    notes.append("re-check spike/HFO morphology on the corrected signal — burst "
                 "correction can attenuate genuine epileptiform transients.")
    return AsrResult(
        available=True, cutoff=cutoff, fraction_corrected=round(frac, 4),
        backend=backend, cleaned_recording=_eeg_only_rec(rec, names, cleaned_uv),
        notes=notes,
    )


def summarize_asr(result: AsrResult) -> dict:
    return {
        "available": result.available,
        "backend": result.backend,
        "cutoff": result.cutoff,
        "fraction_corrected": result.fraction_corrected,
        "notes": result.notes,
    }
