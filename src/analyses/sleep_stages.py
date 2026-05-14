"""Sleep stage classification using YASA.

Returns per-30s-epoch labels (W / N1 / N2 / N3 / REM) for the entire
recording. This unlocks per-stage analyses (per-stage SWI, per-stage
spindle density, sleep architecture metrics).

CAVEAT: YASA's SleepStaging model is trained on ADULT polysomnography
recordings. Applied to pediatric overnight EEG without standard PSG
channels (EOG, EMG), it provides a heuristic estimate — not a clinical
sleep stage classification. The output is flagged as `confidence='heuristic'`
for this reason.

For the reference patient's age (5y), normal NREM stage durations differ from adult
norms (more N3, less N2). Use the per-stage outputs as approximations
suitable for SWI calculation, not as a substitute for human-scored
polysomnography.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..readers.base import EEGRecording


@dataclass
class SleepStageResult:
    epoch_labels: list[str]              # per-30s-epoch: 'W' | 'N1' | 'N2' | 'N3' | 'REM'
    epoch_seconds: float
    confidence: str                      # 'heuristic' | 'fallback'
    stage_minutes: dict[str, float]      # minutes per stage
    sleep_efficiency_pct: float          # % of total recording in any sleep stage
    n_nrem_cycles_estimated: int         # heuristic count
    channel_used: str
    method: str                          # 'yasa' | 'fallback_delta_alpha'


def _yasa_available() -> bool:
    try:
        import yasa  # noqa: F401
        return True
    except ImportError:
        return False


def _fallback_staging(
    rec: EEGRecording,
    epoch_seconds: float,
    target_amplitude_uv: float,
) -> tuple[list[str], str, str]:
    """Heuristic 3-state staging (W/NREM/REM) using delta/alpha ratio.

    Used when YASA is unavailable. Coarse: no N1/N2/N3 distinction —
    all NREM-like epochs are labeled 'N2' as a midline guess so SWI
    calculations still work in a degraded mode.
    """
    from scipy.signal import butter, sosfiltfilt

    central_names = ("Cz", "C3", "C4", "Fz", "Pz")
    indices = []
    for name in central_names:
        i = rec.channel_index(name)
        if i is not None:
            indices.append(i)
    if not indices:
        n = rec.n_epochs
        return ["W"] * n, "fallback", "fallback_delta_alpha"
    channel_used = central_names[0]

    sos_d = butter(4, [0.5, 4.0], btype="band", fs=rec.sfreq, output="sos")
    sos_a = butter(4, [8.0, 13.0], btype="band", fs=rec.sfreq, output="sos")

    labels: list[str] = []
    for _, d in rec.iter_epochs(epoch_seconds=epoch_seconds):
        chans = d[indices].mean(axis=0)
        df = sosfiltfilt(sos_d, chans)
        af = sosfiltfilt(sos_a, chans)
        delta_rms = float(np.sqrt(np.mean(df ** 2)))
        alpha_rms = float(np.sqrt(np.mean(af ** 2)))
        ratio = delta_rms / max(alpha_rms, 1e-3)
        if ratio > 3.0:
            labels.append("N2")  # midline NREM guess
        elif ratio > 1.5:
            labels.append("N1")
        else:
            labels.append("W")
    return labels, "fallback", "fallback_delta_alpha"


def compute_sleep_stages(
    rec: EEGRecording,
    epoch_seconds: float = 30.0,
    target_amplitude_uv: float = 20.0,
    method: str = "auto",
) -> SleepStageResult:
    """Classify each 30s epoch as W/N1/N2/N3/REM.

    Parameters
    ----------
    rec : EEGRecording
    epoch_seconds : float
    target_amplitude_uv : float
        Signal scaling target for YASA (its model assumes µV-range input).
    method : str
        'auto' (YASA if installed, else fallback), 'yasa', or 'fallback'.
    """
    if method == "auto":
        method = "yasa" if _yasa_available() else "fallback"

    # Resolve a usable central channel
    ch_name = None
    ch_idx = None
    for candidate in ("C4", "C3", "Cz", "Fz"):
        ch_idx = rec.channel_index(candidate)
        if ch_idx is not None:
            ch_name = candidate
            break
    if ch_idx is None:
        # No standard central channel — fall back to W-only
        labels = ["W"] * rec.n_epochs
        return _build_result(labels, epoch_seconds, "fallback", "?", "fallback_no_channel")

    if method == "yasa" and _yasa_available():
        try:
            labels = _stage_with_yasa(rec, ch_idx, epoch_seconds, target_amplitude_uv)
            return _build_result(labels, epoch_seconds, "heuristic", ch_name, "yasa")
        except Exception:
            # If YASA fails for any reason (model load, data issues), fall back
            pass

    labels, conf, m = _fallback_staging(rec, epoch_seconds, target_amplitude_uv)
    return _build_result(labels, epoch_seconds, conf, ch_name, m)


def _stage_with_yasa(
    rec: EEGRecording,
    ch_idx: int,
    epoch_seconds: float,
    target_amplitude_uv: float,
) -> list[str]:
    """Run YASA.SleepStaging on a central channel.

    Builds an MNE Raw object from the EEG channel only (no EOG/EMG), scaled
    to µV. YASA accepts this minimal configuration with reduced confidence.
    """
    import mne
    import yasa
    mne.set_log_level("ERROR")

    # Build continuous trace
    segments = []
    for _, d in rec.iter_epochs(epoch_seconds=epoch_seconds):
        segments.append(d[ch_idx])
    x = np.concatenate(segments).astype(np.float64)
    x = x - x.mean()
    scale = target_amplitude_uv / max(x.std(), 1e-9)
    x_uv = x * scale
    # Convert µV → V for MNE (MNE expects volts internally)
    x_v = x_uv * 1e-6

    info = mne.create_info(
        ch_names=["EEG"], sfreq=rec.sfreq, ch_types=["eeg"]
    )
    raw = mne.io.RawArray(x_v[np.newaxis, :], info, verbose=False)

    sls = yasa.SleepStaging(
        raw, eeg_name="EEG", eog_name=None, emg_name=None,
        metadata=dict(age=None, male=None),
    )
    predictions = sls.predict()
    return list(predictions)


def _build_result(
    labels: list[str],
    epoch_seconds: float,
    confidence: str,
    channel: str,
    method: str,
) -> SleepStageResult:
    minutes_per_label = epoch_seconds / 60.0
    stage_minutes = {
        "W": labels.count("W") * minutes_per_label,
        "N1": labels.count("N1") * minutes_per_label,
        "N2": labels.count("N2") * minutes_per_label,
        "N3": labels.count("N3") * minutes_per_label,
        "REM": labels.count("REM") * minutes_per_label,
    }
    total_min = sum(stage_minutes.values())
    sleep_min = sum(v for k, v in stage_minutes.items() if k != "W")
    sleep_eff = 100 * sleep_min / max(total_min, 1)

    # Heuristic NREM cycle count: count NREM → REM transitions
    n_cycles = 0
    in_nrem = False
    for lab in labels:
        if lab in ("N1", "N2", "N3"):
            in_nrem = True
        elif lab == "REM" and in_nrem:
            n_cycles += 1
            in_nrem = False

    return SleepStageResult(
        epoch_labels=labels,
        epoch_seconds=epoch_seconds,
        confidence=confidence,
        stage_minutes=stage_minutes,
        sleep_efficiency_pct=float(sleep_eff),
        n_nrem_cycles_estimated=n_cycles,
        channel_used=channel,
        method=method,
    )


def summarize_sleep_stages(result: SleepStageResult) -> dict:
    return {
        "method": result.method,
        "confidence": result.confidence,
        "channel_used": result.channel_used,
        "stage_minutes": {k: round(v, 1) for k, v in result.stage_minutes.items()},
        "sleep_efficiency_pct": round(result.sleep_efficiency_pct, 1),
        "n_nrem_cycles_estimated": result.n_nrem_cycles_estimated,
        "n_epochs": len(result.epoch_labels),
    }
