"""Ocular (eye-blink) artifact detection — lightweight, montage-agnostic.

Why this exists
---------------
Eye blinks produce large (~75-300 µV), slow (<4 Hz) deflections maximal at
the frontopolar electrodes (Fp1/Fp2). They contaminate any metric that reads
frontal channels — in this project the spike-topography "drift" toward
frontal sites in early recordings was largely blink, not brain.

Full ICA needs the whole recording in memory and a montage; it is overkill
for a family-facing tool and impractical on a 24 h lazy recording. This
module instead detects blink events directly on the frontopolar channels and
exposes a per-epoch blink mask, so a caller can EXCLUDE blink-heavy epochs
from a sensitive analysis (e.g. topography) rather than silently averaging
the artifact in.

This is detection + masking, not correction — it does not subtract the blink
from other channels. That honesty matches the rest of the pipeline: flag the
artifact and let the analysis avoid it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

from ..readers.base import EEGRecording

_FRONTOPOLAR = ("Fp1", "Fp2")


@dataclass
class OcularResult:
    frontopolar_channels: list[str]
    n_blinks: int
    blink_rate_per_min: float
    n_epochs: int
    n_blink_epochs: int          # epochs containing ≥1 blink
    pct_blink_epochs: float
    blink_epoch_indices: list[int] = field(default_factory=list)
    available: bool = True
    unavailable_reason: str = ""


def detect_ocular_artifact(
    rec: EEGRecording,
    start_epoch: int = 0,
    end_epoch: int | None = None,
    epoch_seconds: float = 30.0,
    blink_amplitude_uv: float = 75.0,
    min_blink_separation_s: float = 0.3,
) -> OcularResult:
    """Detect eye blinks on the frontopolar channels.

    A blink is a low-frequency (1-4 Hz) deflection on Fp1/Fp2 whose absolute
    amplitude exceeds blink_amplitude_uv. Returns counts, rate, and the set of
    epochs that contain at least one blink (for masking).

    Requires Fp1 and/or Fp2 to be present; otherwise returns available=False.
    """
    if end_epoch is None:
        end_epoch = rec.n_epochs

    fp_idx = []
    fp_names = []
    for nm in _FRONTOPOLAR:
        i = rec.channel_index(nm)
        if i is not None:
            fp_idx.append(i)
            fp_names.append(rec.channel_names[i])
    if not fp_idx:
        return OcularResult(
            frontopolar_channels=[], n_blinks=0, blink_rate_per_min=0.0,
            n_epochs=0, n_blink_epochs=0, pct_blink_epochs=0.0,
            available=False,
            unavailable_reason="no frontopolar (Fp1/Fp2) channel present",
        )

    sf = rec.sfreq
    # Blink band is 0.5-4 Hz: a blink's dominant energy sits below 1 Hz (it is a
    # slow deflection, not a fast spike), so a 1 Hz high-pass would attenuate it
    # below the amplitude threshold and miss most blinks. Verified: a 200 µV
    # synthetic blink survives 0.5-4 Hz at ~93 µV but only ~31 µV through 1-4 Hz.
    sos = butter(4, [0.5, 4.0], btype="band", fs=sf, output="sos")
    min_dist = max(1, int(min_blink_separation_s * sf))

    total_blinks = 0
    blink_epochs: list[int] = []
    n_eps = 0
    for ep, d in rec.iter_epochs(
        epoch_seconds=epoch_seconds, start=start_epoch, end=end_epoch
    ):
        n_eps += 1
        # Use the max across the available frontopolar channels (a blink shows
        # on both Fp1 and Fp2; taking the max is robust to one being noisy).
        fp = np.max(np.abs(np.array([d[i] for i in fp_idx])), axis=0)
        fp = np.abs(sosfiltfilt(sos, fp))
        peaks, _ = find_peaks(fp, height=blink_amplitude_uv, distance=min_dist)
        if len(peaks):
            total_blinks += int(len(peaks))
            blink_epochs.append(ep)

    total_min = (n_eps * epoch_seconds) / 60.0 if n_eps else 0.0
    rate = total_blinks / total_min if total_min > 0 else 0.0

    return OcularResult(
        frontopolar_channels=fp_names,
        n_blinks=total_blinks,
        blink_rate_per_min=round(rate, 2),
        n_epochs=n_eps,
        n_blink_epochs=len(blink_epochs),
        pct_blink_epochs=round(100.0 * len(blink_epochs) / max(n_eps, 1), 1),
        blink_epoch_indices=blink_epochs,
        available=True,
    )


def clean_epoch_indices(
    rec: EEGRecording,
    blink_epoch_indices: list[int],
    start_epoch: int = 0,
    end_epoch: int | None = None,
) -> list[int]:
    """Return the epoch indices in [start, end) that are NOT blink-contaminated.

    A caller (e.g. topography) can iterate only these to keep eye-blink artifact
    out of a frontal-sensitive metric.
    """
    if end_epoch is None:
        end_epoch = rec.n_epochs
    blink_set = set(blink_epoch_indices)
    return [ep for ep in range(start_epoch, end_epoch) if ep not in blink_set]


def summarize_ocular(result: OcularResult) -> dict:
    return {
        "available": result.available,
        "unavailable_reason": result.unavailable_reason,
        "frontopolar_channels": result.frontopolar_channels,
        "n_blinks": result.n_blinks,
        "blink_rate_per_min": result.blink_rate_per_min,
        "n_blink_epochs": result.n_blink_epochs,
        "pct_blink_epochs": result.pct_blink_epochs,
        # blink_epoch_indices intentionally omitted from the summary (can be
        # long); callers that need masking use the result object directly.
    }
