"""Automatic sleep-onset and sleep-window detection.

For long overnight recordings, manually specifying when sleep starts and ends
is error-prone — and small window differences can shift SWI calculations
dramatically. This module estimates the main sleep period from spectral
features that are robust across pediatric recordings:

- Delta-band power rises during sleep
- Alpha-band power drops during sleep
- The delta/alpha ratio per epoch is a robust binary-ish signal

Strategy:
1. Compute delta_rms and alpha_rms per 30s epoch across the whole recording
2. Compute log(delta/alpha) per epoch
3. Smooth with a 5-minute moving average
4. Find all contiguous non-wake runs; primary (longest) sleep is ≥ 4h

For all-day (≥ 16h) recordings the algorithm also:
- Identifies secondary blocks (naps / short sleeps) of ≥ 20 min
- Flags a suspected acclimatization period at recording start if the first
  sleep block shows paradoxically high alpha power (best-effort heuristic —
  not clinically validated; YASA commonly misclassifies quiet wake as N3)

This is a heuristic — not a substitute for proper polysomnographic sleep
staging — but works well as a starting window for the other analyses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..readers.base import EEGRecording


@dataclass
class SleepWindowResult:
    sleep_start_epoch: int
    sleep_end_epoch: int
    sleep_start_hours: float    # hours from recording start
    sleep_end_hours: float
    sleep_duration_hours: float
    confidence: str             # "high" | "medium" | "low"
    wake_indices: list[int]     # epochs identified as wake (for background analysis)
    delta_alpha_ratio_log: list[float]  # one value per epoch (for plotting/debug)
    # v0.14.1 additions:
    additional_blocks: list[dict] = field(default_factory=list)
    # Each dict: {start_h, end_h, dur_h, kind: 'nap'|'short_sleep'}
    acclimatization_end_hours: float | None = None
    # Best-effort: if first block looks like quiet wake misclassified as sleep.
    # Heuristic only — treat as a suggestion, not a clinical finding.
    note: str = ""


def _find_contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end) index pairs for True runs in mask."""
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _bridge_short_gaps(
    above: np.ndarray, max_gap_epochs: int
) -> np.ndarray:
    """Bridge gaps ≤ max_gap_epochs that are surrounded by True runs."""
    bridged = above.copy()
    i = 0
    while i < len(bridged):
        if not bridged[i]:
            j = i
            while j < len(bridged) and not bridged[j]:
                j += 1
            gap = j - i
            if (gap <= max_gap_epochs and i > 0 and j < len(bridged)
                    and bridged[i - 1] and above[j]):
                bridged[i:j] = True
            i = j
        else:
            i += 1
    return bridged


def _compute_acclimatization_check(
    rec: EEGRecording,
    first_run: tuple[int, int],
    primary_run: tuple[int, int],
    epoch_seconds: float,
    occipital_channels: tuple[str, ...] = ("O1", "O2"),
) -> bool:
    """Best-effort heuristic: is the first sleep block misclassified quiet wake?

    Returns True if the first block has HIGHER alpha power than the primary
    sleep block, suggesting quiet wakefulness (alpha) was mislabeled as sleep
    (N3) by YASA's adult model.

    This is a heuristic — treat output as a suggestion, not a clinical finding.
    Alpha dominance in the first block relative to primary sleep is a necessary
    but not sufficient condition for misclassification.
    """
    occ_indices = []
    for name in occipital_channels:
        idx = rec.channel_index(name)
        if idx is not None:
            occ_indices.append(idx)
    # Fall back to central channels if no occipital channels available
    if not occ_indices:
        for name in ("Cz", "C3", "C4"):
            idx = rec.channel_index(name)
            if idx is not None:
                occ_indices.append(idx)
    if not occ_indices:
        return False

    try:
        sos_alpha = butter(4, [8.0, 13.0], btype="band",
                           fs=rec.sfreq, output="sos")

        def _mean_alpha(run: tuple[int, int]) -> float:
            vals = []
            for ep, d in rec.iter_epochs(
                epoch_seconds=epoch_seconds,
                start=run[0],
                end=min(run[1], run[0] + 20),  # limit to first 10min
            ):
                chans = d[occ_indices].mean(axis=0)
                af = sosfiltfilt(sos_alpha, chans)
                vals.append(float(np.sqrt(np.mean(af ** 2))))
            return float(np.mean(vals)) if vals else 0.0

        first_alpha = _mean_alpha(first_run)
        primary_alpha = _mean_alpha(primary_run)
        # Flag if first block alpha is > 30% higher than primary sleep alpha
        return first_alpha > primary_alpha * 1.30
    except Exception:
        return False


def detect_sleep_window(
    rec: EEGRecording,
    epoch_seconds: float = 30.0,
    central_channels: tuple[str, ...] = ("Cz", "C3", "C4", "Fz", "Pz"),
    smoothing_minutes: float = 5.0,
    min_sleep_hours: float = 1.0,
) -> SleepWindowResult:
    """Auto-detect the main sleep window from spectral features.

    For all-day recordings (≥ 16h), also returns secondary sleep blocks
    (naps / short sleeps ≥ 20 min) and a best-effort acclimatization flag
    for the first block if it appears to be misclassified quiet wake.

    Parameters
    ----------
    rec : EEGRecording
    epoch_seconds : float
    central_channels : iterable of str
        Channels averaged for the spectral computation. Central is best for
        delta/alpha discrimination across sleep stages.
    smoothing_minutes : float
        Moving-average window for the delta/alpha ratio.
    min_sleep_hours : float
        Minimum duration to be considered a valid sleep window. Below this,
        returns low-confidence result.
    """
    # Resolve channels
    indices = []
    for name in central_channels:
        i = rec.channel_index(name)
        if i is not None:
            indices.append(i)
    if not indices:
        raise ValueError("No central channels available for sleep detection.")

    sos_delta = butter(4, [0.5, 4.0], btype="band", fs=rec.sfreq, output="sos")
    sos_alpha = butter(4, [8.0, 13.0], btype="band", fs=rec.sfreq, output="sos")

    n_epochs = rec.n_epochs
    delta_rms = np.zeros(n_epochs)
    alpha_rms = np.zeros(n_epochs)

    for ep, d in rec.iter_epochs(epoch_seconds=epoch_seconds, end=n_epochs):
        chans = d[indices].mean(axis=0)
        df = sosfiltfilt(sos_delta, chans)
        af = sosfiltfilt(sos_alpha, chans)
        delta_rms[ep] = float(np.sqrt(np.mean(df ** 2)))
        alpha_rms[ep] = float(np.sqrt(np.mean(af ** 2)))

    # log delta/alpha ratio (higher = more sleep-like)
    eps_floor = 1e-3
    ratio_log = np.log(np.maximum(delta_rms, eps_floor) /
                       np.maximum(alpha_rms, eps_floor))

    # Smooth with moving average
    win_epochs = max(1, int(smoothing_minutes * 60 / epoch_seconds))
    kernel = np.ones(win_epochs) / win_epochs
    ratio_smooth = np.convolve(ratio_log, kernel, mode="same")

    # Threshold: 50th percentile of the smoothed ratio. Sleep should be the
    # upper ~30-50% of the recording for a typical overnight, but spike
    # clusters during sleep can briefly suppress the ratio.
    threshold = np.percentile(ratio_smooth, 50)
    above = ratio_smooth > threshold

    # Bridge short gaps (≤ 15 minutes) — sleep is rarely interrupted by long
    # wake epochs on otherwise-continuous overnight studies, and spike clusters
    # commonly create brief dips below threshold mid-sleep.
    max_gap_epochs = int(15 * 60 / epoch_seconds)
    bridged = _bridge_short_gaps(above, max_gap_epochs)

    # Find all contiguous runs on the bridged signal
    all_runs = _find_contiguous_runs(bridged)

    total_h = n_epochs * epoch_seconds / 3600
    is_all_day = total_h >= 16.0

    def _fallback_window():
        """Sensible default for an overnight: skip first 6h, take next 8h."""
        if total_h >= 14:
            start = int(6 * 3600 / epoch_seconds)
            end = int(14 * 3600 / epoch_seconds)
        elif total_h >= 6:
            start = n_epochs // 4
            end = 3 * n_epochs // 4
        else:
            start = 0
            end = n_epochs
        return start, end

    # Primary sleep = longest block ≥ 4h (or longest available for short recs)
    _min_primary_epochs = int(4 * 3600 / epoch_seconds) if total_h >= 14 else 0
    long_runs = [r for r in all_runs
                 if (r[1] - r[0]) * epoch_seconds / 3600 >= 4.0]
    short_runs = [r for r in all_runs
                  if (r[1] - r[0]) * epoch_seconds / 3600 < 4.0]

    acclimatization_end_hours: float | None = None
    additional_blocks: list[dict] = []
    notes: list[str] = []

    if not all_runs:
        sleep_start, sleep_end = _fallback_window()
        confidence = "low"
        primary_run = (sleep_start, sleep_end)
    else:
        if long_runs:
            # Longest block among ≥ 4h candidates is the primary
            primary_run = max(long_runs, key=lambda r: r[1] - r[0])
        else:
            # No block reaches 4h — take the longest available
            primary_run = max(all_runs, key=lambda r: r[1] - r[0])

        sleep_start, sleep_end = primary_run
        duration_h = (sleep_end - sleep_start) * epoch_seconds / 3600

        # Sanity check: for recordings ≥ 14h, the detected sleep window should
        # be at least 4 hours. If it isn't, the heuristic is unreliable — fall
        # back to a conventional overnight window.
        if total_h >= 14 and duration_h < 4.0:
            sleep_start, sleep_end = _fallback_window()
            primary_run = (sleep_start, sleep_end)
            confidence = "low"
        elif duration_h < min_sleep_hours:
            confidence = "low"
        elif 6.0 <= duration_h <= 12.0:
            confidence = "high"
        elif 4.0 <= duration_h < 6.0 or 12.0 < duration_h <= 14.0:
            confidence = "medium"
        elif duration_h < 3.0:
            confidence = "medium"
        else:
            confidence = "low"

        # Secondary blocks (naps / short sleeps): all runs except primary
        # that are ≥ 20 min AND are not the primary block
        min_secondary_epochs = int(20 * 60 / epoch_seconds)
        for run in all_runs:
            if run == primary_run:
                continue
            dur_s_h = (run[1] - run[0]) * epoch_seconds / 3600
            if (run[1] - run[0]) < min_secondary_epochs:
                continue
            kind = "nap" if 0.5 <= dur_s_h <= 3.0 else "short_sleep"
            additional_blocks.append({
                "start_h": float(run[0] * epoch_seconds / 3600),
                "end_h": float(run[1] * epoch_seconds / 3600),
                "dur_h": float(dur_s_h),
                "kind": kind,
            })

        # Acclimatization detection (all-day recordings only).
        # Best-effort heuristic: if there is a run BEFORE the primary sleep
        # that starts at hour < 1 AND shows higher occipital alpha than the
        # primary block, flag it as possible quiet-wake misclassified as sleep.
        if is_all_day and all_runs:
            first_run = all_runs[0]
            first_start_h = first_run[0] * epoch_seconds / 3600
            first_dur_h = (first_run[1] - first_run[0]) * epoch_seconds / 3600
            # Only flag if: starts early (< 1h), short (< 3h), not the primary block
            if (first_start_h < 1.0 and first_dur_h < 3.0
                    and first_run != primary_run):
                is_acclim = _compute_acclimatization_check(
                    rec, first_run, primary_run, epoch_seconds
                )
                if is_acclim:
                    acclimatization_end_hours = float(
                        first_run[1] * epoch_seconds / 3600
                    )
                    notes.append(
                        "acclimatization_suspected: first block has elevated "
                        "alpha vs primary sleep — likely quiet wake misclassified "
                        "by YASA adult model (best-effort heuristic, not validated)"
                    )

    # Wake epochs = NOT sleep and bounded away from sleep onset/end by 30 min
    pad_eps = int(30 * 60 / epoch_seconds)
    wake_mask = np.ones(n_epochs, dtype=bool)
    wake_mask[max(0, sleep_start - pad_eps):min(n_epochs, sleep_end + pad_eps)] = False
    wake_indices = np.where(wake_mask)[0].tolist()

    sleep_start_h = sleep_start * epoch_seconds / 3600
    sleep_end_h = sleep_end * epoch_seconds / 3600

    return SleepWindowResult(
        sleep_start_epoch=int(sleep_start),
        sleep_end_epoch=int(sleep_end),
        sleep_start_hours=float(sleep_start_h),
        sleep_end_hours=float(sleep_end_h),
        sleep_duration_hours=float(sleep_end_h - sleep_start_h),
        confidence=confidence,
        wake_indices=wake_indices,
        delta_alpha_ratio_log=ratio_smooth.tolist(),
        additional_blocks=additional_blocks,
        acclimatization_end_hours=acclimatization_end_hours,
        note="; ".join(notes),
    )


def summarize_sleep_window(result: SleepWindowResult) -> dict:
    out = {
        "sleep_start_epoch": result.sleep_start_epoch,
        "sleep_end_epoch": result.sleep_end_epoch,
        "sleep_start_hours": round(result.sleep_start_hours, 2),
        "sleep_end_hours": round(result.sleep_end_hours, 2),
        "sleep_duration_hours": round(result.sleep_duration_hours, 2),
        "confidence": result.confidence,
        "n_wake_epochs_available": len(result.wake_indices),
        "n_additional_blocks": len(result.additional_blocks),
        "additional_blocks": [
            {k: (round(v, 2) if isinstance(v, float) else v)
             for k, v in blk.items()}
            for blk in result.additional_blocks
        ],
    }
    if result.acclimatization_end_hours is not None:
        out["acclimatization_end_hours"] = round(
            result.acclimatization_end_hours, 2
        )
    if result.note:
        out["note"] = result.note
    return out
