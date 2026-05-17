"""Automatic sleep-onset and sleep-window detection.

v0.14.3 — Redesigned to fix the all-day primary use-case.

Strategy (in order of preference):

1. **YASA-first for long recordings (≥ 12h)**
   If YASA is available, compute the per-epoch hypnogram and locate the
   longest contiguous non-wake block. This is far more reliable on
   all-day pediatric recordings than a spectral threshold, because
   YASA models the joint stage distribution rather than a single
   delta/alpha ratio.

2. **Spectral fallback (absolute threshold + long smoothing)**
   For shorter recordings, or if YASA fails, use an absolute threshold
   on log(delta/alpha) — sleep occurs when delta dominates, i.e.
   log(delta/alpha) > 0 — with a 15-min smoothing window. This is
   honest about uncertainty: if no block of sufficient duration is
   found, we return confidence='low' rather than silently guessing.

3. **Honest fallback**
   Only used when neither path produces a plausible block. Returns
   confidence='low' and a clear `note` entry so the UI can warn the
   user instead of silently presenting a synthetic default.

The acclimatization heuristic from v0.14.1 is retained but now requires:
- first block start hour < 1
- first block alpha-O1 > primary block alpha (existing)
- first block N3-dominance >= 70% (new — guards against false-positive
  when the first block was genuinely brief sleep)
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
    acclimatization_end_hours: float | None = None
    # Legacy single-string note (joined). Retained for back-compat with tests
    # that pass `note=...` to the constructor. The new `notes` list is the
    # canonical representation.
    note: str = ""
    # v0.14.3 additions:
    notes: list[str] = field(default_factory=list)
    apply_safe: bool = True     # False when result is a synthetic fallback —
                                # UI should NOT silently apply it as a default

    def __post_init__(self):
        # Reconcile legacy `note` with new `notes` list
        if self.note and not self.notes:
            self.notes = [self.note]
        elif self.notes and not self.note:
            self.note = "; ".join(self.notes)


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
    yasa_labels: list[str] | None = None,
    occipital_channels: tuple[str, ...] = ("O1", "O2"),
) -> bool:
    """Best-effort heuristic: is the first sleep block misclassified quiet wake?

    Conditions (ALL must be true to fire):
    (a) first block alpha-O1 power > 1.30 × primary block alpha-O1 power
    (b) first block YASA N3-dominance >= 70%  (guards false-positives;
        skipped if yasa_labels not provided)

    Caller is responsible for checking that the first block starts at
    hour < 1 and isn't the primary block.
    """
    occ_indices = []
    for name in occipital_channels:
        idx = rec.channel_index(name)
        if idx is not None:
            occ_indices.append(idx)
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
                end=min(run[1], run[0] + 20),
            ):
                chans = d[occ_indices].mean(axis=0)
                af = sosfiltfilt(sos_alpha, chans)
                vals.append(float(np.sqrt(np.mean(af ** 2))))
            return float(np.mean(vals)) if vals else 0.0

        first_alpha = _mean_alpha(first_run)
        primary_alpha = _mean_alpha(primary_run)
        alpha_condition = first_alpha > primary_alpha * 1.30

        if not alpha_condition:
            return False

        # N3-dominance guard
        if yasa_labels is not None:
            seg = yasa_labels[first_run[0]:first_run[1]]
            if len(seg) == 0:
                return False
            n3_frac = sum(1 for s in seg if s == "N3") / len(seg)
            if n3_frac < 0.70:
                return False

        return True
    except Exception:
        return False


def _detect_via_yasa(
    rec: EEGRecording,
    epoch_seconds: float,
) -> tuple[list[tuple[int, int]], list[str]] | None:
    """Run YASA, return (sleep_runs, labels) or None on failure.

    `sleep_runs` is the list of all contiguous non-W blocks.
    """
    try:
        from .sleep_stages import compute_sleep_stages
        ss = compute_sleep_stages(rec, epoch_seconds=epoch_seconds, method="yasa")
        if ss.method != "yasa":
            return None
        labels = ss.epoch_labels
        is_sleep = np.array([l != "W" for l in labels], dtype=bool)
        runs = _find_contiguous_runs(is_sleep)
        return runs, labels
    except Exception:
        return None


def _spectral_runs(
    rec: EEGRecording,
    epoch_seconds: float,
    indices: list[int],
    smoothing_minutes: float,
    min_block_hours: float,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Compute spectral sleep-runs via absolute log(delta/alpha) threshold.

    Returns (runs, ratio_smooth) — runs are contiguous blocks where the
    smoothed ratio exceeds 0 (delta dominates alpha).
    """
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

    eps_floor = 1e-3
    ratio_log = np.log(np.maximum(delta_rms, eps_floor) /
                       np.maximum(alpha_rms, eps_floor))

    win_epochs = max(1, int(smoothing_minutes * 60 / epoch_seconds))
    kernel = np.ones(win_epochs) / win_epochs
    ratio_smooth = np.convolve(ratio_log, kernel, mode="same")

    # Absolute threshold: delta dominates → log(delta/alpha) > 0
    above = ratio_smooth > 0.0

    # Bridge short gaps (≤ 15 min)
    max_gap_epochs = int(15 * 60 / epoch_seconds)
    bridged = _bridge_short_gaps(above, max_gap_epochs)
    runs = _find_contiguous_runs(bridged)
    return runs, ratio_smooth


def detect_sleep_window(
    rec: EEGRecording,
    epoch_seconds: float = 30.0,
    central_channels: tuple[str, ...] = ("Cz", "C3", "C4", "Fz", "Pz"),
    smoothing_minutes: float = 5.0,
    min_sleep_hours: float = 1.0,
) -> SleepWindowResult:
    """Auto-detect the main sleep window.

    v0.14.3: YASA-first for ≥ 12h recordings, honest spectral fallback,
    no silent synthetic defaults.
    """
    indices = []
    for name in central_channels:
        i = rec.channel_index(name)
        if i is not None:
            indices.append(i)
    if not indices:
        raise ValueError("No central channels available for sleep detection.")

    n_epochs = rec.n_epochs
    total_h = n_epochs * epoch_seconds / 3600
    is_all_day = total_h >= 16.0
    is_long = total_h >= 12.0

    # Always compute spectral ratio for plotting/debug (cheap relative to YASA)
    # For long recordings use larger smoothing window
    spectral_smooth_min = 15.0 if is_long else smoothing_minutes
    min_block_hours = 4.0 if is_long else 2.0
    spectral_runs, ratio_smooth = _spectral_runs(
        rec, epoch_seconds, indices, spectral_smooth_min, min_block_hours,
    )

    notes: list[str] = []
    additional_blocks: list[dict] = []
    acclimatization_end_hours: float | None = None
    yasa_labels: list[str] | None = None

    all_runs: list[tuple[int, int]] = []
    primary_run: tuple[int, int] | None = None
    used_yasa = False

    # === 1) YASA-first path for long recordings ===
    if is_long:
        yasa_result = _detect_via_yasa(rec, epoch_seconds)
        if yasa_result is not None:
            yasa_runs, yasa_labels = yasa_result
            long_yasa = [r for r in yasa_runs
                         if (r[1] - r[0]) * epoch_seconds / 3600 >= min_block_hours]
            # Sanity guard: if YASA labels everything as a single huge sleep block
            # (>95% of recording), distrust it (likely synthetic/degenerate input)
            # and fall through to spectral.
            n_sleep = sum(1 for l in yasa_labels if l != "W")
            degenerate = (n_sleep / max(len(yasa_labels), 1)) > 0.95
            if long_yasa and not degenerate:
                primary_run = max(long_yasa, key=lambda r: r[1] - r[0])
                all_runs = yasa_runs
                used_yasa = True
                notes.append("yasa_used")
                notes.append("yasa_primary_block")

    # === 2) Spectral fallback ===
    if primary_run is None:
        long_spec = [r for r in spectral_runs
                     if (r[1] - r[0]) * epoch_seconds / 3600 >= min_block_hours]
        if long_spec:
            primary_run = max(long_spec, key=lambda r: r[1] - r[0])
            all_runs = spectral_runs
            notes.append("spectral_used")
        elif spectral_runs:
            # No block reaches min_block_hours — take longest available but
            # mark low-confidence
            primary_run = max(spectral_runs, key=lambda r: r[1] - r[0])
            all_runs = spectral_runs
            notes.append("spectral_used")
            notes.append("no_block_meets_min_duration")

    # === 3) Honest fallback — no clear sleep block found ===
    if primary_run is None:
        notes.append("no_clear_sleep_block_found")
        notes.append("fallback_window_synthetic")
        # Return an honest-low result: synthetic mid-recording window so
        # downstream analyses don't crash, but apply_safe=False to signal
        # UI not to silently treat as a default.
        if total_h >= 14:
            start = int(6 * 3600 / epoch_seconds)
            end = int(14 * 3600 / epoch_seconds)
        elif total_h >= 6:
            start = n_epochs // 4
            end = 3 * n_epochs // 4
        else:
            start = 0
            end = n_epochs
        sleep_start_h = start * epoch_seconds / 3600
        sleep_end_h = end * epoch_seconds / 3600
        return SleepWindowResult(
            sleep_start_epoch=int(start),
            sleep_end_epoch=int(end),
            sleep_start_hours=float(sleep_start_h),
            sleep_end_hours=float(sleep_end_h),
            sleep_duration_hours=float(sleep_end_h - sleep_start_h),
            confidence="low",
            wake_indices=[],
            delta_alpha_ratio_log=ratio_smooth.tolist(),
            additional_blocks=[],
            acclimatization_end_hours=None,
            notes=notes,
            apply_safe=False,
        )

    sleep_start, sleep_end = primary_run
    duration_h = (sleep_end - sleep_start) * epoch_seconds / 3600

    # Confidence determination
    if used_yasa and duration_h >= 4.0:
        confidence = "high"
    elif is_long and duration_h < 4.0:
        confidence = "low"
    elif duration_h < min_sleep_hours:
        confidence = "low"
    elif 6.0 <= duration_h <= 12.0:
        confidence = "high"
    elif 4.0 <= duration_h < 6.0 or 12.0 < duration_h <= 14.0:
        confidence = "medium"
    elif 2.0 <= duration_h < 4.0:
        confidence = "medium"
    else:
        confidence = "low"

    # Secondary blocks: all runs except primary, ≥ 20 min
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

    # Acclimatization heuristic — only on all-day recordings.
    if is_all_day and all_runs:
        first_run = all_runs[0]
        first_start_h = first_run[0] * epoch_seconds / 3600
        first_dur_h = (first_run[1] - first_run[0]) * epoch_seconds / 3600
        if (first_start_h < 1.0 and first_dur_h < 3.0
                and first_run != primary_run):
            is_acclim = _compute_acclimatization_check(
                rec, first_run, primary_run, epoch_seconds,
                yasa_labels=yasa_labels,
            )
            if is_acclim:
                acclimatization_end_hours = float(
                    first_run[1] * epoch_seconds / 3600
                )
                notes.append(
                    "acclimatization_suspected"
                )

    # Wake indices: outside [start-30min, end+30min]
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
        notes=notes,
        apply_safe=True,
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
        "apply_safe": result.apply_safe,
    }
    if result.acclimatization_end_hours is not None:
        out["acclimatization_end_hours"] = round(
            result.acclimatization_end_hours, 2
        )
    if result.note:
        out["note"] = result.note
    if result.notes:
        out["notes"] = list(result.notes)
    return out
