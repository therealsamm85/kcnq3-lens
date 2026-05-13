"""Compare YASA's validated spindle detection vs our heuristic on the reference patient's EEG.

This is an evaluation script — it does NOT decide for us how to integrate.
It surfaces the numerical differences so we can make an informed call.

Run with:
    .venv/bin/python -m tests.compare_yasa
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mne
import yasa
from scipy.signal import butter, sosfiltfilt, hilbert

from src.readers import load_eeg
from src.analyses import compute_spindle_density
from src.analyses.spindles import summarize_spindles


# Match what we used in earlier analysis
EEG_PATH = "/path/to/eeg/FA06301E.EEG"
SLEEP_START_EP = 1080   # ~22:37 sleep window start
SLEEP_END_EP   = 1800   # ~07:37 sleep window end
SFREQ = 200


def main():
    print("=" * 72)
    print("YASA vs KCNQ3-Lens spindle detection comparison")
    print("Recording: the reference patient, pre-treatment night EEG")
    print("=" * 72)

    # --- Load ---
    print("\nLoading recording...")
    rec = load_eeg(EEG_PATH)
    print(f"  {rec.format_name}, {rec.sfreq:.0f} Hz, {rec.duration_s/3600:.2f} h")

    # ===== Method 1: Our implementation =====
    print("\n[1/2] KCNQ3-Lens implementation (heuristic)...")
    t0 = time.time()
    our_result = compute_spindle_density(
        rec,
        sleep_start_epoch=SLEEP_START_EP,
        sleep_end_epoch=SLEEP_END_EP,
        channel="Cz",
        age_years=5.0,
    )
    our_time = time.time() - t0
    ours = summarize_spindles(our_result)
    print(f"  Detected: {ours['n_spindles']} spindles")
    print(f"  Density:  {ours['density_per_minute']:.2f} / min")
    print(f"  Mean dur: {ours['mean_duration_s']:.2f} s")
    print(f"  Peak Hz:  {ours['median_peak_freq_hz']:.1f}")
    print(f"  Runtime:  {our_time:.1f} s")

    # ===== Method 2: YASA =====
    print("\n[2/2] YASA spindles_detect (validated)...")

    # Build a Cz trace across the sleep window
    cz_idx = rec.channel_index("Cz")
    segments = []
    for ep in range(SLEEP_START_EP, SLEEP_END_EP):
        d = rec.read_epoch(ep)
        if d is None:
            continue
        segments.append(d[cz_idx])
    cz = np.concatenate(segments)
    duration_h = len(cz) / SFREQ / 3600
    print(f"  Cz trace: {len(cz)} samples ({duration_h:.2f} h)")

    # Signal scaling: NK EEG-1200A stores raw int16 ADC counts, but YASA's
    # amplitude-based thresholds (rms) assume input is in µV. We rescale so the
    # standard deviation lands in a typical EEG range (~20 µV) — this preserves
    # all relative patterns while making YASA's defaults usable.
    print(f"  Cz signal: mean={cz.mean():.0f}, std={cz.std():.0f}, "
          f"range=[{cz.min():.0f}, {cz.max():.0f}]")
    cz_centered = cz.astype(np.float64) - cz.mean()
    target_std_uv = 20.0
    cz_uv = cz_centered * (target_std_uv / cz_centered.std())
    print(f"  After scaling to ~{target_std_uv} µV std: std={cz_uv.std():.1f}, "
          f"range=[{cz_uv.min():.1f}, {cz_uv.max():.1f}]")

    t0 = time.time()
    result = yasa.spindles_detect(
        cz_uv,
        sf=SFREQ,
        freq_sp=(11, 16),
        freq_broad=(1, 30),
        duration=(0.5, 2.5),
        min_distance=500,
        thresh={'corr': 0.65, 'rel_pow': 0.20, 'rms': 1.5},
        multi_only=False,
        remove_outliers=False,
        verbose=False,
    )
    yasa_time = time.time() - t0

    if result is None:
        print(f"  YASA detected: 0 spindles")
        print(f"  Runtime: {yasa_time:.1f} s")
        return

    sp_df = result.summary()  # YASA 0.7 returns SpindlesResults — convert to DataFrame
    if len(sp_df) == 0:
        print(f"  YASA detected: 0 spindles")
        print(f"  Runtime: {yasa_time:.1f} s")
        return

    n_yasa = len(sp_df)
    density_yasa = n_yasa / (duration_h * 60)
    print(f"  Detected: {n_yasa} spindles")
    print(f"  Density:  {density_yasa:.2f} / min")
    print(f"  Mean dur: {sp_df['Duration'].mean():.2f} s")
    print(f"  Peak Hz:  {sp_df['Frequency'].median():.1f}")
    print(f"  Mean amp: {sp_df['Amplitude'].mean():.1f}")
    print(f"  Runtime:  {yasa_time:.1f} s")

    # ===== Comparison =====
    print("\n" + "=" * 72)
    print("COMPARISON")
    print("=" * 72)
    print(f"  {'Metric':<25}  {'Ours':>10}  {'YASA':>10}  {'Ratio':>8}")
    print(f"  {'Spindles':<25}  {ours['n_spindles']:>10}  {n_yasa:>10}  "
          f"{n_yasa/max(ours['n_spindles'],1):>7.2f}x")
    print(f"  {'Density (/min)':<25}  {ours['density_per_minute']:>10.2f}  "
          f"{density_yasa:>10.2f}  {density_yasa/max(ours['density_per_minute'],0.01):>7.2f}x")
    print(f"  {'Mean duration (s)':<25}  {ours['mean_duration_s']:>10.2f}  "
          f"{sp_df['Duration'].mean():>10.2f}  -")
    print(f"  {'Peak freq (Hz)':<25}  {ours['median_peak_freq_hz']:>10.1f}  "
          f"{sp_df['Frequency'].median():>10.1f}  -")
    print(f"  {'Runtime (s)':<25}  {our_time:>10.1f}  {yasa_time:>10.1f}  -")

    print("\nAge-5 normative range: 3.0–5.0 spindles/min")
    print(f"\nInterpretation:")
    for label, density in [("Ours", ours['density_per_minute']), ("YASA", density_yasa)]:
        if density < 3.0:
            verdict = "BELOW normative — reduced consolidation capacity"
        elif density > 5.0:
            verdict = "ABOVE normative"
        else:
            verdict = "WITHIN normative"
        print(f"  {label:<6}: {density:>5.2f}/min → {verdict}")


if __name__ == "__main__":
    main()
