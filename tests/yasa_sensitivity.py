"""Test YASA spindle detection at different signal scales and thresholds.

Purpose: figure out whether YASA's near-zero count is due to scaling, threshold
sensitivity, or a genuine finding about Liyana's EEG.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yasa
from scipy.signal import butter, sosfiltfilt

from src.readers import load_eeg


EEG_PATH = "/Volumes/INTENSO/NKT/EEG2100/FA06301E.EEG"
SFREQ = 200


def build_cz(rec, start_ep, end_ep, bandpass=None):
    cz_idx = rec.channel_index("Cz")
    segments = []
    for ep in range(start_ep, end_ep):
        d = rec.read_epoch(ep)
        if d is None:
            continue
        segments.append(d[cz_idx])
    cz = np.concatenate(segments).astype(np.float64)
    cz = cz - cz.mean()
    if bandpass:
        sos = butter(4, list(bandpass), btype="band", fs=SFREQ, output="sos")
        cz = sosfiltfilt(sos, cz)
    return cz


def run_yasa(signal, label, freq_sp=(11, 16), thresh=None):
    n_min = len(signal) / SFREQ / 60
    res = yasa.spindles_detect(
        signal, sf=SFREQ, freq_sp=freq_sp,
        thresh=thresh or {'corr': 0.65, 'rel_pow': 0.20, 'rms': 1.5},
        verbose=False,
    )
    if res is None:
        return label, 0, 0.0
    df = res.summary()
    n = len(df)
    return label, n, n / n_min


def main():
    rec = load_eeg(EEG_PATH)

    # ─── Window 1: original 6h mid-sleep (1080–1800) ─────────────────────────
    print("=" * 76)
    print("MID-SLEEP WINDOW (epochs 1080–1800, 6h)")
    print("=" * 76)

    cz_raw = build_cz(rec, 1080, 1800)
    print(f"Raw Cz: std={cz_raw.std():.0f}, range=[{cz_raw.min():.0f}, {cz_raw.max():.0f}]")

    cz_hp = build_cz(rec, 1080, 1800, bandpass=(0.5, 40))
    print(f"0.5-40 Hz filtered: std={cz_hp.std():.0f}")
    print()

    print(f"  {'Scaling':<24}  {'Spindles':>10}  {'/min':>10}")
    print("-" * 50)

    # Try several target stds on the broadband-filtered signal
    for target_std in [5, 10, 15, 20, 30, 50, 100]:
        scaled = cz_hp * (target_std / cz_hp.std())
        label, n, density = run_yasa(scaled, f"target_std={target_std} µV")
        print(f"  {label:<24}  {n:>10}  {density:>10.2f}")

    # Try lower correlation/rel_pow thresholds (more permissive)
    print()
    print("Permissive thresholds at target_std=20 µV:")
    scaled = cz_hp * (20.0 / cz_hp.std())
    for thresh_name, thresh in [
        ("default",      {'corr': 0.65, 'rel_pow': 0.20, 'rms': 1.5}),
        ("loose corr",   {'corr': 0.50, 'rel_pow': 0.15, 'rms': 1.0}),
        ("very loose",   {'corr': 0.40, 'rel_pow': 0.10, 'rms': 1.0}),
        ("only rel_pow", {'corr': 0.30, 'rel_pow': 0.10, 'rms': 0.5}),
    ]:
        label, n, density = run_yasa(scaled, thresh_name, thresh=thresh)
        print(f"  {label:<24}  {n:>10}  {density:>10.2f}")

    # ─── Window 2: quieter mid-sleep window (1440–1680, where we expected
    # deepest NREM with most spindles) ───────────────────────────────────────
    print()
    print("=" * 76)
    print("DEEP-NREM WINDOW (epochs 1440–1680, 2h, peak spindle window)")
    print("=" * 76)

    cz_hp2 = build_cz(rec, 1440, 1680, bandpass=(0.5, 40))
    print(f"Filtered Cz: std={cz_hp2.std():.0f}")
    print()
    print(f"  {'Scaling':<24}  {'Spindles':>10}  {'/min':>10}")
    print("-" * 50)
    for target_std in [10, 20, 30, 50]:
        scaled = cz_hp2 * (target_std / cz_hp2.std())
        label, n, density = run_yasa(scaled, f"target_std={target_std} µV")
        print(f"  {label:<24}  {n:>10}  {density:>10.2f}")


if __name__ == "__main__":
    main()
