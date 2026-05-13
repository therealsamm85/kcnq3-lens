"""Quick end-to-end smoke test against the reference patient's EEG.

Run with:
    python -m tests.test_end_to_end /path/to/eeg/FA06301E.EEG
"""

import sys
from pathlib import Path

# Add parent dir to path so we can import src.* from this script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.readers import load_eeg
from src.analyses import (
    compute_topography,
    compute_spindle_density,
    compute_background_power,
    compute_sustained_bursts,
    compute_spike_morphology,
)
from src.analyses.topography import summarize_topography
from src.analyses.spindles import summarize_spindles
from src.analyses.background import summarize_background
from src.analyses.bursts import summarize_bursts
from src.analyses.morphology import summarize_morphology


def main(eeg_path: str):
    print(f"\n=== KCNQ3-Lens smoke test: {eeg_path} ===\n")

    print("Loading recording...")
    rec = load_eeg(eeg_path)
    print(f"  Format: {rec.format_name}")
    print(f"  Sampling rate: {rec.sfreq} Hz")
    print(f"  Duration: {rec.duration_s/3600:.2f} h")
    print(f"  Channels in file: {rec.n_channels_in_file}")
    print(f"  EEG channels: {rec.n_channels}")
    print(f"  Names: {rec.channel_names[:10]}...")

    # Use sleep window covering ~22:30 to ~07:30 if recording starts ~14:37
    # Sleep window: roughly epochs 960-2040 for a 24h recording
    sleep_start = min(960, rec.n_epochs // 4)
    sleep_end = min(2040, int(rec.n_epochs * 0.85))
    wake_indices = list(range(120, min(600, rec.n_epochs // 4)))

    print(f"\nWindows used:")
    print(f"  Sleep: epochs {sleep_start}-{sleep_end} ({(sleep_end-sleep_start)*30/3600:.1f}h)")
    print(f"  Wake:  epochs {wake_indices[0]}-{wake_indices[-1]} ({len(wake_indices)*30/60:.0f} min)")

    print("\n[1/5] Topography...")
    topo = compute_topography(rec, start_epoch=sleep_start, end_epoch=sleep_end)
    summary = summarize_topography(topo, top_n=5)
    print(f"  Top channels by median kurtosis:")
    for ch in summary["top_channels"]:
        print(f"    {ch['name']:<5} {ch['median_kurtosis']}")

    print("\n[2/5] Sleep spindles...")
    spindle = compute_spindle_density(
        rec, sleep_start_epoch=sleep_start, sleep_end_epoch=sleep_end, age_years=5.0
    )
    s = summarize_spindles(spindle)
    print(f"  Channel: {s['channel']}")
    print(f"  Density: {s['density_per_minute']:.2f}/min (n={s['n_spindles']})")
    print(f"  Peak freq: {s['median_peak_freq_hz']:.1f} Hz")
    print(f"  Interpretation: {s['interpretation']} (norm: {s['age_normative_range']})")

    print("\n[3/5] Background power...")
    try:
        bg = compute_background_power(
            rec,
            wake_epoch_indices=wake_indices,
            age_years=5.0,
            delta_artifact_threshold=None,
        )
        b = summarize_background(bg)
        print(f"  PDR: {b['posterior_dominant_rhythm_hz']:.1f} Hz")
        print(f"  Delta/Alpha ratio: {b['delta_alpha_ratio']:.2f}")
        print(f"  Bands (D/T/A/B): "
              f"{b['delta_pct']:.0f}/{b['theta_pct']:.0f}/{b['alpha_pct']:.0f}/{b['beta_pct']:.0f}")
        print(f"  Interpretation: {b['interpretation']}")
    except Exception as e:
        print(f"  Skipped: {e}")

    print("\n[4/5] Sustained bursts...")
    bursts = compute_sustained_bursts(rec, start_epoch=sleep_start, end_epoch=sleep_end)
    br = summarize_bursts(bursts)
    print(f"  Channel: {br['primary_channel']}")
    print(f"  Bursts ≥3s : {br['n_bursts']}")
    print(f"  Bursts ≥5s : {br['n_bursts_5s_or_longer']}")
    print(f"  Bursts ≥10s: {br['n_bursts_10s_or_longer']}")
    print(f"  Max duration: {br['max_duration_s']:.1f} s")

    print("\n[5/5] Spike morphology...")
    morph = compute_spike_morphology(rec, start_epoch=sleep_start, end_epoch=sleep_end)
    m = summarize_morphology(morph)
    print(f"  Channel: {m['channel']}")
    print(f"  Events: {m['n_events']} ({m['events_per_minute']:.1f}/min)")
    print(f"  Simple:  {m['pct_simple_spikes']:.0f}%")
    print(f"  Sharp:   {m['pct_sharp_waves']:.0f}%")
    print(f"  Complex: {m['pct_complex_spike_wave']:.0f}%")
    print(f"  Classification: {m['classification']}")

    print("\n=== Smoke test PASSED ===\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default for the reference patient's recording
        path = "/path/to/eeg/FA06301E.EEG"
    else:
        path = sys.argv[1]
    main(path)
