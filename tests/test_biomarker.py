"""Tests for src/longitudinal/biomarker.py — spike-burden trajectory (Wave 9)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.readers.base import EEGRecording
from src.longitudinal.biomarker import track_spike_rate, summarize_trajectory

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
n_pass = n_fail = 0


def check(name, cond, detail=""):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"  {PASS} {name}")
    else:
        n_fail += 1
        print(f"  {FAIL} {name}  {detail}")


def _make_rec_with_spikes(spikes_per_min: float, sfreq=200.0, minutes=2.0,
                          channel="Pz"):
    """Synthetic single-channel recording with a controlled spike rate."""
    n = int(minutes * 60 * sfreq)
    rng = np.random.RandomState(int(spikes_per_min * 7))
    data = 15.0 * rng.randn(1, n).astype(np.float32)
    n_spikes = int(spikes_per_min * minutes)
    # Place sharp Gaussian spikes (width 2 samples ≈ 10 ms at 200 Hz) — a clean
    # sharp transient with energy in the detector's 10-30 Hz band.
    spike = 200.0 * np.exp(-0.5 * (np.arange(-8, 9) / 2.0) ** 2)
    for _ in range(n_spikes):
        c = rng.randint(50, n - 50)
        data[0, c - 8:c + 9] += spike.astype(np.float32)
    rec = EEGRecording(
        path=Path("/tmp/syn.eeg"), sfreq=sfreq, n_channels=1,
        duration_s=n / sfreq, channel_names=[channel], n_channels_in_file=1,
        eeg_channel_indices=[0], format_name="synthetic",
    )
    rec._full_data = data
    return rec


print("\n── Wave 9: longitudinal spike-burden biomarker ────────────────────")

# Build a falling trajectory: high → low spike rate across 3 timepoints.
specs = [
    (_make_rec_with_spikes(30), "t0_pretreatment", 4.0, "2025-01-01"),
    (_make_rec_with_spikes(18), "t1_on_drug", 4.3, "2025-04-01"),
    (_make_rec_with_spikes(6), "t2_on_drug", 4.6, "2025-07-01"),
]
traj = track_spike_rate(specs, channel="Pz", mad_multiplier=6.0)

check("one point per recording", len(traj.points) == 3,
      f"got {len(traj.points)}")
check("all points measured on the fixed channel Pz",
      all(p.channel == "Pz" for p in traj.points))
check("all points used the same threshold",
      all(p.mad_multiplier == 6.0 for p in traj.points))
rates = [p.rate_per_min for p in traj.points]
check("detected rates are monotonically falling", rates[0] > rates[1] > rates[2],
      f"rates={rates}")
check("trend direction = falling", traj.direction == "falling",
      f"got {traj.direction} (slope={traj.slope_per_year})")
check("slope is negative", traj.slope_per_year is not None and traj.slope_per_year < 0,
      f"slope={traj.slope_per_year}")
check("summary is JSON-serializable",
      isinstance(json.dumps(summarize_trajectory(traj)), str))

# Comparability guard: mixing channels must be flagged.
specs2 = [
    (_make_rec_with_spikes(20, channel="Pz"), "a", 4.0, None),
    (_make_rec_with_spikes(20, channel="Cz"), "b", 4.5, None),
]
traj2 = track_spike_rate(specs2, channel="Pz")
# Both resolve to their only channel; the tracker should notice they differ.
mixed_note = any("different channels" in n for n in traj2.notes)
check("mixing channels across timepoints is flagged", mixed_note,
      f"notes={traj2.notes}")

# Single point → insufficient data, no crash.
traj3 = track_spike_rate([(_make_rec_with_spikes(10), "solo", 4.0, None)])
check("single timepoint → insufficient_data, no crash",
      traj3.direction == "insufficient_data")

print(f"\n{'='*60}\n  PASS: {n_pass}\n  FAIL: {n_fail}\n{'='*60}")
if n_fail:
    sys.exit(1)
