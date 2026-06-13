"""Tests for src/analyses/spike_sharp.py — broadband sharpness-gated spikes (Wave 7)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.readers.base import EEGRecording
from src.analyses.spike_sharp import detect_sharp_spikes, summarize_sharp_spikes
from src.analyses.morphology import compute_spike_morphology

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


def _mk(sig, sf=200.0, name="Pz"):
    n = sig.shape[0]
    rec = EEGRecording(
        path=Path("/tmp/s.eeg"), sfreq=sf, n_channels=1, duration_s=n / sf,
        channel_names=[name], n_channels_in_file=1, eeg_channel_indices=[0],
        format_name="syn",
    )
    rec._full_data = sig.reshape(1, -1).astype(np.float32)
    return rec


print("\n── Wave 7: broadband sharpness-gated spike detector ───────────────")

sf = 200.0
n = int(120 * sf)
rng = np.random.RandomState(0)

# (1) Channel with genuine sharp spikes only.
sharp = 15.0 * rng.randn(n).astype(np.float32)
spike = 200.0 * np.exp(-0.5 * (np.arange(-8, 9) / 2.0) ** 2)
for _ in range(40):
    c = rng.randint(50, n - 50)
    sharp[c - 8:c + 9] += spike
rec_sharp = _mk(sharp)

# (2) Channel with rhythmic 10 Hz (mu-like) bursts — NOT spikes.
rhythmic = 15.0 * rng.randn(n).astype(np.float32)
for _ in range(40):
    c = rng.randint(200, n - 200)
    w = np.arange(200)
    rhythmic[c:c + 200] += 80.0 * np.sin(2 * np.pi * 10 * w / sf) * np.hanning(200)
rec_rhythm = _mk(rhythmic)

morph_s = compute_spike_morphology(rec_sharp, 0, rec_sharp.n_epochs, target_channel="Pz")
sharp_s = detect_sharp_spikes(rec_sharp, 0, rec_sharp.n_epochs, target_channel="Pz")
morph_r = compute_spike_morphology(rec_rhythm, 0, rec_rhythm.n_epochs, target_channel="Pz")
sharp_r = detect_sharp_spikes(rec_rhythm, 0, rec_rhythm.n_epochs, target_channel="Pz")

check("sharp detector agrees with morphology on genuine spikes",
      abs(sharp_s.sharp_rate_per_min - morph_s.n_events_per_minute) < 5.0,
      f"sharp={sharp_s.sharp_rate_per_min} morph={morph_s.n_events_per_minute}")
check("nearly all genuine-spike candidates pass the sharpness gate (>90%)",
      sharp_s.pct_candidates_sharp > 90.0, f"{sharp_s.pct_candidates_sharp}%")
check("sharp detector REJECTS rhythmic mu that morphology over-counts",
      sharp_r.sharp_rate_per_min < 0.4 * morph_r.n_events_per_minute,
      f"sharp={sharp_r.sharp_rate_per_min} morph={morph_r.n_events_per_minute}")
check("few rhythmic candidates pass the sharpness gate (<40%)",
      sharp_r.pct_candidates_sharp < 40.0, f"{sharp_r.pct_candidates_sharp}%")
check("summary JSON-serializable",
      isinstance(json.dumps(summarize_sharp_spikes(sharp_s)), str))
check("rates are non-negative and finite",
      sharp_s.sharp_rate_per_min >= 0 and np.isfinite(sharp_s.sharp_rate_per_min))

print(f"\n{'='*60}\n  PASS: {n_pass}\n  FAIL: {n_fail}\n{'='*60}")
if n_fail:
    sys.exit(1)
