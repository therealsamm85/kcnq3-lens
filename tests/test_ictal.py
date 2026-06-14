"""Tests for C3 — ictal screener (analyses/ictal.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.readers.base import EEGRecording
from src.analyses.ictal import screen_ictal, summarize_ictal

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


SF = 200.0


def _rec(signal, seconds=120):
    rec = EEGRecording(
        path=Path("/tmp/ic.eeg"), sfreq=SF, n_channels=1, duration_s=seconds,
        channel_names=["C3"], n_channels_in_file=1, eeg_channel_indices=[0],
        format_name="synthetic",
    )
    rec._full_data = signal.reshape(1, -1).astype(np.float32)
    return rec


def _chirp(t, f0, f1):
    """Linear frequency sweep f0→f1 over the time vector t."""
    dur = t[-1] - t[0]
    k = (f1 - f0) / dur
    phase = 2 * np.pi * (f0 * (t - t[0]) + 0.5 * k * (t - t[0]) ** 2)
    return np.sin(phase)


print("\n── C3: evolving rhythmic seizure is flagged ───────────────────────")
n = 120 * int(SF)
rng = np.random.RandomState(0)
sig = 5.0 * rng.randn(n)
# 40–60 s within one 30 s epoch (30–60): 20 s of 8→5 Hz, 150 µV → flaggable run.
a, b = int(40 * SF), int(60 * SF)
tt = np.arange(a, b) / SF
sig[a:b] += 150.0 * _chirp(tt, 8.0, 5.0)
res = screen_ictal(_rec(sig))

check("at least one candidate flagged", res.n_candidates >= 1, f"got {res.n_candidates}")
hit = [c for c in res.candidates if c.start_s <= 60 and c.start_s + c.duration_s >= 40]
check("a candidate overlaps the seizure window", len(hit) >= 1,
      f"cands={[(c.start_s,c.duration_s) for c in res.candidates]}")
if hit:
    check("flagged run has frequency drift ≥ 1 Hz", hit[0].freq_drift_hz >= 1.0,
          f"got {hit[0].freq_drift_hz}")
    check("confidence is moderate (evolving)", hit[0].confidence == "moderate",
          f"got {hit[0].confidence}")
check("no candidate is ever 'high' confidence",
      all(c.confidence in ("low", "moderate") for c in res.candidates))
check("caveat present (screening only)", "SCREENING ONLY" in res.caveat)
check("minutes screened reported", res.minutes_screened == 2.0, f"got {res.minutes_screened}")


print("\n── C3: noise-only and short bursts not flagged ────────────────────")
noise_only = screen_ictal(_rec(5.0 * rng.randn(n)))
check("pure noise → no candidates", noise_only.n_candidates == 0,
      f"got {noise_only.n_candidates}")

sig_short = 5.0 * rng.randn(n)
c0 = int(40 * SF)
c1 = int(45 * SF)   # only 5 s < min_event_s (10 s)
ts = np.arange(c0, c1) / SF
sig_short[c0:c1] += 150.0 * _chirp(ts, 8.0, 6.0)
short = screen_ictal(_rec(sig_short))
check("sub-threshold-duration burst → not flagged", short.n_candidates == 0,
      f"got {short.n_candidates}")


print("\n── C3: serialization ──────────────────────────────────────────────")
check("summary JSON-serializable", isinstance(json.dumps(summarize_ictal(res)), str))


print(f"\n{'='*60}\n  PASS: {n_pass}\n  FAIL: {n_fail}\n{'='*60}")
if n_fail:
    sys.exit(1)
