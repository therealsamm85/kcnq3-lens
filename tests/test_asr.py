"""Tests for B2 — ASR burst correction (preprocessing/asr.py).

asrpy is not installed here, so this exercises the burst-limiter fallback."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.readers.base import EEGRecording
from src.preprocessing.asr import run_asr, summarize_asr

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


SF = 100.0
n = int(60 * SF)
rng = np.random.RandomState(0)
data = (10.0 * rng.randn(2, n)).astype(np.float32)   # clean ±10 µV
# A single high-amplitude burst on ch0 at 30–31 s.
b0, b1 = int(30 * SF), int(31 * SF)
data[0, b0:b1] += 500.0

rec = EEGRecording(
    path=Path("/tmp/asr.eeg"), sfreq=SF, n_channels=2, duration_s=n / SF,
    channel_names=["C3", "C4"], n_channels_in_file=2, eeg_channel_indices=[0, 1],
    format_name="synthetic",
)
rec._full_data = data


print("\n── B2: burst-limiter fallback ─────────────────────────────────────")

res = run_asr(rec, cutoff=5.0)
check("ASR ran (available)", res.available is True, f"notes={res.notes}")
check("fallback backend is burst_limiter (no asrpy)",
      res.backend == "burst_limiter", f"got {res.backend}")
check("asrpy note present", any("asrpy not installed" in n for n in res.notes))
check("epileptiform-recheck caveat present", any("re-check spike" in n for n in res.notes))

clean = res.cleaned_recording._full_data
check("burst amplitude reduced", clean[0, b0:b1].max() < data[0, b0:b1].max() / 5,
      f"before={data[0,b0:b1].max():.0f} after={clean[0,b0:b1].max():.0f}")
check("clean region unchanged (first 10 s)",
      np.allclose(clean[0, :int(10 * SF)], data[0, :int(10 * SF)], atol=1e-3))
check("only a small fraction corrected", 0.0 < res.fraction_corrected < 0.05,
      f"got {res.fraction_corrected}")


print("\n── B2: clean signal barely touched ────────────────────────────────")

clean_rec = EEGRecording(
    path=Path("/tmp/c.eeg"), sfreq=SF, n_channels=2, duration_s=n / SF,
    channel_names=["C3", "C4"], n_channels_in_file=2, eeg_channel_indices=[0, 1],
    format_name="synthetic",
)
clean_rec._full_data = (10.0 * rng.randn(2, n)).astype(np.float32)
res_clean = run_asr(clean_rec, cutoff=20.0)
check("clean recording → ~nothing corrected", res_clean.fraction_corrected < 0.01,
      f"got {res_clean.fraction_corrected}")

check("summary JSON-serializable", isinstance(json.dumps(summarize_asr(res)), str))


print(f"\n{'='*60}\n  PASS: {n_pass}\n  FAIL: {n_fail}\n{'='*60}")
if n_fail:
    sys.exit(1)
