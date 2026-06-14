"""Tests for B1 — ICA + component classification (preprocessing/ica.py).

mne-icalabel is not installed here, so this exercises the montage-free
frontal-weight ocular heuristic backend (mne only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.readers.base import EEGRecording
from src.preprocessing.ica import run_ica_cleanup, summarize_ica

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


def _band_power(x, sf, lo=0.3, hi=3.0):
    sos = butter(4, [lo, hi], btype="band", fs=sf, output="sos")
    return float(np.var(sosfiltfilt(sos, x)))


SF = 100.0
n = int(60 * SF)
t = np.arange(n) / SF
rng = np.random.RandomState(0)

# Frontal-dominant blink source + two independent posterior brain sources.
blink = np.zeros(n)
for c in range(int(2 * SF), n, int(3 * SF)):
    blink += np.exp(-0.5 * ((np.arange(n) - c) / (0.15 * SF)) ** 2)
brain1 = np.sin(2 * np.pi * 8 * t) + 0.3 * rng.randn(n)
brain2 = np.sin(2 * np.pi * 11 * t) + 0.3 * rng.randn(n)

data = np.vstack([
    80 * blink + 10 * brain1,   # Fp1 — frontal, blink-dominant
    75 * blink + 10 * brain2,   # Fp2 — frontal, blink-dominant
    3 * blink + 40 * brain1,    # C3 — posterior
    3 * blink + 40 * brain2,    # C4 — posterior
]).astype(np.float32)

rec = EEGRecording(
    path=Path("/tmp/ica.eeg"), sfreq=SF, n_channels=4, duration_s=n / SF,
    channel_names=["Fp1", "Fp2", "C3", "C4"], n_channels_in_file=4,
    eeg_channel_indices=[0, 1, 2, 3], format_name="synthetic",
)
rec._full_data = data


print("\n── B1: ICA ocular component removal (frontal heuristic) ───────────")

res = run_ica_cleanup(rec)
check("ICA ran (available)", res.available is True, f"notes={res.notes}")
check("fallback backend is frontal_heuristic (no mne-icalabel)",
      res.backend == "frontal_heuristic", f"got {res.backend}")
check("≥1 ocular component removed", len(res.removed_components) >= 1,
      f"got {res.removed_components}")
check("cleaned recording returned", res.cleaned_recording is not None)

if res.cleaned_recording is not None:
    clean = res.cleaned_recording._full_data
    before = _band_power(data[0], SF)        # Fp1 blink-band power before
    after = _band_power(clean[0], SF)         # after ICA removal
    check("frontal blink-band power substantially reduced",
          after < 0.5 * before, f"before={before:.0f} after={after:.0f}")
    # Posterior brain rhythm should be largely preserved.
    b_before = _band_power(data[2], SF, 6, 10)
    b_after = _band_power(clean[2], SF, 6, 10)
    check("posterior 8 Hz rhythm largely preserved",
          b_after > 0.5 * b_before, f"before={b_before:.1f} after={b_after:.1f}")


print("\n── B1: graceful degradation ───────────────────────────────────────")

rec1 = EEGRecording(
    path=Path("/tmp/x.eeg"), sfreq=SF, n_channels=2, duration_s=n / SF,
    channel_names=["C3", "C4"], n_channels_in_file=2, eeg_channel_indices=[0, 1],
    format_name="synthetic",
)
rec1._full_data = data[2:4]
res1 = run_ica_cleanup(rec1)
check("<3 channels → available False + note", res1.available is False
      and any("≥3 EEG channels" in n for n in res1.notes))

# No frontal channels → heuristic removes nothing, but does not crash.
rec_nf = EEGRecording(
    path=Path("/tmp/nf.eeg"), sfreq=SF, n_channels=4, duration_s=n / SF,
    channel_names=["C3", "C4", "P3", "P4"], n_channels_in_file=4,
    eeg_channel_indices=[0, 1, 2, 3], format_name="synthetic",
)
rec_nf._full_data = data
res_nf = run_ica_cleanup(rec_nf)
check("no frontal channels → runs, removes nothing, notes why",
      res_nf.available is True and len(res_nf.removed_components) == 0
      and any("no frontal" in n.lower() for n in res_nf.notes))

check("summary JSON-serializable", isinstance(json.dumps(summarize_ica(res)), str))


print("\n── B1: audit regressions — non-finite + length preservation ───────")
# Non-finite input degrades gracefully (available=False), not an exception.
nan_d = data.copy()
nan_d[0, 1000] = np.nan
nan_rec = EEGRecording(
    path=Path("/tmp/nanica.eeg"), sfreq=SF, n_channels=4, duration_s=n / SF,
    channel_names=["Fp1", "Fp2", "C3", "C4"], n_channels_in_file=4,
    eeg_channel_indices=[0, 1, 2, 3], format_name="synthetic")
nan_rec._full_data = nan_d
res_nan = run_ica_cleanup(nan_rec)
check("non-finite input → available False (no exception)",
      res_nan.available is False and any("non-finite" in nt for nt in res_nan.notes))

# A non-30s-multiple recording keeps its full length (no silent tail drop).
n47 = int(47 * SF)
d47 = np.vstack([80 * blink[:n47] + 10 * brain1[:n47], 75 * blink[:n47] + 10 * brain2[:n47],
                 3 * blink[:n47] + 40 * brain1[:n47], 3 * blink[:n47] + 40 * brain2[:n47]]).astype(np.float32)
rec47 = EEGRecording(
    path=Path("/tmp/ica47.eeg"), sfreq=SF, n_channels=4, duration_s=n47 / SF,
    channel_names=["Fp1", "Fp2", "C3", "C4"], n_channels_in_file=4,
    eeg_channel_indices=[0, 1, 2, 3], format_name="synthetic")
rec47._full_data = d47
res47 = run_ica_cleanup(rec47)
check("47s recording cleaned at full length (no dropped tail)",
      res47.available and res47.cleaned_recording._full_data.shape[1] == n47,
      f"got {res47.cleaned_recording._full_data.shape[1] if res47.cleaned_recording is not None else None}")


print(f"\n{'='*60}\n  PASS: {n_pass}\n  FAIL: {n_fail}\n{'='*60}")
if n_fail:
    sys.exit(1)
