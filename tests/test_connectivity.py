"""Tests for src/analyses/connectivity.py — debiased wPLI (Wave 6)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.readers.base import EEGRecording
from src.analyses.connectivity import compute_connectivity, summarize_connectivity

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


def _mk(data, names, sf=100.0):
    n = data.shape[1]
    rec = EEGRecording(
        path=Path("/tmp/c.eeg"), sfreq=sf, n_channels=len(names),
        duration_s=n / sf, channel_names=names, n_channels_in_file=len(names),
        eeg_channel_indices=list(range(len(names))), format_name="syn",
    )
    rec._full_data = data.astype(np.float32)
    return rec


print("\n── Wave 6: wPLI connectivity ──────────────────────────────────────")

sf = 100.0
n = int(300 * sf)
t = np.arange(n) / sf
f0 = 10.0  # alpha
A = np.sin(2 * np.pi * f0 * t)
B = np.sin(2 * np.pi * f0 * t - np.pi / 2)       # 90° lag = genuine coupling
C = np.random.RandomState(0).randn(n)            # independent
D = A.copy()                                     # zero-lag = volume conduction
rec = _mk(np.vstack([A, B, C, D]) * 20, ["C3", "C4", "O1", "O2"], sf)

res = compute_connectivity(rec)
mat = np.array(res.matrices_by_band["alpha"])
check("wPLI detects genuine phase-lagged coupling (C3-C4 > 0.5)",
      mat[0, 1] > 0.5, f"{mat[0,1]:.2f}")
check("wPLI ~0 for independent channels (C3-O1 < 0.3)",
      mat[0, 2] < 0.3, f"{mat[0,2]:.2f}")
check("wPLI discounts zero-lag / volume conduction (C3-O2 < 0.3)",
      mat[0, 3] < 0.3, f"{mat[0,3]:.2f}")
check("all bands present", set(res.bands) == {"delta", "theta", "alpha", "beta"})
check("mean wPLI values in [0,1]",
      all(0.0 <= v <= 1.0 for v in res.mean_wpli_by_band.values()))
check("matrix is symmetric with zero diagonal",
      np.allclose(mat, mat.T) and np.allclose(np.diag(mat), 0.0))
check("summary JSON-serializable",
      isinstance(json.dumps(summarize_connectivity(res)), str))

# <2 channels → graceful unavailable
rec1 = _mk(np.vstack([A]) * 20, ["Cz"], sf)
res1 = compute_connectivity(rec1)
check("single channel → unavailable, no crash",
      res1.n_epochs_used == 0 and any("fewer than 2" in nt for nt in res1.notes))

print(f"\n{'='*60}\n  PASS: {n_pass}\n  FAIL: {n_fail}\n{'='*60}")
if n_fail:
    sys.exit(1)
