"""Smoke test for v0.19.0 runner wiring — the 6 new analyses + SCORE report must
appear in the findings dict produced by run_all_analyses on a real run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.readers.base import EEGRecording
from src.runner import run_all_analyses

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


# Synthetic 19-channel 10-20 recording, 5 minutes @ 200 Hz.
NAMES = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
         "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
SF = 200.0
n = int(300 * SF)
rng = np.random.RandomState(0)
t = np.arange(n) / SF
data = (15.0 * rng.randn(len(NAMES), n)).astype(np.float32)
# Posterior alpha + occasional sharp spikes on Pz so detectors produce output.
for oi in (NAMES.index("O1"), NAMES.index("O2"), NAMES.index("Pz")):
    data[oi] += 25.0 * np.sin(2 * np.pi * 9 * t)
spike = 180.0 * np.exp(-0.5 * (np.arange(-8, 9) / 2.0) ** 2)
for c in range(int(5 * SF), n - 50, int(7 * SF)):
    data[NAMES.index("Pz"), c - 8:c + 9] += spike.astype(np.float32)

rec = EEGRecording(
    path=Path("/tmp/wire.eeg"), sfreq=SF, n_channels=len(NAMES), duration_s=n / SF,
    channel_names=list(NAMES), n_channels_in_file=len(NAMES),
    eeg_channel_indices=list(range(len(NAMES))), format_name="synthetic",
)
rec._full_data = data


print("\n── v0.19.0: runner wiring smoke test ──────────────────────────────")
findings = run_all_analyses(
    rec, sleep_start_epoch=0, sleep_end_epoch=10,
    wake_epoch_indices=[0, 1, 2, 3], age_years=5.0,
)

for key in ("entropy", "graph_metrics", "spike_average", "hfo_classify",
            "ictal", "normative", "score_report"):
    check(f"findings['{key}'] present", key in findings, f"keys={list(findings)[:30]}")

# The whole findings dict must still be JSON-serializable + finite (sanitize pass).
check("findings JSON-serializable (strict, no NaN/Inf)",
      isinstance(json.dumps({k: v for k, v in findings.items()
                             if not k.startswith("_")}, allow_nan=False), str))

# Spot-check shape of a couple of the new findings.
check("entropy has a metrics dict",
      isinstance(findings.get("entropy", {}).get("metrics"), dict))
check("score_report has the SCORE sections + impression",
      "sections" in findings.get("score_report", {})
      and "impression" in findings.get("score_report", {}))
check("normative ran (age supplied) and flags unverified placeholder norms",
      findings.get("normative", {}).get("any_verified") is False)
check("ictal produced a candidate count",
      "n_candidates" in findings.get("ictal", {}))

# Without an age, normative is skipped (no crash).
f2 = run_all_analyses(rec, 0, 10, [0, 1, 2, 3], age_years=None)
check("no age → normative skipped, run still completes",
      "normative" not in f2 and "score_report" in f2)


print(f"\n{'='*60}\n  PASS: {n_pass}\n  FAIL: {n_fail}\n{'='*60}")
if n_fail:
    sys.exit(1)
