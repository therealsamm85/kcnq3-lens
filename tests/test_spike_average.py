"""Tests for C1 — spike-triggered averaging + topography (analyses/spike_average.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.readers.base import EEGRecording
from src.analyses.spike_average import (
    compute_spike_average, summarize_spike_average, _hemisphere,
)

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
KERNEL = -200.0 * np.exp(-0.5 * (np.arange(-20, 21) / 4.0) ** 2)  # sharp spike


def _rec_with_spikes(inject_channels, times, n_ch=4, seconds=60):
    n = int(seconds * SF)
    rng = np.random.RandomState(3)
    data = (3.0 * rng.randn(n_ch, n)).astype(np.float32)
    for t in times:
        c = int(t * SF)
        for ch in inject_channels:
            data[ch, c - 20: c + 21] += KERNEL.astype(np.float32)
    rec = EEGRecording(
        path=Path("/tmp/s.eeg"), sfreq=SF, n_channels=n_ch, duration_s=n / SF,
        channel_names=[f"C{i}" for i in range(n_ch)], n_channels_in_file=n_ch,
        eeg_channel_indices=list(range(n_ch)), format_name="synthetic",
    )
    rec._full_data = data
    return rec


print("\n── C1: hemisphere parsing ─────────────────────────────────────────")
check("odd digit → left", _hemisphere("C3") == "left")
check("even digit → right", _hemisphere("C4") == "right")
check("z → midline", _hemisphere("Cz") == "mid")


print("\n── C1: focal spike (one channel) ──────────────────────────────────")
times = [3 + 3 * k for k in range(15)]   # 15 spikes
rec = _rec_with_spikes([3], times)        # C3 = left
events = [{"time_s": t} for t in times]
res = compute_spike_average(rec, events)
check("all 15 spikes averaged", res.n_spikes_averaged == 15, f"got {res.n_spikes_averaged}")
check("peak channel is the injected C3", res.peak_channel == "C3", f"got {res.peak_channel}")
check("peak latency near 0 ms", abs(res.peak_latency_ms) <= 5.0, f"got {res.peak_latency_ms}")
check("injected channel dominates the topography",
      abs(res.peak_topography["C3"]) > 5 * max(abs(res.peak_topography[c]) for c in ("C0", "C1", "C2")),
      f"topo={res.peak_topography}")
check("field spread classified focal", res.field_spread == "focal", f"got {res.field_spread}")


print("\n── C1: bilateral spike (homologous channels) ──────────────────────")
rec_b = _rec_with_spikes([2, 3], times)   # C2 right + C3 left
res_b = compute_spike_average(rec_b, events)
check("field spread classified bilateral", res_b.field_spread == "bilateral",
      f"got {res_b.field_spread}")


print("\n── C1: degenerate / honesty ───────────────────────────────────────")
empty = compute_spike_average(rec, [])
check("no events → 0 averaged + note", empty.n_spikes_averaged == 0
      and any("nothing to average" in n for n in empty.notes))
few = compute_spike_average(_rec_with_spikes([3], [3, 6, 9]), [{"time_s": t} for t in (3, 6, 9)])
check("<10 spikes → instability note", any("unstable" in n for n in few.notes))
check("source-localisation honesty note present",
      any("source localisation" in n for n in res.notes))

import json
check("summary JSON-serializable", isinstance(json.dumps(summarize_spike_average(res)), str))


print(f"\n{'='*60}\n  PASS: {n_pass}\n  FAIL: {n_fail}\n{'='*60}")
if n_fail:
    sys.exit(1)
