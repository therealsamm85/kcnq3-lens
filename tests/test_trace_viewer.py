"""Tests for A4 — raw-trace viewer (utils/trace_viewer.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.readers.base import EEGRecording
from src.utils.trace_viewer import (
    read_trace_window, render_trace_window, to_mne_raw, _MAX_CHANNELS,
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


def _make_index_rec(n_ch=4, sfreq=100.0, seconds=60):
    """Each channel holds the absolute sample index, so a window's first value
    must equal start_sample — a sample-accuracy ground truth."""
    n = int(seconds * sfreq)
    data = np.tile(np.arange(n, dtype=np.float32), (n_ch, 1))
    rec = EEGRecording(
        path=Path("/tmp/v.eeg"), sfreq=sfreq, n_channels=n_ch,
        duration_s=n / sfreq, channel_names=[f"C{i}" for i in range(n_ch)],
        n_channels_in_file=n_ch, eeg_channel_indices=list(range(n_ch)),
        format_name="synthetic",
    )
    rec._full_data = data
    return rec


sf = 100.0
rec = _make_index_rec(n_ch=4, sfreq=sf, seconds=60)

print("\n── A4: sample-accurate window read ────────────────────────────────")

names, t, data = read_trace_window(rec, start_s=5.0, duration_s=2.0)
check("window has correct length (2s × 100Hz = 200)", data.shape[1] == 200, f"got {data.shape}")
check("window starts at the right sample (5s → index 500)", data[0, 0] == 500.0,
      f"got {data[0,0]}")
check("window is contiguous (sample index increments)",
      bool(np.all(np.diff(data[0]) == 1.0)))
check("time axis matches (starts at 5.0s)", abs(t[0] - 5.0) < 1e-9)

# Window crossing a 30 s epoch boundary stays contiguous.
_n, _t, d2 = read_trace_window(rec, start_s=29.0, duration_s=2.0)
check("epoch-boundary window starts at sample 2900", d2[0, 0] == 2900.0, f"got {d2[0,0]}")
check("epoch-boundary window stays contiguous (no gap at 3000)",
      bool(np.all(np.diff(d2[0]) == 1.0)))

# Channel subset selection preserves order.
names_sub, _t, d3 = read_trace_window(rec, 0.0, 1.0, channels=["C2", "C0"])
check("channel subset selected in requested order", names_sub == ["C2", "C0"])

# Channel cap.
big = _make_index_rec(n_ch=30, seconds=60)
names_big, _t, dbig = read_trace_window(big, 0.0, 1.0)
check(f"channel count capped at {_MAX_CHANNELS}", dbig.shape[0] == _MAX_CHANNELS,
      f"got {dbig.shape[0]}")


print("\n── A4: figure rendering ───────────────────────────────────────────")

fig = render_trace_window(rec, 0.0, 5.0, channels=["C0", "C1", "C2"])
ax = fig.axes[0]
check("figure has one axes", len(fig.axes) == 1)
check("one line per channel (no events)", len(ax.lines) == 3, f"got {len(ax.lines)}")
check("channel labels on y-axis", [lbl.get_text() for lbl in ax.get_yticklabels()] == ["C0", "C1", "C2"])

fig2 = render_trace_window(rec, 0.0, 5.0, channels=["C0", "C1"],
                           event_times_s=[2.0, 99.0])  # 99s is out of window
ax2 = fig2.axes[0]
check("in-range event adds a marker line; out-of-range does not",
      len(ax2.lines) == 3, f"got {len(ax2.lines)} (expected 2 traces + 1 event)")


print("\n── A4: MNE bridge for desktop browser ─────────────────────────────")

raw = to_mne_raw(rec, max_seconds=10.0)
check("RawArray has the EEG channels", raw.ch_names == ["C0", "C1", "C2", "C3"])
check("RawArray sfreq preserved", abs(raw.info["sfreq"] - sf) < 1e-9)
# µV → V conversion: sample index 500 µV → 500e-6 V at t=5s.
val = raw.get_data(picks=[0])[0, 500]
check("data scaled µV→V for MNE (500 µV → 5e-4 V)", abs(val - 500e-6) < 1e-9,
      f"got {val}")


print("\n── A4: audit regressions — short recording + duplicate names ──────")
# A recording shorter than one 30 s epoch must still render its samples.
short = _make_index_rec(n_ch=2, sfreq=100.0, seconds=3)   # 3 s < 30 s epoch
_n, _t, sd = read_trace_window(short, 0.0, 3.0)
check("sub-30s recording returns its samples (not a blank window)",
      sd.shape[1] == 300, f"got {sd.shape}")

# Trailing partial epoch (47 s, not a multiple of 30) read in full from memory.
rec47 = _make_index_rec(n_ch=2, sfreq=100.0, seconds=47)
_n, _t, d47 = read_trace_window(rec47, 0.0, 47.0)
check("47s in-memory recording read in full (no dropped tail)",
      d47.shape[1] == 4700 and d47[0, -1] == 4699.0, f"got {d47.shape}")

# Duplicate channel names must not alias to one file index.
dup = EEGRecording(
    path=Path("/tmp/dup.eeg"), sfreq=100.0, n_channels=3, duration_s=60,
    channel_names=["EEG", "EEG", "Cz"], n_channels_in_file=3,
    eeg_channel_indices=[0, 1, 2], format_name="synthetic")
dup._full_data = np.array([np.full(6000, 1.0), np.full(6000, 2.0), np.full(6000, 3.0)],
                          dtype=np.float32)
raw_dup = to_mne_raw(dup, max_seconds=10.0)
rows = raw_dup.get_data()[:, 0] * 1e6   # back to µV
check("duplicate-named channels keep distinct data (no aliasing)",
      list(rows) == [1.0, 2.0, 3.0], f"got {list(rows)}")


print(f"\n{'='*60}\n  PASS: {n_pass}\n  FAIL: {n_fail}\n{'='*60}")
if n_fail:
    sys.exit(1)
