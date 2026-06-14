"""Tests for A1 — annotated EDF+ write-back (reports/edf_writeback.py)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.readers.base import EEGRecording
from src.reports.edf_writeback import (
    export_annotated_edf, collect_events_from_findings,
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


def _make_rec(n_ch=3, sfreq=100.0, seconds=60, eeg_idx=None):
    n = int(seconds * sfreq)
    rng = np.random.RandomState(1)
    data = (20.0 * rng.randn(n_ch, n)).astype(np.float32)
    rec = EEGRecording(
        path=Path("/tmp/syn.eeg"), sfreq=sfreq, n_channels=n_ch,
        duration_s=n / sfreq, channel_names=[f"C{i}" for i in range(n_ch)],
        n_channels_in_file=n_ch,
        eeg_channel_indices=eeg_idx if eeg_idx is not None else list(range(n_ch)),
        format_name="synthetic",
    )
    rec._full_data = data
    return rec


print("\n── A1: annotated EDF+ write-back ──────────────────────────────────")

rec = _make_rec(n_ch=3, sfreq=100.0, seconds=60)
events = [
    {"onset_s": 5.0, "duration_s": 0.0, "label": "spike C0"},
    {"time_s": 12.3, "label": "spike C1"},          # alt onset key
    {"onset_s": 40.0, "duration_s": 1.0, "label": "HFO C2"},
    {"onset_s": 999.0, "label": "out-of-range"},     # should be dropped
]

tmp = Path(tempfile.mkdtemp()) / "out.edf"
res = export_annotated_edf(rec, tmp, events)

check("EDF file written", tmp.exists() and tmp.stat().st_size > 0)
check("3 channels written", res.channels_written == 3, f"got {res.channels_written}")
check("3 in-range annotations (out-of-range dropped)", res.n_annotations == 3,
      f"got {res.n_annotations}")
check("duration ~60s", abs(res.duration_s - 60.0) < 0.01, f"got {res.duration_s}")

# Round-trip: re-read with edfio and confirm structure survived.
import edfio
back = edfio.read_edf(tmp)
check("round-trip: 3 signals", len(back.signals) == 3, f"got {len(back.signals)}")
labels = [s.label.strip() for s in back.signals]
check("round-trip: channel labels preserved", labels == ["C0", "C1", "C2"], f"got {labels}")
ann_texts = [a.text for a in back.annotations]
check("round-trip: spike annotation present", any("spike" in t for t in ann_texts),
      f"got {ann_texts}")
check("round-trip: HFO annotation present", any("HFO" in t for t in ann_texts))

# eeg_only excludes non-EEG channels.
rec2 = _make_rec(n_ch=3, eeg_idx=[0, 1])
res2 = export_annotated_edf(rec2, Path(tempfile.mkdtemp()) / "o2.edf", [], eeg_only=True)
check("eeg_only writes only the 2 EEG channels", res2.channels_written == 2,
      f"got {res2.channels_written}")

# Flat/dead channel does not crash the physical-range handling.
rec3 = _make_rec(n_ch=2, seconds=60)
rec3._full_data[1, :] = 0.0  # dead channel
res3 = export_annotated_edf(rec3, Path(tempfile.mkdtemp()) / "o3.edf", [])
check("dead/flat channel handled (no zero-width range crash)", res3.channels_written == 2)

# collect_events_from_findings flattens _*_events with type labels.
findings = {
    "_morphology_events": [{"time_s": 1.0}, {"time_s": 2.0, "channel": "Cz"}],
    "_hfo_ripples_events": [{"time_s": 3.0}],
    "background": {"posterior_dominant_rhythm_hz": 5.0},  # ignored (not an events key)
}
collected = collect_events_from_findings(findings)
check("collect: 3 events flattened from findings", len(collected) == 3, f"got {len(collected)}")
check("collect: spike label applied", any(c["label"].startswith("spike") for c in collected))
check("collect: channel appended to label", any("Cz" in c["label"] for c in collected))
check("collect: sorted by onset", [c["onset_s"] for c in collected] == [1.0, 2.0, 3.0])


print(f"\n{'='*60}\n  PASS: {n_pass}\n  FAIL: {n_fail}\n{'='*60}")
if n_fail:
    sys.exit(1)
