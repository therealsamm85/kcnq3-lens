"""Tests for src/reports/annotations.py — event export for human review (Wave 8)."""
from __future__ import annotations

import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reports.annotations import export_events, load_events

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


print("\n── Wave 8: event annotation export ────────────────────────────────")

tmp = Path(tempfile.mkdtemp(prefix="annot_"))
try:
    # Mixed event shapes: morphology-style {time_s}, plus a richer one.
    events = [
        {"time_s": 5.0},
        {"time_s": 1.0, "amplitude": 250.0, "type": "spike"},
        {"onset": 3.0, "duration": 0.07, "channel": "C3"},
    ]
    out = tmp / "sub-x_task-rest_events.tsv"
    res = export_events(events, out, detector="test", channel="Pz",
                        default_duration=0.05)

    check("all 3 events exported", res.n_events == 3, f"{res.n_events}")
    header = out.read_text().splitlines()[0]
    check("BIDS-compatible header (onset, duration first)",
          header.startswith("onset\tduration"), header)
    check("file is tab-separated", "\t" in header)

    back = load_events(out)
    check("round-trip count matches", len(back) == 3)
    check("events sorted by onset (1,3,5)",
          [e["onset"] for e in back] == [1.0, 3.0, 5.0],
          str([e["onset"] for e in back]))
    check("onset coerced to float", isinstance(back[0]["onset"], float))
    check("missing time defaults applied (duration on first event)",
          back[0]["duration"] in (0.05, 0.07),
          str(back[0]["duration"]))
    check("detector provenance preserved", back[0]["detector"] == "test")
    check("type/trial_type carried through",
          any(e["trial_type"] == "spike" for e in back))
    check("amplitude carried when present",
          any(e.get("amplitude_uv") == 250.0 for e in back))
    check("blank 'reviewed' column present for the human",
          all("reviewed" in e for e in back))

    # Reviewer marks an event, re-ingest.
    lines = out.read_text().splitlines()
    lines[1] = lines[1].rsplit("\t", 1)[0] + "\tepileptiform"
    (tmp / "rev.tsv").write_text("\n".join(lines) + "\n")
    rev = load_events(tmp / "rev.tsv")
    check("reviewer verdict re-ingests", rev[0]["reviewed"] == "epileptiform",
          rev[0]["reviewed"])

    # Events with no time are skipped, not crashed.
    res2 = export_events([{"foo": 1}, {"time_s": 2.0}], tmp / "e2.tsv")
    check("events without a time are skipped", res2.n_events == 1,
          f"{res2.n_events}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'='*60}\n  PASS: {n_pass}\n  FAIL: {n_fail}\n{'='*60}")
if n_fail:
    sys.exit(1)
