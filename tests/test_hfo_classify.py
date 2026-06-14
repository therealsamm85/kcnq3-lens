"""Tests for C2 — two-stage HFO classification (analyses/hfo_classify.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyses.hfo_classify import classify_hfos, summarize_hfo_classify

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


def hfo(peak_s, freq=150.0, dur_ms=40.0, rms_z=5.0, spike=False):
    return {"peak_s": peak_s, "peak_freq_hz": freq, "duration_ms": dur_ms,
            "rms_z": rms_z, "co_occurs_with_spike": spike}


print("\n── C2: per-event classification ───────────────────────────────────")

events = [
    hfo(1.0, freq=150, dur_ms=40),               # 6 cycles → real
    hfo(2.0, freq=150, dur_ms=10),               # 1.5 cycles → artifact (blip)
    hfo(3.0, freq=150, dur_ms=40, rms_z=50),     # extreme amplitude → artifact (pop)
    hfo(4.0, freq=40, dur_ms=40),                # out of band → artifact
    hfo(5.0, freq=200, dur_ms=50, spike=True),   # 10 cycles + detector spike flag → spkHFO
]
res = classify_hfos(events)
cls = [e["classification"] for e in res.per_event]
check("genuine ripple → real", cls[0] == "real", f"got {cls[0]}")
check("too-few-cycles blip → artifact", cls[1] == "artifact", f"got {cls[1]}")
check("extreme amplitude → artifact", cls[2] == "artifact", f"got {cls[2]}")
check("out-of-band → artifact", cls[3] == "artifact", f"got {cls[3]}")
check("detector spike-flag → real_spike_coupled", cls[4] == "real_spike_coupled", f"got {cls[4]}")
check("counts: 3 artifact", res.n_artifact == 3, f"got {res.n_artifact}")
check("counts: 2 real (incl. spkHFO)", res.n_real == 2, f"got {res.n_real}")
check("counts: 1 spike-coupled", res.n_spike_coupled == 1, f"got {res.n_spike_coupled}")
check("n_cycles recorded", res.per_event[0]["n_cycles"] == 6.0, f"got {res.per_event[0]['n_cycles']}")


print("\n── C2: spike coupling via supplied spike times ────────────────────")

ev2 = [hfo(10.0, freq=180, dur_ms=40)]            # real, no detector flag
coupled = classify_hfos(ev2, spike_events=[{"time_s": 10.02}])  # 20 ms away ≤ 50 ms
check("HFO within coupling window of a spike → spkHFO",
      coupled.per_event[0]["classification"] == "real_spike_coupled")
uncoupled = classify_hfos(ev2, spike_events=[{"time_s": 10.5}])  # 500 ms away
check("HFO far from any spike → plain real",
      uncoupled.per_event[0]["classification"] == "real")


print("\n── C2: honesty + degenerate ───────────────────────────────────────")
check("eHFO not claimed (epileptogenic disclaimer present)",
      any("eHFO" in n and "NOT claimed" in n for n in res.notes))
empty = classify_hfos([])
check("no candidates → zeros + note", empty.n_input == 0 and empty.n_real == 0
      and any("no HFO candidates" in n for n in empty.notes))
check("summary JSON-serializable", isinstance(json.dumps(summarize_hfo_classify(res)), str))


print(f"\n{'='*60}\n  PASS: {n_pass}\n  FAIL: {n_fail}\n{'='*60}")
if n_fail:
    sys.exit(1)
