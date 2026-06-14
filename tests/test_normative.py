"""Tests for D1 — age-normative qEEG z-scores (analyses/normative.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyses.normative import (
    compute_normative_z, summarize_normative, render_normative_md, NORMS_PLACEHOLDER,
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


# findings shaped as the runner produces them.
findings = {
    "background": {"posterior_dominant_rhythm_hz": 4.0, "delta_alpha_ratio": 5.0},
}


print("\n── D1: z-score engine ─────────────────────────────────────────────")

# Age 5 → pdr norm bin (4,6): mean 8, sd 1 → z = (4-8)/1 = -4.
res = compute_normative_z(findings, age_years=5.0)
by = {p.metric: p for p in res.points}
check("PDR z computed", by["pdr_hz"].z is not None)
check("severely slow PDR → z ≈ -4", abs(by["pdr_hz"].z + 4.0) < 0.01, f"got {by['pdr_hz'].z}")
check("norm mean/sd recorded", by["pdr_hz"].norm_mean == 8.0 and by["pdr_hz"].norm_sd == 1.0)
check("delta/alpha z computed and positive (5.0 >> norm 1.8)",
      by["delta_alpha_ratio"].z is not None and by["delta_alpha_ratio"].z > 0)

# A value exactly at the norm mean → z ≈ 0.
at_mean = compute_normative_z({"background": {"posterior_dominant_rhythm_hz": 8.0}}, 5.0)
check("value at norm mean → z ≈ 0", abs(at_mean.points[0].z) < 1e-9, f"got {at_mean.points[0].z}")


print("\n── D1: honesty (UNVERIFIED placeholder) ───────────────────────────")
check("placeholder norms are flagged unverified",
      all(p.norm_verified is False for p in res.points))
check("any_verified is False", res.any_verified is False)
check("note warns norms are unverified + not for clinical use",
      any("UNVERIFIED" in n and "clinical" in n for n in res.notes))
md = render_normative_md(res)
check("render leads with a do-not-use banner",
      "not for clinical use" in md and "UNVERIFIED" in md)


print("\n── D1: coverage + degenerate ──────────────────────────────────────")

# Age outside coverage → z None + note.
old = compute_normative_z(findings, age_years=25.0)
check("age outside norm coverage → z None + note",
      old.points[0].z is None and "outside norm coverage" in old.points[0].note)

# Metric absent from findings → skipped (not errored).
empty = compute_normative_z({"background": {}}, 5.0)
check("absent metrics skipped → no points + note", len(empty.points) == 0
      and any("no normed metrics" in n for n in empty.notes))

# Pluggable: a verified norm flips the banner off.
custom = {"pdr_hz": {"source": "MyPeds 2026", "verified": True,
                     "bins": [(4, 6, 8.0, 1.0)]}}
res_v = compute_normative_z(findings, 5.0, norms=custom)
check("verified norms set any_verified True", res_v.any_verified is True)
check("verified norms → no unverified banner in render",
      "not for clinical use" not in render_normative_md(res_v))

check("summary JSON-serializable", isinstance(json.dumps(summarize_normative(res)), str))


print(f"\n{'='*60}\n  PASS: {n_pass}\n  FAIL: {n_fail}\n{'='*60}")
if n_fail:
    sys.exit(1)
