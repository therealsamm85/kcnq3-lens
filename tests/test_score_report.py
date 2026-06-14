"""Tests for D2 — SCORE/IFCN structured report (reports/score_report.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reports.score_report import build_score_report, render_score_markdown

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


findings = {
    "background": {"posterior_dominant_rhythm_hz": 4.5, "delta_alpha_ratio": 3.2,
                   "pdr_aperiodic_corrected_hz": 5.4},
    "morphology": {"events_per_minute": 14.3, "channel": "Cz"},
    "swi": {"swi_n3_only_pct": 38.0, "csws_criterion_met": False},
    "ictal": {"n_candidates": 2},
    "hfo_ripples": {"rate_per_min": 1.2},
    "spike_average": {"field_spread": "bilateral"},
}


print("\n── D2: section mapping ────────────────────────────────────────────")
rep = build_score_report(findings)

check("background section has PDR + 1/f-corrected",
      any("Posterior dominant rhythm: 4.5 Hz" in s and "5.4" in s
          for s in rep.sections["Background activity"]))
check("interictal section has spike rate + channel",
      any("14.3/min" in s and "Cz" in s for s in rep.sections["Interictal epileptiform activity"]))
check("interictal section has SWI + below-threshold tag",
      any("Spike-wave index (N3): 38%" in s and "below CSWS" in s
          for s in rep.sections["Interictal epileptiform activity"]))
check("interictal section has averaged-spike field spread",
      any("bilateral" in s for s in rep.sections["Interictal epileptiform activity"]))
check("ictal section reports the 2 candidates as screening-only",
      any("2 candidate" in s and "review" in s for s in rep.sections["Ictal findings"]))
check("other-quantitative has HFO rate",
      any("HFO/ripple rate: 1.2/min" in s for s in rep.sections["Other quantitative findings"]))
check("reactivity reported as not assessed (honesty)",
      any("not assessed" in s for s in rep.sections["Background activity"]))


print("\n── D2: impression + notes ─────────────────────────────────────────")
check("impression carries the headline numbers",
      any("PDR 4.5 Hz" in i for i in rep.impression)
      and any("14.3/min" in i for i in rep.impression))
check("note states it is not a clinician's SCORE report",
      any("NOT a substitute" in n for n in rep.notes))


print("\n── D2: missing analyses → not assessed (no invention) ─────────────")
sparse = build_score_report({"background": {"posterior_dominant_rhythm_hz": 9.0}})
check("absent sleep → 'not assessed'",
      any("not assessed" in s for s in sparse.sections["Sleep"]))
check("absent ictal screening → 'not run'",
      any("not run" in s for s in sparse.sections["Ictal findings"]))
check("absent interictal → 'No interictal ... quantified'",
      any("No interictal" in s for s in sparse.sections["Interictal epileptiform activity"]))


print("\n── D2: audit regression — non-finite never renders ────────────────")
nan_findings = {
    "background": {"posterior_dominant_rhythm_hz": float("nan"),
                   "delta_alpha_ratio": float("inf")},
    "morphology": {"events_per_minute": float("nan")},
    "swi": {"swi_n3_only_pct": float("nan"), "csws_criterion_met": True},
    "ictal": {"n_candidates": float("nan")},
    "hfo_ripples": {"rate_per_min": float("inf")},
}
import re
nan_rep = build_score_report(nan_findings)
nan_md = render_score_markdown(nan_rep).lower()
# Word-boundary search so legit words ("dominant") don't false-match.
check("no standalone 'nan' token leaks into the report",
      re.search(r"\bnan\b", nan_md) is None, nan_md[:200])
check("no standalone 'inf' token leaks into the report",
      re.search(r"\binf\b", nan_md) is None)
check("non-finite PDR → 'not assessed' (not a real value)",
      any("not assessed" in s for s in nan_rep.sections["Background activity"]))
check("non-finite SWI → CSWS criterion is NOT asserted MET on garbage",
      not any("CSWS criterion MET" in s for s in nan_rep.sections["Interictal epileptiform activity"]))
check("NaN n_candidates → 'not run' (NOT a false 'no seizures')",
      any("not run" in s for s in nan_rep.sections["Ictal findings"])
      and not any("No electrographic" in s for s in nan_rep.sections["Ictal findings"]))


print("\n── D2: render + serialization ─────────────────────────────────────")
md = render_score_markdown(rep)
check("markdown has all five sections + impression",
      all(h in md for h in ("Background activity", "Sleep",
                            "Interictal epileptiform activity", "Ictal findings",
                            "Other quantitative findings", "Impression")))
check("empty findings → no crash, impression placeholder",
      build_score_report({}).impression[0].startswith("Insufficient"))
check("report JSON-serializable",
      isinstance(json.dumps({"sections": rep.sections, "impression": rep.impression,
                             "notes": rep.notes}), str))


print(f"\n{'='*60}\n  PASS: {n_pass}\n  FAIL: {n_fail}\n{'='*60}")
if n_fail:
    sys.exit(1)
