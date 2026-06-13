"""Tests for the treatment-response dashboard (Wave 10) + shared time/polarity
helpers. Synthetic ground-truth plus a real round-trip through the SQLite
storage + diary API (not just in-memory dataclasses)."""
from __future__ import annotations

import datetime
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.longitudinal.storage import StoredEntry, save_entry, load_all_entries
from src.longitudinal.diary import DiaryEntry, append_entry, load_all_entries as load_diary
from src.longitudinal.trends import METRICS
from src.longitudinal.time_align import (
    parse_date, split_before_after, nearest_within, days_between,
)
from src.longitudinal.metric_polarity import (
    polarity_of, direction_label, CONFOUNDED_BY_MATURATION,
)
from src.longitudinal.treatment_response import (
    compute_treatment_response, summarize_treatment_response,
    render_treatment_response_md,
)

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
n_pass = n_fail = 0

_LABEL_TO_PATH = {label: path for label, path in METRICS}


def check(name, cond, detail=""):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"  {PASS} {name}")
    else:
        n_fail += 1
        print(f"  {FAIL} {name}  {detail}")


def _entry(date: str, label: str, **metric_values) -> StoredEntry:
    """Build a StoredEntry whose findings nest each metric at its real
    trends.METRICS path, so the dashboard reads it exactly as production would."""
    findings: dict = {}
    for metric, value in metric_values.items():
        path = _LABEL_TO_PATH[metric]
        cur = findings
        for k in path[:-1]:
            cur = cur.setdefault(k, {})
        cur[path[-1]] = value
    return StoredEntry(recording_date=date, label=label, findings=findings)


# ── helper unit checks ─────────────────────────────────────────────────────
print("\n── Wave 10: time_align + metric_polarity helpers ──────────────────")

check("parse_date handles plain date", parse_date("2025-03-01") == datetime.date(2025, 3, 1))
check("parse_date handles timestamp suffix",
      parse_date("2025-03-01T08:30:00") == datetime.date(2025, 3, 1))
check("parse_date rejects junk", parse_date("not-a-date") is None and parse_date("") is None)

series = [(datetime.date(2025, 1, 1), 10.0), (datetime.date(2025, 6, 1), 5.0),
          (datetime.date(2025, 3, 1), 8.0)]
before, after = split_before_after(series, datetime.date(2025, 3, 1))
check("split: pivot day counts as 'after'", after == (datetime.date(2025, 3, 1), 8.0),
      f"after={after}")
check("split: last-before is the most recent prior point",
      before == (datetime.date(2025, 1, 1), 10.0), f"before={before}")
b2, a2 = split_before_after(series, datetime.date(2024, 1, 1))
check("split: all-after → before is None", b2 is None and a2 is not None)
b3, a3 = split_before_after(series, datetime.date(2026, 1, 1))
check("split: all-before → after is None", a3 is None and b3 is not None)

near = nearest_within(series, datetime.date(2025, 3, 10), max_days=30)
check("nearest_within picks closest + signed gap",
      near == (datetime.date(2025, 3, 1), 8.0, -9), f"near={near}")
check("nearest_within respects the window",
      nearest_within(series, datetime.date(2025, 3, 10), max_days=5) is None)

check("polarity: spikes lower-better", polarity_of("spike_rate_per_min") == -1)
check("polarity: PDR higher-better", polarity_of("pdr_hz") == +1)
check("polarity: rem latency ambiguous", polarity_of("rem_latency_minutes") == 0)
check("polarity: unknown metric defaults ambiguous", polarity_of("nonsense") == 0)
check("direction: spikes falling = improved",
      direction_label("spike_rate_per_min", -5.0) == "improved")
check("direction: PDR falling = worsened",
      direction_label("pdr_hz", -1.0) == "worsened")
check("direction: dead-band = no_clear_change",
      direction_label("spike_rate_per_min", -0.1, tol=0.5) == "no_clear_change")
check("direction: ambiguous metric stays ambiguous",
      direction_label("rem_latency_minutes", -30.0) == "ambiguous")


# ── treatment-response: clear improvement ──────────────────────────────────
print("\n── Wave 10: treatment-response — clear improvement ────────────────")

entries = [
    _entry("2025-01-01", "pre", spike_rate_per_min=20.0, pdr_hz=4.0,
           swi_n3_pct=60.0, delta_alpha_ratio=5.0, spindle_density_per_min=0.5,
           sleep_efficiency_pct=70.0),
    _entry("2025-06-01", "post", spike_rate_per_min=8.0, pdr_hz=5.5,
           swi_n3_pct=20.0, delta_alpha_ratio=2.0, spindle_density_per_min=1.2,
           sleep_efficiency_pct=85.0),
]
diary = [DiaryEntry(date="2025-03-15", medication_change="Started Sultiam 5mg/kg")]
resp = compute_treatment_response(entries, diary)

check("one intervention detected", resp.n_interventions == 1 and len(resp.interventions) == 1,
      f"got {resp.n_interventions}")
iv = resp.interventions[0]
ch = {mc.metric: mc for mc in iv.metric_changes}
check("spike rate flagged improved", ch["spike_rate_per_min"].direction == "improved",
      f"got {ch['spike_rate_per_min'].direction}")
check("spike rate delta is -12", ch["spike_rate_per_min"].delta == -12.0,
      f"got {ch['spike_rate_per_min'].delta}")
check("spike rate pct change is -60%", ch["spike_rate_per_min"].pct_change == -60.0,
      f"got {ch['spike_rate_per_min'].pct_change}")
check("PDR rising flagged improved", ch["pdr_hz"].direction == "improved",
      f"got {ch['pdr_hz'].direction}")
check("PDR flagged maturation-confounded", ch["pdr_hz"].maturation_confounded is True)
check("SWI N3 falling flagged improved", ch["swi_n3_pct"].direction == "improved")
check("spindle density rising flagged improved",
      ch["spindle_density_per_min"].direction == "improved")
check("baseline/followup dates captured",
      ch["spike_rate_per_min"].baseline_date == "2025-01-01"
      and ch["spike_rate_per_min"].followup_date == "2025-06-01")
check("gap_days computed (151)", ch["spike_rate_per_min"].gap_days == 151,
      f"got {ch['spike_rate_per_min'].gap_days}")
check("global caveat about confounding present",
      any("maturation" in n for n in resp.notes))


# ── treatment-response: worsening + dead-band + not-evaluable ──────────────
print("\n── Wave 10: treatment-response — worsening / dead-band / gaps ──────")

entries_w = [
    _entry("2025-01-01", "pre", spike_rate_per_min=8.0),
    _entry("2025-06-01", "post", spike_rate_per_min=20.0),
]
resp_w = compute_treatment_response(entries_w, diary, metrics=["spike_rate_per_min"])
check("spike rate rising flagged worsened",
      resp_w.interventions[0].metric_changes[0].direction == "worsened")

entries_flat = [
    _entry("2025-01-01", "pre", spike_rate_per_min=20.0),
    _entry("2025-06-01", "post", spike_rate_per_min=19.0),  # -5% < 10% dead-band
]
resp_flat = compute_treatment_response(entries_flat, diary, metrics=["spike_rate_per_min"])
check("sub-threshold change → no_clear_change",
      resp_flat.interventions[0].metric_changes[0].direction == "no_clear_change",
      f"got {resp_flat.interventions[0].metric_changes[0].direction}")

# Only after-recordings exist → not_evaluable.
entries_after = [_entry("2025-06-01", "post", spike_rate_per_min=8.0)]
resp_after = compute_treatment_response(entries_after, diary, metrics=["spike_rate_per_min"])
check("missing baseline → not_evaluable",
      resp_after.interventions[0].metric_changes[0].direction == "not_evaluable")

# Low separation (< min_gap_days) emits a caveat.
entries_close = [
    _entry("2025-03-10", "pre", spike_rate_per_min=20.0),
    _entry("2025-03-20", "post", spike_rate_per_min=8.0),  # 5 days after pivot, 10d gap
]
resp_close = compute_treatment_response(entries_close, diary, metrics=["spike_rate_per_min"])
check("low before/after separation is flagged",
      any("low separation" in n for n in resp_close.interventions[0].notes),
      f"notes={resp_close.interventions[0].notes}")

# Ambiguous-polarity metric reports direction 'ambiguous'.
entries_amb = [
    _entry("2025-01-01", "pre", rem_latency_minutes=120.0),
    _entry("2025-06-01", "post", rem_latency_minutes=60.0),
]
resp_amb = compute_treatment_response(entries_amb, diary, metrics=["rem_latency_minutes"])
check("ambiguous-polarity metric → direction 'ambiguous'",
      resp_amb.interventions[0].metric_changes[0].direction == "ambiguous")

# No interventions → explanatory note, no crash.
resp_none = compute_treatment_response(entries, [])
check("no med changes → note, zero interventions",
      resp_none.n_interventions == 0 and any("No medication" in n for n in resp_none.notes))


# ── serialization + markdown ───────────────────────────────────────────────
print("\n── Wave 10: serialization + render ────────────────────────────────")

check("summary is JSON-serializable",
      isinstance(json.dumps(summarize_treatment_response(resp)), str))
md = render_treatment_response_md(resp)
check("markdown is a non-empty string", isinstance(md, str) and len(md) > 50)
check("markdown names the intervention", "Started Sultiam 5mg/kg" in md)
check("markdown carries a caveat line", "maturation" in md)


# ── real round-trip through SQLite storage + diary API ─────────────────────
print("\n── Wave 10: real storage/diary round-trip (reference-like 5 points) ──")

tmp = Path(tempfile.mkdtemp())
# reference-like trajectory: non-maturing PDR (~4-5 Hz), persistent spike burden.
reference = [
    _entry("2024-02-01", "age3.9", spike_rate_per_min=14.0, pdr_hz=4.0, swi_n3_pct=35.0),
    _entry("2024-08-01", "age4.4", spike_rate_per_min=15.0, pdr_hz=4.2, swi_n3_pct=40.0),
    _entry("2025-01-01", "age4.8", spike_rate_per_min=13.0, pdr_hz=4.5, swi_n3_pct=33.0),
    _entry("2025-06-01", "age5.2", spike_rate_per_min=14.0, pdr_hz=4.8, swi_n3_pct=38.0),
    _entry("2025-11-01", "age5.6", spike_rate_per_min=12.0, pdr_hz=5.0, swi_n3_pct=30.0),
]
for e in reference:
    save_entry(e, storage_dir=tmp)
append_entry(DiaryEntry(date="2025-03-01", medication_change="Started trial AED",
                        word_count=40), path=tmp)

loaded = load_all_entries(storage_dir=tmp)
loaded_diary = load_diary(path=tmp)
check("5 recordings persisted + reloaded", len(loaded) == 5, f"got {len(loaded)}")
check("nested findings survive the DB round-trip",
      loaded[0].findings.get("morphology", {}).get("events_per_minute") == 14.0,
      f"got {loaded[0].findings}")
check("diary med change persisted", any(d.medication_change for d in loaded_diary))

resp_real = compute_treatment_response(loaded, loaded_diary)
check("real-API dashboard computes one intervention", resp_real.n_interventions == 1)
real_ch = {mc.metric: mc for mc in resp_real.interventions[0].metric_changes}
# 13.0 (2025-01-01, last before) → 14.0 (2025-06-01, first after) = +1, +7.7% < 10% band
check("real spike change within dead-band → no_clear_change",
      real_ch["spike_rate_per_min"].direction == "no_clear_change",
      f"got {real_ch['spike_rate_per_min'].direction} "
      f"({real_ch['spike_rate_per_min'].pct_change}%)")
check("real-API summary serializes",
      isinstance(json.dumps(summarize_treatment_response(resp_real)), str))


print(f"\n{'='*60}\n  PASS: {n_pass}\n  FAIL: {n_fail}\n{'='*60}")
if n_fail:
    sys.exit(1)
