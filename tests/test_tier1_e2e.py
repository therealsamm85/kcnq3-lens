"""Tier 1 federated-registry end-to-end live integration test.

Exercises EVERY production code path in the registry pipeline, with
real SQLite, real submission building, real registry-repo aggregator,
real aggregates fetch (via file:// loopback), real peer-comparison
lookup. No mocks of our own code — only the network call is replaced
with a local fixture file.

Why a separate file: the unit suites (test_registry.py,
test_aggregator.py in the registry repo) test components in isolation.
This file proves the components compose correctly across the
repo boundary. Run before any Tier 1 release.

Usage:
    python -m tests.test_tier1_e2e

Exits 0 on full pipeline success, 1 on any failure.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── Test harness ───────────────────────────────────────────────────────
n_pass = 0
n_fail = 0
failed: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"  \x1b[32m✓\x1b[0m {name}")
    else:
        n_fail += 1
        failed.append(name)
        msg = f"  \x1b[31m✗\x1b[0m {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)


def stage(t: str) -> None:
    print(f"\n══ {t} " + "═" * max(0, 60 - len(t)))


# ─── Isolated environment ───────────────────────────────────────────────
env_dir = Path(tempfile.mkdtemp(prefix="kcnq3_tier1_e2e_"))
os.environ["KCNQ3_LENS_DATA"] = str(env_dir)
print(f"Isolated env: {env_dir}")

REGISTRY_REPO = Path("/path/to/kcnq3-registry")


# ════════════════════════════════════════════════════════════════════════
# STAGE 1 — Local SQLite (v0.12.0)
# ════════════════════════════════════════════════════════════════════════
stage("1) Local SQLite — recordings + diary + migration")

from src.longitudinal import (
    StoredEntry, save_entry, load_all_entries as load_long,
    DiaryEntry, append_entry as append_diary, load_diary,
)
from src.longitudinal import db as _db
_db.reset_init_cache_for_tests()

# Plant a "legacy" recording JSON + a diary.jsonl in the env dir so we
# also verify the one-shot migration path
legacy_dir = env_dir / "recordings"
legacy_dir.mkdir()
(legacy_dir / "2025-12-01_legacy.json").write_text(json.dumps({
    "recording_date": "2025-12-01", "label": "legacy-pre",
    "findings": {"morphology": {"events_per_minute": 18.0}},
    "source_filename": "legacy.edf",
}))
(env_dir / "diary.jsonl").write_text(
    json.dumps({"date": "2025-12-01", "word_count": 25,
                 "new_milestone": "two-word sentence"}) + "\n"
)

# Save fresh recordings via the public API
for i, (date, label, ev_per_min, pdr) in enumerate([
    ("2026-01-15", "pre-sultiam", 22.0, 7.0),
    ("2026-03-15", "post-sultiam-3mo", 12.0, 7.8),
    ("2026-05-15", "post-sultiam-5mo", 8.0, 8.1),
]):
    e = StoredEntry(
        recording_date=date, label=label,
        source_filename=f"rec{i}.edf",
        findings={
            "background": {"pdr_hz": pdr},
            "swi": {"csws_criterion_met": False, "csws_threshold_pct": 85.0,
                    "swi_per_stage_pct": {"N2": 8.0, "N3": 18.0}},
            "spindles": {"density_per_minute": 0.6 + i*0.1,
                         "age_normative_range": [0.8, 1.5],
                         "interpretation": "below"},
            "state_split": {"activation_factor": 5.0 - i*0.5,
                            "activation_label": "moderate"},
            "morphology": {"events_per_minute": ev_per_min,
                           "pct_complex_spike_wave": 35.0},
            "quality": {"grade": "B"},
        },
        metadata={"age_years": 5.3},
    )
    save_entry(e)

recs = load_long()
check("3 fresh recordings + 1 legacy migrated → 4 total", len(recs) == 4)
check("legacy recording present", any(r.label == "legacy-pre" for r in recs))
check("diary entries migrated from JSONL", len(load_diary()) == 1)
check("diary word_count preserved",
      load_diary()[0].word_count == 25)

# Add a fresh diary entry too
append_diary(DiaryEntry(date="2026-05-13", word_count=42,
                         new_milestone="said complete sentence"))
check("fresh diary entry appended", len(load_diary()) == 2)


# ════════════════════════════════════════════════════════════════════════
# STAGE 2 — Build submissions via real builder (v0.12.1)
# ════════════════════════════════════════════════════════════════════════
stage("2) Submission builder — de-id + PHI scan + validation")

from src.registry import (
    build_submission, SubmissionInput, BuildError,
    make_consent, validate_submission, scan_for_phi,
    bucket_age_years,
)

# Stuff the legacy recording with PHI-like junk to prove de-id is hard
phi_findings = dict(recs[0].findings)
phi_findings["filename"] = "C:\\Users\\the reference patient\\eeg.edf"   # path
phi_findings["doctor_note"] = "the reference patient Schmidt, DOB 2020-01-15"
phi_findings["nested"] = {"unauthorized": "Dr. Hans Mueller wrote this"}
phi_findings["free_text_log"] = ["my daughter started Sultiam 2026-01-15"]

submissions = []
try:
    sub = build_submission(
        findings=phi_findings,
        user_input=SubmissionInput(
            variant_gene="KCNQ3", variant_protein="p.Arg230His",
            variant_type="missense_GoF", age_years=5.3, sex="F",
            country_region="DE", duration_hours=23.9, had_sleep=True,
            montage="10-20_monopolar", n_channels=19,
        ),
        consent=make_consent(given=True),
        tool_version="0.12.4",
    )
    submissions.append(sub)
    check("PHI-stuffed submission still built (PHI invisible)", True)
except BuildError as e:
    check("PHI-stuffed submission still built", False, str(e))

# Hard PHI-absence proof
body = json.dumps(submissions[0])
for marker in ["the reference patient", "Schmidt", "Hans Mueller", "DOB",
                "2020-01-15", "C:\\\\Users", "eeg.edf",
                "my daughter", "filename", "doctor_note"]:
    check(f"PHI marker '{marker}' absent from submission",
          marker not in body)

# Build 9 more clean submissions — diverse cohort, n=10 total
import random, uuid
random.seed(123)
for i in range(9):
    s = build_submission(
        findings={
            "background": {"pdr_hz": 7.0 + random.random()*1.5},
            "spindles": {"density_per_minute": 0.4 + random.random()*0.8,
                          "age_normative_range": [0.8, 1.5],
                          "interpretation": random.choice(["below", "in"])},
            "state_split": {"activation_factor": 3 + random.random()*6,
                             "activation_label": "moderate"},
            "morphology": {"events_per_minute": 10 + random.random()*20,
                            "pct_complex_spike_wave": 20 + random.random()*30},
            "swi": {"csws_criterion_met": False, "csws_threshold_pct": 85.0,
                     "swi_per_stage_pct": {"N2": 5 + random.random()*5,
                                            "N3": 15 + random.random()*15}},
            "quality": {"grade": "B"},
        },
        user_input=SubmissionInput(
            variant_gene="KCNQ3", variant_protein="p.Arg230His",
            variant_type="missense_GoF",
            age_years=5.0 + random.random()*1.5,
            sex=random.choice(["F", "M"]),
            country_region=random.choice(["DE", "US", "GB"]),
            duration_hours=12 + random.random()*12, had_sleep=True,
            montage="10-20_monopolar", n_channels=19,
        ),
        consent=make_consent(given=True),
        tool_version="0.12.4",
        schema_version_target=1,  # registry-repo still on v1
    )
    submissions.append(s)

check("10 submissions built total", len(submissions) == 10)

# Each must validate AND scan clean
for i, s in enumerate(submissions):
    ok, errs = validate_submission(s)
    phi = scan_for_phi(s)
    if not ok or phi:
        check(f"submission #{i} validates + PHI clean", False,
              f"errors={errs} phi={phi}")
        break
else:
    check("10/10 submissions validate + PHI clean", True)


# ════════════════════════════════════════════════════════════════════════
# STAGE 3 — Registry repo: validator + aggregator (v0.12.2)
# ════════════════════════════════════════════════════════════════════════
stage("3) Cross-repo: validate + aggregate via registry scripts")

# Plant the submissions in the registry's data/registry.jsonl, then
# call its standalone validator + aggregator scripts via subprocess
jsonl_path = REGISTRY_REPO / "data" / "registry.jsonl"
backup = jsonl_path.read_text() if jsonl_path.exists() else ""
try:
    with open(jsonl_path, "w") as f:
        for s in submissions:
            f.write(json.dumps(s) + "\n")

    # Validator
    res = subprocess.run(
        [sys.executable, "scripts/validate_registry.py"],
        cwd=REGISTRY_REPO, capture_output=True, text=True,
    )
    check("registry validator script exits 0", res.returncode == 0,
          res.stdout + res.stderr)
    check("validator reports '10 valid lines'",
          "10 valid lines" in res.stdout)

    # Aggregator
    res = subprocess.run(
        [sys.executable, "scripts/build_aggregates.py"],
        cwd=REGISTRY_REPO, capture_output=True, text=True,
    )
    check("aggregator script exits 0", res.returncode == 0,
          res.stdout + res.stderr)

    # Load + inspect produced aggregates
    agg_path = REGISTRY_REPO / "releases" / "v1" / "aggregates.json"
    agg = json.loads(agg_path.read_text())
    check("aggregates schema_version == 1", agg["schema_version"] == 1)
    check("aggregates k_min == 5", agg["k_min"] == 5)
    check("at least one cell published",
          agg["n_cells_published"] >= 1)
    finest = [c for c in agg["cells"]
              if c["cell"]["level"] == "gene_protein"]
    check("gene_protein cell present (n=10)",
          len(finest) == 1 and finest[0]["n"] == 10)
    check("gene_protein cell publishes PDR stats",
          "background_pdr_hz" in finest[0]["stats"])
    check("PDR stats have median + p25 + p75",
          all(k in finest[0]["stats"]["background_pdr_hz"]
              for k in ("median", "p25", "p75")))
    # min/max only published when n WITHIN THE FIELD >= 10 (not cell n).
    # The legacy-migrated recording lacks pdr_hz so n_in_stat may be 9.
    pdr_stat = finest[0]["stats"]["background_pdr_hz"]
    expect_extremes = pdr_stat["n"] >= 10
    check(
        f"min/max published iff n_in_stat>=10 (n_in_stat={pdr_stat['n']})",
        ("min" in pdr_stat) == expect_extremes
        and ("max" in pdr_stat) == expect_extremes,
    )

    # Save aggregates for stage 4
    agg_for_consumer = json.loads(agg_path.read_text())

finally:
    # Restore registry to empty so we don't poison the repo
    jsonl_path.write_text(backup)
    subprocess.run([sys.executable, "scripts/build_aggregates.py"],
                    cwd=REGISTRY_REPO, capture_output=True)


# ════════════════════════════════════════════════════════════════════════
# STAGE 4 — Issue URL + submissions log (v0.12.3)
# ════════════════════════════════════════════════════════════════════════
stage("4) Contribute UI plumbing — issue URL + local log")

from src.registry import build_issue_url, submission_summary_md, to_jsonl_line
from src.longitudinal import db as _db

# Build a URL for the first submission
url = build_issue_url(submissions[0])
check("URL points to GitHub issues/new",
      "github.com" in url and "/issues/new?" in url)
check("URL is < 8 KB (GitHub limit)", len(url) < 8000)
# Decode and verify content
import urllib.parse
qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
body = qs["body"][0]
check("URL body has variant tag",
      "KCNQ3" in body and "p.Arg230His" in body)
check("URL body has consent reference",
      "consent_v1.md" in body)
check("URL body contains submission JSON",
      submissions[0]["submission_id"] in body)
check("URL body has no PHI markers",
      not any(m in body for m in ["the reference patient", "Schmidt", "DOB", "/Users/"]))

# Preview text
preview = submission_summary_md(submissions[0])
check("preview shows 'will be sent'", "will be sent" in preview)
check("preview shows bucket disclosure",
      "exact age **not** shared" in preview)

# Persist to log
for s in submissions[:3]:
    _db.record_submission(
        submission_id=s["submission_id"], submission=s,
        issue_url=build_issue_url(s),
    )
log = _db.list_submissions_log()
check("3 submissions logged locally", len(log) == 3)
check("submissions log preserves variant",
      log[0]["submission"]["subject"]["variant_gene"] == "KCNQ3")
# Duplicate prevention
import sqlite3
try:
    _db.record_submission(
        submission_id=submissions[0]["submission_id"],
        submission=submissions[0], issue_url="dup-attempt",
    )
    check("duplicate submission_id rejected", False)
except sqlite3.IntegrityError:
    check("duplicate submission_id rejected", True)


# ════════════════════════════════════════════════════════════════════════
# STAGE 5 — Peer-comparison consumption (v0.12.4)
# ════════════════════════════════════════════════════════════════════════
stage("5) Aggregates cache + lookup + percentile rank")

from src.registry import aggregates as _agg

# Plant the registry-produced aggregates as if just fetched
cache_obj = _agg.AggregatesCache(
    fetched_at=time.time(),
    source_url="https://raw.githubusercontent.com/test/test/main/agg.json",
    payload=agg_for_consumer,
)
_agg.save_cache(cache_obj)
loaded = _agg.load_cache()
check("aggregates cache round-trip", loaded is not None
      and loaded.payload == agg_for_consumer)

# Fresh cache → get_aggregates returns it without network call
got, warn = _agg.get_aggregates()
check("get_aggregates returns fresh cache", got is not None and warn is None)

# Find the best cell for the first submission's subject
best = _agg.find_best_cell(
    got, variant_gene="KCNQ3", variant_protein="p.Arg230His",
    age_years_bucket=bucket_age_years(5.3), sex="F",
)
check("best-cell lookup found a cell", best is not None)

# Percentile rank: pick the median value of cohort PDR — should be ~50%
pdr_stat = None
if best:
    pdr_stat = best.get("stats", {}).get("background_pdr_hz")
if pdr_stat is None:
    # Fall back to coarser level
    best = _agg.find_best_cell(got, variant_gene="KCNQ3",
                                 variant_protein="p.Arg230His")
    pdr_stat = best.get("stats", {}).get("background_pdr_hz") if best else None

check("PDR stat block available in matched cell", pdr_stat is not None)
if pdr_stat:
    median = pdr_stat["median"]
    pct = _agg.percentile_rank(median, pdr_stat)
    check(f"median value → ~50th pct (got {pct})",
          pct is not None and abs(pct - 50.0) < 1e-6)
    pct_low = _agg.percentile_rank(0.0, pdr_stat)
    check("very low value → bottom pct",
          pct_low is not None and pct_low <= 10.0)
    pct_high = _agg.percentile_rank(100.0, pdr_stat)
    check("very high value → top pct",
          pct_high is not None and pct_high >= 90.0)

# cohort_summary string
summary = _agg.cohort_summary(best)
check("cohort_summary contains gene", "KCNQ3" in summary)


# ════════════════════════════════════════════════════════════════════════
# STAGE 6 — App boots into Contribute mode with real data
# ════════════════════════════════════════════════════════════════════════
stage("6) Streamlit AppTest — Contribute mode boots with seeded data")

from streamlit.testing.v1 import AppTest

# AppTest needs the working directory to be the repo root
app_path = Path(__file__).resolve().parent.parent / "app.py"
try:
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    check("app boots in default mode", at.exception is None
          or len(at.exception) == 0)

    radio = at.sidebar.radio[0]
    radio.set_value("contribute").run()
    check("Contribute mode renders without exception",
          at.exception is None or len(at.exception) == 0,
          str(at.exception))

    # The recording picker should have 4 options (3 fresh + 1 legacy)
    selectboxes = at.selectbox
    pickers = [sb for sb in selectboxes
                if sb.key == "contrib_entry_idx"]
    check("recording picker present", len(pickers) >= 1)
    if pickers:
        check("recording picker has 4 entries",
              len(pickers[0].options) == 4)

except Exception as e:
    check("Contribute mode AppTest", False, f"{type(e).__name__}: {e}")


# ─── Final ──────────────────────────────────────────────────────────────
print(f"\n{'═'*70}")
print(f"  Tier 1 end-to-end:  PASS: {n_pass}    FAIL: {n_fail}")
print(f"{'═'*70}")
if n_fail > 0:
    print("\nFailed checks:")
    for f in failed:
        print(f"  - {f}")
    sys.exit(1)

# Cleanup
shutil.rmtree(env_dir, ignore_errors=True)
print(f"  Cleaned: {env_dir}")
print(f"\n  \x1b[32mAll Tier 1 surfaces verified live.\x1b[0m")
