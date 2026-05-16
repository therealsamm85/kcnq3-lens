"""Rigorous tests for the registry de-identification + validation layer.

Five layers:
- L1 Schema: positive + negative cases for every schema field
- L2 De-id : fuzz the builder with PHI-stuffed inputs; assert NO PHI
            leaks to the output regardless of input
- L3 Round-trip: build → validate → re-validate after JSON round-trip
- L4 PHI scanner: every PHI pattern detected; clean inputs accepted
- L5 Adversarial: weird inputs (unicode, deep nests, schema-version
                  attacks, link-to-nonexistent IDs) all fail safely
"""

from __future__ import annotations

import json
import random
import re
import string
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Ensure repo root is importable when run as `python -m tests.test_registry`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.registry import (
    SCHEMA_VERSION, Consent, CURRENT_CONSENT_VERSION, make_consent,
    SubmissionInput, BuildError, build_submission,
    validate_submission, scan_for_phi, is_clean,
    bucket_age_years, bucket_duration_hours,
    AGE_BUCKETS, DURATION_BUCKETS,
)
from src.registry import phi_check
from src.registry import schema as _schema


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


def section(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


def _good_consent() -> Consent:
    return Consent(version=CURRENT_CONSENT_VERSION, given=True,
                   given_at_month="2026-05")


def _good_input(**over) -> SubmissionInput:
    base = dict(
        variant_gene="KCNQ3",
        variant_protein="p.Arg230His",
        variant_type="missense_GoF",
        age_years=5.0,
        sex="F",
        country_region="DE",
        duration_hours=23.9,
        had_sleep=True,
        montage="10-20_monopolar",
        n_channels=19,
    )
    base.update(over)
    return SubmissionInput(**base)


def _good_findings() -> dict:
    return {
        "background": {"pdr_hz": 7.8},
        "swi": {
            "csws_criterion_met": False,
            "csws_threshold_pct": 85.0,
            "swi_per_stage_pct": {"WAKE": 0.0, "N1": 5.0, "N2": 8.0,
                                   "N3": 22.0, "REM": 4.0},
        },
        "spindles": {
            "density_per_minute": 0.6,
            "age_normative_range": [0.8, 1.5],
            "interpretation": "below",
        },
        "state_split": {
            "activation_factor": 7.2,
            "activation_label": "moderate",
        },
        "morphology": {
            "events_per_minute": 25.0,
            "pct_complex_spike_wave": 42.5,
        },
        "sleep_stages": {
            "stage_pct": {"WAKE": 10.0, "N1": 5.0, "N2": 45.0,
                           "N3": 25.0, "REM": 15.0},
        },
        "sleep_architecture": {"n_cycles": 4},
        "quality": {"grade": "B"},
    }


# ═══════════════════════════════════════════════════════════════════════
# L1 — SCHEMA POSITIVE / NEGATIVE
# ═══════════════════════════════════════════════════════════════════════

section("L1 Schema — positive case (golden path)")

sub = build_submission(
    findings=_good_findings(),
    user_input=_good_input(),
    consent=_good_consent(),
    tool_version="0.12.1",
)
ok, errs = validate_submission(sub)
check("golden submission validates", ok, "; ".join(errs))
check("golden submission has schema_version 1",
      sub["schema_version"] == SCHEMA_VERSION)
check("golden submission_id is uuid4",
      bool(_schema.UUID4_RE.match(sub["submission_id"])))
check("golden submitted_at_month is YYYY-MM",
      bool(_schema.SUBMITTED_AT_MONTH_RE.match(sub["submitted_at_month"])))
check("golden age bucket is 5-7", sub["subject"]["age_years_bucket"] == "5-7")
check("golden duration bucket is 12-24",
      sub["recording"]["duration_hours_bucket"] == "12-24")
check("golden findings.spindle_density present",
      "spindle_density_per_min_central" in sub["findings"])


section("L1 Schema — negative cases (every rule has a failing example)")

# Gene
try:
    build_submission(findings={}, user_input=_good_input(variant_gene="kcnq3"),
                     consent=_good_consent(), tool_version="0.12.1")
    check("lowercase gene rejected", False)
except BuildError:
    check("lowercase gene rejected", True)

try:
    build_submission(findings={},
                     user_input=_good_input(variant_gene="K"),
                     consent=_good_consent(), tool_version="0.12.1")
    check("too-short gene rejected", False)
except BuildError:
    check("too-short gene rejected", True)

try:
    build_submission(findings={},
                     user_input=_good_input(
                         variant_gene="GENE_WITH_UNDERSCORE"),
                     consent=_good_consent(), tool_version="0.12.1")
    check("gene with underscore rejected", False)
except BuildError:
    check("gene with underscore rejected", True)

# Variant protein
try:
    build_submission(findings={},
                     user_input=_good_input(variant_protein="R230H"),
                     consent=_good_consent(), tool_version="0.12.1")
    check("variant without 'p.' prefix rejected", False)
except BuildError:
    check("variant without 'p.' prefix rejected", True)

try:
    build_submission(findings={},
                     user_input=_good_input(
                         variant_protein="p.JohnSmith"),
                     consent=_good_consent(), tool_version="0.12.1")
    check("variant with name-like text rejected", False)
except BuildError:
    check("variant with name-like text rejected", True)

# Variant type
try:
    build_submission(findings={},
                     user_input=_good_input(variant_type="GoF"),
                     consent=_good_consent(), tool_version="0.12.1")
    check("non-vocab variant_type rejected", False)
except BuildError:
    check("non-vocab variant_type rejected", True)

# Sex
try:
    build_submission(findings={},
                     user_input=_good_input(sex="female"),
                     consent=_good_consent(), tool_version="0.12.1")
    check("free-text sex rejected", False)
except BuildError:
    check("free-text sex rejected", True)

# Country
try:
    build_submission(findings={},
                     user_input=_good_input(country_region="Germany"),
                     consent=_good_consent(), tool_version="0.12.1")
    check("non-ISO country rejected", False)
except BuildError:
    check("non-ISO country rejected", True)

# Age
try:
    build_submission(findings={},
                     user_input=_good_input(age_years=-3),
                     consent=_good_consent(), tool_version="0.12.1")
    check("negative age rejected", False)
except BuildError:
    check("negative age rejected", True)

try:
    build_submission(findings={},
                     user_input=_good_input(age_years=float("nan")),
                     consent=_good_consent(), tool_version="0.12.1")
    check("NaN age rejected", False)
except BuildError:
    check("NaN age rejected", True)

# Montage
try:
    build_submission(findings={},
                     user_input=_good_input(montage="custom_clinic"),
                     consent=_good_consent(), tool_version="0.12.1")
    check("non-vocab montage rejected", False)
except BuildError:
    check("non-vocab montage rejected", True)

# n_channels
try:
    build_submission(findings={},
                     user_input=_good_input(n_channels=999),
                     consent=_good_consent(), tool_version="0.12.1")
    check("n_channels > 256 rejected", False)
except BuildError:
    check("n_channels > 256 rejected", True)

# Consent
try:
    bad_consent = Consent(version=CURRENT_CONSENT_VERSION,
                          given=False, given_at_month="2026-05")
    build_submission(findings={}, user_input=_good_input(),
                     consent=bad_consent, tool_version="0.12.1")
    check("consent.given=False rejected", False)
except BuildError:
    check("consent.given=False rejected", True)

try:
    old_consent = Consent(version=0, given=True, given_at_month="2026-05")
    build_submission(findings={}, user_input=_good_input(),
                     consent=old_consent, tool_version="0.12.1")
    check("stale consent version rejected", False)
except BuildError:
    check("stale consent version rejected", True)

# Intervention name length
try:
    long_name = "A" * 100
    build_submission(findings={}, user_input=_good_input(
        intervention_type="medication", intervention_name=long_name,
        intervention_record_kind="post"),
        consent=_good_consent(), tool_version="0.12.1")
    check("intervention_name >64 chars rejected", False)
except BuildError:
    check("intervention_name >64 chars rejected", True)

# Intervention without name
try:
    build_submission(findings={}, user_input=_good_input(
        intervention_type="medication", intervention_name="",
        intervention_record_kind="post"),
        consent=_good_consent(), tool_version="0.12.1")
    check("intervention_type without name rejected", False)
except BuildError:
    check("intervention_type without name rejected", True)


# ═══════════════════════════════════════════════════════════════════════
# L2 — DE-IDENTIFICATION FUZZ
# ═══════════════════════════════════════════════════════════════════════

section("L2 De-id fuzz — PHI cannot leak via findings dict")

PHI_STRINGS = [
    "the reference patient Schmidt",
    "patient John Doe",
    "DOB 2020-01-15",
    "recorded 15.01.2020 at home",
    "MRN 123456789",
    "contact: family@example.com",
    "+49 30 12345678",
    "C:\\Users\\Alice\\eeg\\file.edf",
    "/home/parent/clinical/recording.edf",
    "my daughter screamed during",
    "neurologist: Dr. Hans Müller",
    "scheduled appointment 2026-06-01",
]

def _spike_with_phi(findings: dict) -> dict:
    """Stuff PHI into many places in a findings dict — known keys with
    string values, unknown keys, nested dicts, lists, etc. All of this
    MUST be invisible in the output."""
    out = json.loads(json.dumps(findings))  # deep copy
    for phi in PHI_STRINGS:
        # Inject as values where the extractor expects numbers (rejected
        # silently), and as new keys (ignored by the allowlist).
        out.setdefault("background", {})["unauthorized_note"] = phi
        out.setdefault("morphology", {})["channel_label"] = phi
        out.setdefault("free_text_artifact_log"
                       , []).append(phi)
        out.setdefault("filename_hint", phi)
        out.setdefault("doctor_note", {"text": phi, "ts": "2025-01-02"})
    # Also: nested structure should NOT survive
    out["nested"] = {"deep": {"deeper": {"deepest": PHI_STRINGS[0]}}}
    # Inject PHI in legitimate KEYS but as wrong types
    out["spindles"]["density_per_minute"] = (
        "0.6/min for the reference patient"  # wrong type → must be dropped
    )
    return out

fuzz_findings = _spike_with_phi(_good_findings())
sub_fuzz = build_submission(
    findings=fuzz_findings, user_input=_good_input(),
    consent=_good_consent(), tool_version="0.12.1",
)
sub_fuzz_json = json.dumps(sub_fuzz)

for phi in PHI_STRINGS:
    check(
        f"PHI '{phi[:30]}...' not present in built submission",
        phi not in sub_fuzz_json,
    )

# Unknown keys from the findings input must be absent
check("unknown 'free_text_artifact_log' key absent from output",
      "free_text_artifact_log" not in sub_fuzz_json)
check("unknown 'doctor_note' key absent from output",
      "doctor_note" not in sub_fuzz_json)
check("nested.deep.deeper key absent from output",
      "deepest" not in sub_fuzz_json)
# Wrong-type spindle density was dropped (string not number)
check("invalid-type spindle density was dropped",
      "spindle_density_per_min_central" not in sub_fuzz["findings"])


section("L2 De-id fuzz — 1000 randomized inputs, no crash, no leak")

random.seed(0xDE1D)

def _rand_string(n: int) -> str:
    pool = string.ascii_letters + string.digits + " ./-:@"
    return "".join(random.choice(pool) for _ in range(n))

def _rand_findings() -> dict:
    # Build a chaotic findings dict with random keys + values, mostly
    # not matching the schema. Some values are numbers, some PHI strings.
    keys = ["background", "swi", "spindles", "state_split", "morphology",
            "sleep_stages", "quality", "junk_a", "junk_b", "junk_c"]
    out: dict = {}
    for k in random.sample(keys, k=random.randint(2, len(keys))):
        sub = {}
        for _ in range(random.randint(1, 6)):
            inner_k = _rand_string(random.randint(3, 12))
            v_type = random.choice(["str", "num", "neg", "nan", "list", "dict"])
            if v_type == "str":
                sub[inner_k] = random.choice(PHI_STRINGS + [_rand_string(20)])
            elif v_type == "num":
                sub[inner_k] = random.uniform(-1000, 1000)
            elif v_type == "neg":
                sub[inner_k] = -abs(random.random())
            elif v_type == "nan":
                sub[inner_k] = float("nan")
            elif v_type == "list":
                sub[inner_k] = [random.random() for _ in range(3)]
            else:
                sub[inner_k] = {"x": random.choice(PHI_STRINGS)}
        out[k] = sub
    # Sometimes insert valid fields by accident
    if random.random() < 0.5:
        out.setdefault("background", {})["pdr_hz"] = random.uniform(1, 15)
    if random.random() < 0.5:
        out.setdefault("morphology", {})[
            "events_per_minute"] = random.uniform(0, 500)
    return out

leaked = 0
crashed = 0
built = 0
for i in range(1000):
    try:
        s = build_submission(
            findings=_rand_findings(),
            user_input=_good_input(),
            consent=_good_consent(),
            tool_version="0.12.1",
        )
        built += 1
        body = json.dumps(s)
        if any(phi in body for phi in PHI_STRINGS):
            leaked += 1
    except BuildError:
        # Allowed — some random inputs may trip the PHI scan via the
        # legitimate intervention_name field (we keep that field None in
        # _good_input though, so it shouldn't). Any BuildError that
        # isn't a leak is fine.
        pass
    except Exception as e:  # noqa: BLE001
        crashed += 1
        if crashed <= 3:
            print(f"    crash: {type(e).__name__}: {e}")

check(f"1000-input fuzz: zero PHI leaks (built {built})", leaked == 0,
      f"leaks={leaked}")
check("1000-input fuzz: zero crashes", crashed == 0, f"crashes={crashed}")


# ═══════════════════════════════════════════════════════════════════════
# L3 — ROUND-TRIP
# ═══════════════════════════════════════════════════════════════════════

section("L3 Round-trip — JSON serialize → parse → validate")

sub_a = build_submission(
    findings=_good_findings(),
    user_input=_good_input(),
    consent=_good_consent(),
    tool_version="0.12.1",
)
serialized = json.dumps(sub_a)
reparsed = json.loads(serialized)
check("JSON serialize round-trip preserves shape", sub_a == reparsed)
ok, errs = validate_submission(reparsed)
check("Reparsed submission still validates", ok, "; ".join(errs))


# ═══════════════════════════════════════════════════════════════════════
# L4 — PHI SCANNER
# ═══════════════════════════════════════════════════════════════════════

section("L4 PHI scanner — every pattern detected, clean inputs accepted")

check("date YYYY-MM-DD detected",
      bool(phi_check.scan_for_phi({"x": "see 2025-01-15"})))
check("numeric date DD.MM.YYYY detected",
      bool(phi_check.scan_for_phi({"x": "15.01.2025"})))
check("email detected",
      bool(phi_check.scan_for_phi({"x": "a@b.co"})))
check("long number detected",
      bool(phi_check.scan_for_phi({"x": "MRN 1234567"})))
check("phone-like detected",
      bool(phi_check.scan_for_phi({"x": "+49 30 12345678"})))
check("unix path detected",
      bool(phi_check.scan_for_phi({"x": "/home/me/recording.edf"})))
check("windows path detected",
      bool(phi_check.scan_for_phi({"x": "C:\\Users\\me\\recording.edf"})))
check("name-like 'John Doe' detected",
      bool(phi_check.scan_for_phi({"x": "John Doe"})))
check("narrative 'my daughter' detected",
      bool(phi_check.scan_for_phi({"x": "my daughter is sleeping"})))
check("string too long detected",
      bool(phi_check.scan_for_phi({"x": "A" * 200})))

check("clean submission has no PHI findings",
      is_clean(sub_a))
check("variant gene 'KCNQ3' passes scanner (uppercase short token)",
      not phi_check.scan_for_phi({"variant_gene": "KCNQ3"}))
check("variant protein 'p.Arg230His' passes scanner",
      not phi_check.scan_for_phi({"variant_protein": "p.Arg230His"}))
check("YYYY-MM by itself does NOT trigger date detector",
      not phi_check.scan_for_phi({"submitted_at_month": "2026-05"}))


# ═══════════════════════════════════════════════════════════════════════
# L5 — ADVERSARIAL
# ═══════════════════════════════════════════════════════════════════════

section("L5 Adversarial — weird inputs must fail safely, never crash")

# Unicode + emoji in intervention name — must trip PHI scan if too long
# or pass cleanly if short and ASCII-y.
try:
    sub_u = build_submission(
        findings=_good_findings(),
        user_input=_good_input(
            intervention_type="medication",
            intervention_name="sultiam",
            intervention_record_kind="post",
        ),
        consent=_good_consent(),
        tool_version="0.12.1",
    )
    ok, _ = validate_submission(sub_u)
    check("ASCII intervention name 'sultiam' validates", ok)
except BuildError as e:
    check("ASCII intervention name 'sultiam' validates", False, str(e))

# Try injecting a date inside intervention_name
try:
    build_submission(
        findings=_good_findings(),
        user_input=_good_input(
            intervention_type="medication",
            intervention_name="started 2025-01-15",
            intervention_record_kind="post",
        ),
        consent=_good_consent(),
        tool_version="0.12.1",
    )
    check("date inside intervention_name rejected by PHI scan", False)
except BuildError:
    check("date inside intervention_name rejected by PHI scan", True)

# Submission with extra top-level key
sub_b = build_submission(
    findings=_good_findings(), user_input=_good_input(),
    consent=_good_consent(), tool_version="0.12.1",
)
sub_b["extra_field"] = "surprise"
ok, errs = validate_submission(sub_b)
check("validator rejects unknown top-level key", not ok)

# Schema-version downgrade attempt
sub_c = build_submission(
    findings=_good_findings(), user_input=_good_input(),
    consent=_good_consent(), tool_version="0.12.1",
)
sub_c["schema_version"] = 0
ok, errs = validate_submission(sub_c)
check("validator rejects mismatched schema_version", not ok)

# Bogus submission_id
sub_d = build_submission(
    findings=_good_findings(), user_input=_good_input(),
    consent=_good_consent(), tool_version="0.12.1",
)
sub_d["submission_id"] = "not-a-uuid"
ok, errs = validate_submission(sub_d)
check("validator rejects non-uuid submission_id", not ok)

# Linked-pre id pointing to non-uuid garbage
try:
    build_submission(
        findings=_good_findings(),
        user_input=_good_input(
            intervention_type="medication", intervention_name="sultiam",
            intervention_record_kind="post",
            linked_pre_submission_id="garbage",
        ),
        consent=_good_consent(),
        tool_version="0.12.1",
    )
    check("non-uuid linked_pre_submission_id rejected", False)
except BuildError:
    check("non-uuid linked_pre_submission_id rejected", True)

# Inf / -Inf as activation factor in findings — must be dropped
findings_inf = _good_findings()
findings_inf["state_split"]["activation_factor"] = float("inf")
sub_e = build_submission(
    findings=findings_inf, user_input=_good_input(),
    consent=_good_consent(), tool_version="0.12.1",
)
check("Inf activation_factor dropped (not included)",
      "activation_factor" not in sub_e["findings"])

# NaN in PDR — must be dropped
findings_nan = _good_findings()
findings_nan["background"]["pdr_hz"] = float("nan")
sub_f = build_submission(
    findings=findings_nan, user_input=_good_input(),
    consent=_good_consent(), tool_version="0.12.1",
)
check("NaN PDR dropped (not included)",
      "background_pdr_hz" not in sub_f["findings"])

# Extra key in findings rejected by validator
sub_g = build_submission(
    findings=_good_findings(), user_input=_good_input(),
    consent=_good_consent(), tool_version="0.12.1",
)
sub_g["findings"]["secret_key"] = "x"
ok, errs = validate_submission(sub_g)
check("validator rejects unknown findings key", not ok)

# Deeply nested object — builder shouldn't accept it (allowlist), and
# even if injected by mutation, validator should reject.
sub_h = build_submission(
    findings=_good_findings(), user_input=_good_input(),
    consent=_good_consent(), tool_version="0.12.1",
)
sub_h["subject"]["nested"] = {"a": {"b": {"c": "deep"}}}
ok, errs = validate_submission(sub_h)
check("validator rejects unknown subject key", not ok)


# ═══════════════════════════════════════════════════════════════════════
# Bucket boundary tests
# ═══════════════════════════════════════════════════════════════════════

section("Bucketing — boundary correctness")

check("age 0 → '0-1'", bucket_age_years(0) == "0-1")
check("age 0.99 → '0-1'", bucket_age_years(0.99) == "0-1")
check("age 1.0 → '1-2'", bucket_age_years(1.0) == "1-2")
check("age 5.0 → '5-7'", bucket_age_years(5.0) == "5-7")
check("age 4.99 → '3-5'", bucket_age_years(4.99) == "3-5")
check("age 30 → '30+'", bucket_age_years(30) == "30+")
check("age 100 → '30+'", bucket_age_years(100) == "30+")
check("age None → None", bucket_age_years(None) is None)
check("age 'five' → None", bucket_age_years("five") is None)
check("age NaN → None", bucket_age_years(float("nan")) is None)
check("age -1 → None", bucket_age_years(-1) is None)

check("duration 0 → '<1'", bucket_duration_hours(0) == "<1")
check("duration 23.9 → '12-24'", bucket_duration_hours(23.9) == "12-24")
check("duration 24 → '24-48'", bucket_duration_hours(24) == "24-48")
check("duration 999 → '48+'", bucket_duration_hours(999) == "48+")


# ═══════════════════════════════════════════════════════════════════════
# v0.12.3 — upload helpers + submissions log
# ═══════════════════════════════════════════════════════════════════════

section("v0.12.3 — upload helpers")

from src.registry.upload import (
    build_issue_url, submission_summary_md, to_jsonl_line,
    DEFAULT_OWNER, DEFAULT_REPO,
)
import urllib.parse as _urlparse

sub_for_url = build_submission(
    findings=_good_findings(), user_input=_good_input(),
    consent=_good_consent(), tool_version="0.12.3",
)
url = build_issue_url(sub_for_url, owner="alice", repo="myreg")
check("issue url uses correct owner/repo",
      url.startswith("https://github.com/alice/myreg/issues/new?"))
parsed = _urlparse.urlparse(url)
qs = _urlparse.parse_qs(parsed.query)
check("issue url has title", "title" in qs and qs["title"])
check("issue url has body", "body" in qs and qs["body"])
check("issue url has labels", "labels" in qs and qs["labels"])
body = qs["body"][0]
check("issue body contains submission JSON code fence",
      "```json" in body and "submission_id" in body)
check("issue body references consent_v1.md",
      "consent_v1.md" in body)
check("issue body does NOT contain free-text PHI markers",
      not any(s in body for s in [
          "patient", "DOB", "my daughter", "@example.com",
      ]))

# Default owner/repo are configurable via env
import os as _os
_os.environ["KCNQ3_REGISTRY_OWNER"] = "envowner"
_os.environ["KCNQ3_REGISTRY_REPO"] = "envrepo"
# Re-import module fresh to pick up env
import importlib as _il
from src.registry import upload as _up_mod
_il.reload(_up_mod)
url2 = _up_mod.build_issue_url(sub_for_url)
check("env-var overrides apply", "/envowner/envrepo/" in url2)
_os.environ.pop("KCNQ3_REGISTRY_OWNER", None)
_os.environ.pop("KCNQ3_REGISTRY_REPO", None)
_il.reload(_up_mod)

# Summary contains de-id stamp, no exact age/duration
preview = submission_summary_md(sub_for_url)
check("summary contains 'will be sent'",
      "everything that will be sent" in preview)
check("summary mentions 'exact age **not** shared'",
      "exact age **not** shared" in preview)
check("summary lists submission_id",
      sub_for_url["submission_id"] in preview)

# JSONL line is single-line and round-trips
line = to_jsonl_line(sub_for_url)
check("JSONL line is single-line",
      "\n" not in line)
check("JSONL line round-trips through json.loads",
      json.loads(line) == sub_for_url)


section("v0.12.3 — submissions_log table")

import tempfile as _tf3
from src.longitudinal import db as _db_v123
_db_v123.reset_init_cache_for_tests()
log_dir = Path(_tf3.mkdtemp(prefix="kcnq3_log_"))
import os as _os2
_os2.environ["KCNQ3_LENS_DATA"] = str(log_dir)

# record_submission + list_submissions_log
rid = _db_v123.record_submission(
    submission_id=sub_for_url["submission_id"],
    submission=sub_for_url,
    issue_url="https://github.com/x/y/issues/123",
)
check("record_submission returns positive id", rid > 0)
hist = _db_v123.list_submissions_log()
check("list_submissions_log returns the row", len(hist) == 1)
check("logged submission_id matches",
      hist[0]["submission_id"] == sub_for_url["submission_id"])
check("logged submission JSON round-trips",
      hist[0]["submission"]["subject"]["variant_gene"] == "KCNQ3")

# find_submission_in_log by id
found = _db_v123.find_submission_in_log(sub_for_url["submission_id"])
check("find_submission_in_log returns the row",
      found is not None
      and found["submission_id"] == sub_for_url["submission_id"])
missing = _db_v123.find_submission_in_log("not-a-real-id")
check("find_submission_in_log returns None for unknown id",
      missing is None)

# UNIQUE constraint: re-inserting same submission_id fails
import sqlite3 as _sqlite3
try:
    _db_v123.record_submission(
        submission_id=sub_for_url["submission_id"],
        submission=sub_for_url,
        issue_url="duplicate-attempt",
    )
    check("duplicate submission_id is rejected by UNIQUE", False)
except _sqlite3.IntegrityError:
    check("duplicate submission_id is rejected by UNIQUE", True)


# ═══════════════════════════════════════════════════════════════════════
# v0.12.4 — aggregates download, cache, lookup, percentile
# ═══════════════════════════════════════════════════════════════════════

section("v0.12.4 — aggregates lookup + percentile rank")

from src.registry import aggregates as _agg
import tempfile as _tf4, time as _time

# Build a synthetic aggregates payload mirroring what the registry CI
# would produce
fake_agg = {
    "schema_version": 1,
    "generated_at_utc": "2026-05-01T00:00:00+00:00",
    "k_min": 5,
    "n_submissions": 30,
    "n_cells_published": 2,
    "cells": [
        {
            "cell": {
                "level": "gene_protein_age_sex",
                "variant_gene": "KCNQ3",
                "variant_protein": "p.Arg230His",
                "age_years_bucket": "5-7",
                "sex": "F",
            },
            "n": 12,
            "stats": {
                "background_pdr_hz": {
                    "n": 12, "mean": 7.5, "sd": 0.6, "median": 7.5,
                    "p10": 6.5, "p25": 7.0, "p75": 8.0, "p90": 8.5,
                    "min": 6.0, "max": 9.0,
                },
                "spindle_density_per_min_central": {
                    "n": 12, "mean": 0.7, "sd": 0.2, "median": 0.7,
                    "p10": 0.4, "p25": 0.55, "p75": 0.85, "p90": 1.0,
                },
            },
            "categorical": {
                "csws_criterion_met": {"true": 2, "false": 10},
            },
        },
        {
            "cell": {
                "level": "gene",
                "variant_gene": "KCNQ3",
            },
            "n": 30,
            "stats": {
                "background_pdr_hz": {
                    "n": 30, "mean": 7.4, "sd": 0.7, "median": 7.4,
                    "p10": 6.4, "p25": 6.9, "p75": 7.9, "p90": 8.4,
                },
            },
            "categorical": {},
        },
    ],
}

# Structural validation
check("valid aggregates pass shape check",
      _agg._validate_aggregates_shape(fake_agg))
check("non-dict rejected",
      not _agg._validate_aggregates_shape([]))
check("wrong schema_version rejected",
      not _agg._validate_aggregates_shape({"schema_version": 99,
                                            "cells": []}))
check("missing cells rejected",
      not _agg._validate_aggregates_shape({"schema_version": 1}))

# Find best cell — finest match
best = _agg.find_best_cell(
    fake_agg, variant_gene="KCNQ3",
    variant_protein="p.Arg230His",
    age_years_bucket="5-7", sex="F",
)
check("finest-level match returned",
      best is not None
      and best["cell"]["level"] == "gene_protein_age_sex")

# Fall back to coarser cell when finest doesn't exist
fallback = _agg.find_best_cell(
    fake_agg, variant_gene="KCNQ3",
    variant_protein="p.Arg230His",
    age_years_bucket="10-13", sex="M",
)
check("falls back to gene when finer cells absent",
      fallback is not None
      and fallback["cell"]["level"] == "gene")

# No match → None
none_match = _agg.find_best_cell(fake_agg, variant_gene="SCN1A")
check("unknown gene returns None", none_match is None)

# Percentile rank — value at median is ~50
stat = best["stats"]["background_pdr_hz"]
p50 = _agg.percentile_rank(7.5, stat)
check(f"value at median → ~50pct (got {p50})", abs(p50 - 50.0) < 1e-6)
p10_val = _agg.percentile_rank(6.5, stat)
check(f"value at p10 → ~10pct (got {p10_val})", abs(p10_val - 10.0) < 1e-6)
p90_val = _agg.percentile_rank(8.5, stat)
check(f"value at p90 → ~90pct (got {p90_val})", abs(p90_val - 90.0) < 1e-6)
below_min = _agg.percentile_rank(2.0, stat)
check("value below min clamps to 0",
      below_min is not None and below_min <= 1e-6)
above_max = _agg.percentile_rank(99.0, stat)
check("value above max clamps to 100",
      above_max is not None and above_max >= 100.0 - 1e-6)

# Interpolation between anchors
between = _agg.percentile_rank(7.25, stat)  # halfway between p25 and median
check(f"interpolated percentile (got {between})",
      between is not None and 25.0 < between < 50.0)

# Bad inputs
check("None value returns None",
      _agg.percentile_rank(None, stat) is None)
check("string value returns None",
      _agg.percentile_rank("seven", stat) is None)
check("empty stat returns None",
      _agg.percentile_rank(7.5, {}) is None)
check("stat with only one anchor returns None",
      _agg.percentile_rank(7.5, {"median": 7.5}) is None)


section("v0.12.4 — cache load/save + cohort summary")

# Cache round-trip via env-isolated dir
cache_dir = Path(_tf4.mkdtemp(prefix="kcnq3_aggcache_"))
import os as _os4
_os4.environ["KCNQ3_LENS_DATA"] = str(cache_dir)

cache_obj = _agg.AggregatesCache(
    fetched_at=_time.time(), source_url="https://example/aggregates.json",
    payload=fake_agg,
)
_agg.save_cache(cache_obj)
loaded = _agg.load_cache()
check("cache round-trip preserves payload",
      loaded is not None and loaded.payload == fake_agg)
check("cache round-trip preserves source_url",
      loaded.source_url == "https://example/aggregates.json")

# Corrupt cache → None
(cache_dir / "aggregates_cache.json").write_text("{ not valid json")
check("corrupt cache returns None", _agg.load_cache() is None)

# Cohort summary string
summary = _agg.cohort_summary(best)
check("cohort_summary mentions gene", "KCNQ3" in summary)
check("cohort_summary mentions n", "n=12" in summary)
check("cohort_summary handles None gracefully",
      _agg.cohort_summary(None) == "no matching cohort")


# ═══════════════════════════════════════════════════════════════════════
# C10 — schema_version edge cases
# ═══════════════════════════════════════════════════════════════════════

section("C10 — schema_version edge cases (validator rejects non-integer / out-of-range)")

# 1.0 (float) — Python's {1, 2} set membership: 1.0 == 1 is True in Python,
# so we must validate that the validator explicitly checks for int type.
_sv_float = dict(
    build_submission(findings=_good_findings(), user_input=_good_input(),
                     consent=_good_consent(), tool_version="0.13.0")
)
_sv_float["schema_version"] = 1.0
_ok_float, _errs_float = validate_submission(_sv_float)
check("schema_version=1.0 (float) rejected",
      not _ok_float,
      f"errors: {_errs_float}")

# -1 — negative integer
_sv_neg = dict(
    build_submission(findings=_good_findings(), user_input=_good_input(),
                     consent=_good_consent(), tool_version="0.13.0")
)
_sv_neg["schema_version"] = -1
_ok_neg, _errs_neg = validate_submission(_sv_neg)
check("schema_version=-1 rejected",
      not _ok_neg,
      f"errors: {_errs_neg}")

# 999999 — enormous integer
_sv_big = dict(
    build_submission(findings=_good_findings(), user_input=_good_input(),
                     consent=_good_consent(), tool_version="0.13.0")
)
_sv_big["schema_version"] = 999999
_ok_big, _errs_big = validate_submission(_sv_big)
check("schema_version=999999 rejected",
      not _ok_big,
      f"errors: {_errs_big}")

# "2" — string instead of int
_sv_str = dict(
    build_submission(findings=_good_findings(), user_input=_good_input(),
                     consent=_good_consent(), tool_version="0.13.0")
)
_sv_str["schema_version"] = "2"
_ok_str, _errs_str = validate_submission(_sv_str)
check('schema_version="2" (string) rejected',
      not _ok_str,
      f"errors: {_errs_str}")


# ═══════════════════════════════════════════════════════════════════════
# Track B — Privacy + PHI hardening patches (B1–B4)
# ═══════════════════════════════════════════════════════════════════════

section("B1 — NFKC Unicode normalisation + ASCII guard")

# Cyrillic А (U+0410) looks identical to Latin A — must be caught after NFKC
# normalisation because "Аnna Smith" doesn't match [A-Z][a-z]{1,} on raw bytes
# but DOES after NFKC → "Anna Smith".
_cyrillic_name = "Аnna Smith"   # Cyrillic А + "nna Smith"
check(
    "Cyrillic-A homoglyph name detected after NFKC normalisation",
    bool(phi_check.scan_for_phi({"x": _cyrillic_name})),
)

# Plain ASCII name must still be caught (regression guard)
check(
    "ASCII 'Anna Smith' still detected",
    bool(phi_check.scan_for_phi({"x": "Anna Smith"})),
)

# Combining diacritics — é is a non-ASCII letter so it gets flagged too
# (homoglyph guard). This is conservative on purpose.
_combining_flagged = "café menu"   # é (non-ASCII letter) → flagged
check(
    "String with non-ASCII letters (e.g. é) flagged by homoglyph guard",
    bool(phi_check.scan_for_phi({"x": _combining_flagged})),
)

# ASCII guard: non-ASCII letter in intervention_name must raise BuildError
try:
    build_submission(
        findings=_good_findings(),
        user_input=_good_input(
            intervention_type="medication",
            intervention_name="sultiamé",   # é is non-ASCII
            intervention_record_kind="post",
        ),
        consent=_good_consent(),
        tool_version="0.13.4",
    )
    check("non-ASCII intervention_name rejected by ASCII guard", False)
except BuildError as _e_b1:
    check(
        "non-ASCII intervention_name rejected by ASCII guard",
        "non-ASCII" in str(_e_b1),
    )

# Pure ASCII intervention name still works
try:
    _sub_ascii = build_submission(
        findings=_good_findings(),
        user_input=_good_input(
            intervention_type="medication",
            intervention_name="sultiam",
            intervention_record_kind="post",
        ),
        consent=_good_consent(),
        tool_version="0.13.4",
    )
    check("pure ASCII intervention_name accepted after ASCII guard", True)
except BuildError as _e_b1b:
    check("pure ASCII intervention_name accepted after ASCII guard",
          False, str(_e_b1b))


section("B2 — SSN, German insurance number, IBAN patterns")

# US SSN
check(
    "US SSN '123-45-6789' detected",
    bool(phi_check.scan_for_phi({"x": "SSN: 123-45-6789"})),
)

# German Versicherungsnummer (coarse: letter + 9 digits)
check(
    "German insurance number 'A123456789' detected",
    bool(phi_check.scan_for_phi({"x": "Versicherung A123456789"})),
)

# IBAN-like (DE-format)
check(
    "IBAN-like 'DE89370400440532013000' detected",
    bool(phi_check.scan_for_phi({"x": "IBAN DE89370400440532013000"})),
)

# Clean short digit string must NOT fire SSN pattern
check(
    "short '12-34' does NOT fire SSN pattern",
    not phi_check.scan_for_phi({"x": "12-34"}),
)


section("B3 — Aggregates download size cap")

import unittest.mock as _mock5
import tempfile as _tf_b3
import os as _os_b3
from src.registry import aggregates as _agg_b3
import json as _json_b3

# Build a minimal valid aggregates payload bigger than _MAX_AGG_BYTES
_big_payload_b3 = {
    "schema_version": 1,
    "cells": [
        {
            "cell": {"level": "gene", "variant_gene": "KCNQ3"},
            "n": 1,
            "stats": {"background_pdr_hz": {"median": 7.0,
                                             "p10": 5.0, "p90": 10.0}},
        }
    ],
    "_padding": "x" * (5 * 1024 * 1024 + 1),   # exceed 5 MB
}
_big_raw_b3 = _json_b3.dumps(_big_payload_b3).encode("utf-8")


class _FakeResp_B3:
    def read(self, n=-1):
        if n == -1:
            return _big_raw_b3
        return _big_raw_b3[:n]
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass


_tf_b3_dir = _tf_b3.mkdtemp(prefix="kcnq3_b3_")
_os_b3.environ["KCNQ3_LENS_DATA"] = _tf_b3_dir
with _mock5.patch("src.registry.aggregates._urlreq.urlopen",
                  return_value=_FakeResp_B3()):
    try:
        _agg_b3.fetch_aggregates()
        check("oversized aggregates payload raises ValueError", False)
    except ValueError as _ve_b3:
        check(
            "oversized aggregates payload raises ValueError",
            "exceeds" in str(_ve_b3),
        )

# Verify cache file was NOT written (cache not poisoned)
_cache_p_b3 = Path(_tf_b3_dir) / "aggregates_cache.json"
check("cache not poisoned after oversized payload", not _cache_p_b3.exists())


section("B4 — Auto-record: build_issue_url purity + record_submission wiring")

import tempfile as _tf_b4
import os as _os_b4
from src.registry.upload import build_issue_url as _build_url_b4

_sub_b4 = build_submission(
    findings=_good_findings(),
    user_input=_good_input(),
    consent=_good_consent(),
    tool_version="0.13.4",
)
_url_b4 = _build_url_b4(_sub_b4)
check(
    "build_issue_url returns a GitHub issues URL",
    _url_b4.startswith("https://github.com/") and "issues/new" in _url_b4,
)
check(
    "build_issue_url URL contains submission_id in body",
    _sub_b4["submission_id"] in _url_b4,
)

# record_submission stores the row and list_submissions_log finds it
_tf_b4_dir = _tf_b4.mkdtemp(prefix="kcnq3_b4_")
_os_b4.environ["KCNQ3_LENS_DATA"] = _tf_b4_dir
from src.longitudinal import db as _db_b4
_db_b4.record_submission(
    submission_id=_sub_b4["submission_id"],
    submission=_sub_b4,
    issue_url=_url_b4,
)
_log_b4 = _db_b4.list_submissions_log()
check(
    "record_submission stores row retrievable by list_submissions_log",
    any(r["submission_id"] == _sub_b4["submission_id"] for r in _log_b4),
)

# Calling record_submission a second time raises IntegrityError (UNIQUE
# constraint); the UI layer (app.py) suppresses it with try/except.
import sqlite3 as _sqlite3_b4
_raised_integrity = False
try:
    _db_b4.record_submission(
        submission_id=_sub_b4["submission_id"],
        submission=_sub_b4,
        issue_url=_url_b4,
    )
except _sqlite3_b4.IntegrityError:
    _raised_integrity = True
check(
    "duplicate record_submission raises IntegrityError (UI must catch it)",
    _raised_integrity,
)


# ─── Final ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  PASS: {n_pass}")
print(f"  FAIL: {n_fail}")
print(f"{'='*60}")
if n_fail > 0:
    print("\nFailed:")
    for name in failed:
        print(f"  - {name}")
    sys.exit(1)
