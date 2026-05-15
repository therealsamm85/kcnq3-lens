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
