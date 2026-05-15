"""Tests for v0.13.3 — automated IED detection + Schema v2 IED additions.

Run as: python -m tests.test_ied_ml_v0133
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analyses.ied_ml import (
    IEDDetectionResult,
    compute_ied_ml,
    summarize_ied_ml,
    _select_method,
    _age_flag,
)
from src.registry import schema as _schema
from src.registry.buckets import (
    bucket_ied_rate, bucket_ied_agreement, bucket_ied_rolandic,
)
from src.registry.deid import build_submission, SubmissionInput
from src.registry.consent import Consent, CURRENT_CONSENT_VERSION
from src.registry.validate import validate_submission
from src.registry.phi_check import _SKIP_PATHS


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


# ─── Synthetic EEGRecording fixture ──────────────────────────────────────────

class _SyntheticRec:
    def __init__(
        self,
        sfreq: float = 250.0,
        duration_s: float = 600.0,
        channel_names: list[str] | None = None,
        signal_override: np.ndarray | None = None,
    ):
        self.sfreq = sfreq
        self.duration_s = duration_s
        n_samples = int(sfreq * duration_s)
        self.channel_names = channel_names or ["Fz", "Cz", "C3", "C4", "T7", "T8", "Pz"]
        n_channels = len(self.channel_names)
        self.eeg_channel_indices = list(range(n_channels))

        rng = np.random.default_rng(42)
        if signal_override is not None:
            raw = signal_override
        else:
            raw = rng.standard_normal((n_channels, n_samples)).astype(np.float32) * 20.0

        self._raw = raw
        self._n_channels = n_channels
        self._n_samples = n_samples

    def channel_index(self, name: str) -> int | None:
        nu = name.upper()
        for i, ch in enumerate(self.channel_names):
            if ch.upper() == nu:
                return i
        return None

    def iter_epochs(self, epoch_seconds: float = 30.0, start: int = 0, end: int | None = None):
        ep_size = int(self.sfreq * epoch_seconds)
        n_epochs = self._n_samples // ep_size
        end = n_epochs if end is None else min(end, n_epochs)
        for ep in range(start, end):
            data = self._raw[:, ep * ep_size: (ep + 1) * ep_size]
            yield ep, data


class _SyntheticSleepStages:
    def __init__(self, n_epochs: int = 20, all_wake: bool = False):
        self.epoch_seconds = 30.0
        if all_wake:
            self.epoch_labels = ["WAKE"] * n_epochs
        else:
            self.epoch_labels = ["N2"] * n_epochs


def _make_spike_signal(
    sfreq: float, duration_s: float, n_ch: int,
    spike_times: list[float],
    focal_channels: list[int],
    add_hf_burst: bool = True,
    spike_polarity_neg: bool = True,
    spike_width_ms: float = 50.0,
    aftercoming_slow_wave: bool = True,
) -> np.ndarray:
    """Build a multi-channel signal containing synthesized epileptiform spikes.

    spike_times: list of seconds at which to place spikes.
    focal_channels: indices of channels where spike is dominant.
    """
    n_samp = int(sfreq * duration_s)
    rng = np.random.default_rng(123)
    sig = rng.standard_normal((n_ch, n_samp)).astype(np.float32) * 5.0  # low noise

    width_samp = max(2, int(round(spike_width_ms * 1e-3 * sfreq)))
    amp = 80.0 if spike_polarity_neg else -80.0  # neg amplitude (will be negated below)
    for t in spike_times:
        center = int(round(t * sfreq))
        lo = max(0, center - width_samp)
        hi = min(n_samp, center + width_samp + 1)
        # Spike: gaussian-like sharp peak (negative-going by default)
        x = np.arange(lo, hi) - center
        kern = np.exp(-(x ** 2) / (2 * (width_samp / 4.0) ** 2))
        spike = -amp * kern if spike_polarity_neg else amp * kern
        for ch in focal_channels:
            sig[ch, lo:hi] += spike.astype(np.float32)
        # HF burst (30-70 Hz) centered at peak
        if add_hf_burst:
            hf_half = int(round(0.05 * sfreq))
            hf_lo = max(0, center - hf_half)
            hf_hi = min(n_samp, center + hf_half)
            tt = np.arange(hf_lo, hf_hi) / sfreq
            hf = 30.0 * np.sin(2 * np.pi * 50.0 * tt) * np.hanning(len(tt))
            for ch in focal_channels:
                sig[ch, hf_lo:hf_hi] += hf.astype(np.float32)
        # Aftercoming slow wave (positive bump 100-300 ms after peak)
        if aftercoming_slow_wave:
            sw_lo = min(n_samp - 1, center + int(round(0.10 * sfreq)))
            sw_hi = min(n_samp, center + int(round(0.35 * sfreq)))
            if sw_hi > sw_lo + 4:
                sw_w = sw_hi - sw_lo
                sw = 40.0 * np.hanning(sw_w)  # positive bump
                for ch in focal_channels:
                    sig[ch, sw_lo:sw_hi] += sw.astype(np.float32)
    return sig


# ─── Mode selection ──────────────────────────────────────────────────────────

section("v0.13.3 — Mode selection")

check("auto + no events + no weights → unavailable",
      _select_method(None, None) == "unavailable")

check("auto + events + no weights → ensemble_heuristic",
      _select_method(None, [{"time_s": 1.0}]) == "ensemble_heuristic")

check("auto + empty events → ensemble_heuristic",
      _select_method(None, []) == "ensemble_heuristic")

# fake missing weights path
fake_missing = "/tmp/__never_exists__.pt"
check("auto + missing weights + events → ensemble_heuristic",
      _select_method(fake_missing, [{"time_s": 1.0}]) == "ensemble_heuristic")

# Create a fake weights file
with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
    tmp.write(b"fake-weights-content" * 100)
    fake_weights = tmp.name

# Whether external_spikenet vs ensemble depends on torch availability
sel = _select_method(fake_weights, [{"time_s": 1.0}])
try:
    import torch  # noqa: F401
    has_torch = True
except ImportError:
    has_torch = False

if has_torch:
    check("auto + valid weights + torch → external_spikenet",
          sel == "external_spikenet")
else:
    check("auto + valid weights + no torch → ensemble_heuristic",
          sel == "ensemble_heuristic")


# ─── Build a recording with a few real spikes ────────────────────────────────

section("v0.13.3 — Ensemble: synthetic spikes")

sfreq = 250.0
duration = 600.0
n_ch = 7  # Fz, Cz, C3, C4, T7, T8, Pz
spike_times = [30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0]
focal_chs = [1, 2]  # Cz, C3 dominant
sig = _make_spike_signal(
    sfreq, duration, n_ch, spike_times, focal_chs,
    add_hf_burst=True, aftercoming_slow_wave=True, spike_width_ms=40.0,
)
rec = _SyntheticRec(sfreq=sfreq, duration_s=duration, signal_override=sig)
stages = _SyntheticSleepStages(n_epochs=20)
morph_events = [{"time_s": t} for t in spike_times]

result_ens = compute_ied_ml(
    rec, sleep_stages=stages, morphology_events=morph_events,
    weights_path=None, method="auto", age_years=15.0,
)
check("ensemble: method == ensemble_heuristic",
      result_ens.method == "ensemble_heuristic")
check("ensemble: available=True", result_ens.available)
check("ensemble: model_license == 'rule-based'",
      result_ens.model_license == "rule-based")
check("ensemble: n_ied_candidates > 0 (got " + str(result_ens.n_ied_candidates) + ")",
      result_ens.n_ied_candidates > 0)
check("ensemble: rate_per_minute > 0",
      result_ens.rate_per_minute > 0)
check("ensemble: at least one high or medium confidence event",
      result_ens.confidence_distribution.get("high", 0)
      + result_ens.confidence_distribution.get("medium", 0) > 0)
check("ensemble: nrem_rate_per_min is set (sleep stages provided)",
      result_ens.nrem_rate_per_min is not None)
check("ensemble: disclaimer present",
      "RESEARCH METRIC" in (result_ens.disclaimer or ""))
check("ensemble: events stored internally",
      isinstance(result_ens.events, list)
      and len(result_ens.events) == result_ens.n_ied_candidates)


# ─── Rules-level checks ──────────────────────────────────────────────────────

section("v0.13.3 — Rule logic via synthesized inputs")

# Generalized event: spike on ALL channels → R3 should fail (channel_count > 3)
gen_sig = _make_spike_signal(
    sfreq, duration, n_ch, [60.0], focal_channels=list(range(n_ch)),
    add_hf_burst=True, aftercoming_slow_wave=True,
)
rec_gen = _SyntheticRec(sfreq=sfreq, duration_s=duration, signal_override=gen_sig)
res_gen = compute_ied_ml(
    rec_gen, sleep_stages=None, morphology_events=[{"time_s": 60.0}],
    weights_path=None, method="auto", age_years=15.0,
)
# We don't crash; either skipped (0/3) or kept with low confidence.
check("generalized event: no crash", res_gen.method == "ensemble_heuristic")
if res_gen.events:
    ev = res_gen.events[0]
    check("generalized event: R3 (focal) = 0",
          ev["rules_passed"][2] == 0,
          detail=f"channels_involved={ev['channels_involved']}, count={ev['channel_count']}")

# No HF burst → R2 = 0
no_hf_sig = _make_spike_signal(
    sfreq, duration, n_ch, [60.0], focal_channels=[1],
    add_hf_burst=False, aftercoming_slow_wave=True, spike_width_ms=30.0,
)
rec_nohf = _SyntheticRec(sfreq=sfreq, duration_s=duration, signal_override=no_hf_sig)
res_nohf = compute_ied_ml(
    rec_nohf, sleep_stages=None, morphology_events=[{"time_s": 60.0}],
    weights_path=None, method="auto", age_years=15.0,
)
# May or may not survive — check rules_passed
if res_nohf.events:
    ev = res_nohf.events[0]
    check("no-HF event: R2 (hf burst) = 0",
          ev["rules_passed"][1] == 0,
          detail=f"hf_burst_ratio={ev['hf_burst_ratio']}")


# ─── Age flag ────────────────────────────────────────────────────────────────

section("v0.13.3 — Age flag handling")

check("age_flag(5) == 'drift_warning'", _age_flag(5) == "drift_warning")
check("age_flag(11.9) == 'drift_warning'", _age_flag(11.9) == "drift_warning")
check("age_flag(12.0) == 'ok'", _age_flag(12.0) == "ok")
check("age_flag(15) == 'ok'", _age_flag(15) == "ok")
check("age_flag(None) == 'untested'", _age_flag(None) == "untested")
check("age_flag(NaN) == 'untested'", _age_flag(float("nan")) == "untested")
check("age_flag(-1) == 'untested'", _age_flag(-1) == "untested")

# Pediatric path: flag in result
res_ped = compute_ied_ml(
    rec, sleep_stages=stages, morphology_events=morph_events,
    weights_path=None, method="auto", age_years=5.0,
)
check("age=5: result flag = drift_warning",
      res_ped.age_appropriateness_flag == "drift_warning")

res_adult = compute_ied_ml(
    rec, sleep_stages=stages, morphology_events=morph_events,
    weights_path=None, method="auto", age_years=15.0,
)
check("age=15: result flag = ok",
      res_adult.age_appropiateness_flag if hasattr(res_adult, "age_appropiateness_flag") else
      res_adult.age_appropriateness_flag == "ok")

res_none = compute_ied_ml(
    rec, sleep_stages=stages, morphology_events=morph_events,
    weights_path=None, method="auto", age_years=None,
)
check("age=None: result flag = untested",
      res_none.age_appropriateness_flag == "untested")


# ─── Rolandic flagging ───────────────────────────────────────────────────────

section("v0.13.3 — Rolandic-benign flagging")

# Spike at C3/T7 (centrotemporal), pediatric
ct_sig = _make_spike_signal(
    sfreq, duration, n_ch, [60.0, 120.0],
    focal_channels=[2, 4],  # C3, T7
    add_hf_burst=True, aftercoming_slow_wave=False,
    spike_width_ms=30.0,
)
rec_ct = _SyntheticRec(sfreq=sfreq, duration_s=duration, signal_override=ct_sig)
res_ct = compute_ied_ml(
    rec_ct, sleep_stages=None,
    morphology_events=[{"time_s": 60.0}, {"time_s": 120.0}],
    weights_path=None, method="auto", age_years=7.0,
)
check("centrotemporal pediatric: n_likely_rolandic_benign >= 1",
      res_ct.n_likely_rolandic_benign >= 1,
      detail=f"got {res_ct.n_likely_rolandic_benign}")

# Adult age: not flagged as Rolandic
res_ct_adult = compute_ied_ml(
    rec_ct, sleep_stages=None,
    morphology_events=[{"time_s": 60.0}, {"time_s": 120.0}],
    weights_path=None, method="auto", age_years=18.0,
)
check("centrotemporal adult: no Rolandic flag",
      res_ct_adult.n_likely_rolandic_benign == 0)


# ─── Agreement metric ────────────────────────────────────────────────────────

section("v0.13.3 — Agreement-with-morphology metric")

# All synthesized spikes have HF + focal + morphology → should match
check("agreement_with_morphology_pct in [0, 100]",
      0.0 <= result_ens.agreement_with_morphology_pct <= 100.0)


# ─── Unavailable path ────────────────────────────────────────────────────────

section("v0.13.3 — Unavailable + edge cases")

res_unavail = compute_ied_ml(
    rec, sleep_stages=stages, morphology_events=None,
    weights_path=None, method="auto", age_years=10.0,
)
check("no events + no weights: method = unavailable",
      res_unavail.method == "unavailable")
check("unavailable: available=False", not res_unavail.available)
check("unavailable: n_ied_candidates == 0",
      res_unavail.n_ied_candidates == 0)
check("unavailable: model_license is None",
      res_unavail.model_license is None)

# Empty events list → ensemble with zero candidates
res_empty = compute_ied_ml(
    rec, sleep_stages=stages, morphology_events=[],
    weights_path=None, method="auto", age_years=10.0,
)
check("empty events: method = ensemble_heuristic",
      res_empty.method == "ensemble_heuristic")
check("empty events: n_ied_candidates == 0",
      res_empty.n_ied_candidates == 0)
check("empty events: no crash, rate = 0",
      res_empty.rate_per_minute == 0.0)

# NaN age
res_nan = compute_ied_ml(
    rec, sleep_stages=None, morphology_events=morph_events,
    weights_path=None, method="auto", age_years=float("nan"),
)
check("NaN age: flag = untested",
      res_nan.age_appropriateness_flag == "untested")

# NaN time_s event (must be skipped, no crash)
res_bad_t = compute_ied_ml(
    rec, sleep_stages=None,
    morphology_events=[{"time_s": float("nan")}, {"time_s": 60.0}],
    weights_path=None, method="auto", age_years=15.0,
)
check("NaN time_s: no crash",
      res_bad_t.method == "ensemble_heuristic")


# ─── External SpikeNet stub path ─────────────────────────────────────────────

section("v0.13.3 — External SpikeNet (stub) fallback")

if has_torch:
    res_sn = compute_ied_ml(
        rec, sleep_stages=stages, morphology_events=morph_events,
        weights_path=fake_weights, method="auto", age_years=15.0,
    )
    # Should fall back to ensemble_heuristic with a warning
    check("spikenet stub: falls back to ensemble_heuristic",
          res_sn.method == "ensemble_heuristic")
    check("spikenet stub: warning recorded",
          any("spikenet_stub" in w for w in res_sn.warnings),
          detail=f"warnings={res_sn.warnings}")
    check("spikenet stub: model_license carries SpikeNet provenance",
          res_sn.model_license is not None and "SpikeNet" in res_sn.model_license,
          detail=f"license={res_sn.model_license}")
    check("spikenet stub: model_version is a sha256 prefix",
          res_sn.model_version is not None and res_sn.model_version.startswith("sha256:"),
          detail=f"version={res_sn.model_version}")
else:
    check("torch not installed → skipped spikenet stub tests", True)

# Missing weights path with method=external_spikenet explicit
res_missing = compute_ied_ml(
    rec, sleep_stages=stages, morphology_events=morph_events,
    weights_path="/tmp/__definitely_missing__.pt",
    method="auto", age_years=15.0,
)
check("missing weights: falls back to ensemble",
      res_missing.method == "ensemble_heuristic")


# ─── JSON safety ─────────────────────────────────────────────────────────────

section("v0.13.3 — JSON serialization")

for label, r in [
    ("ensemble", result_ens),
    ("unavailable", res_unavail),
    ("ensemble_empty", res_empty),
]:
    summary = summarize_ied_ml(r)
    try:
        json.dumps(summary)
        check(f"summarize({label}) is JSON-serializable", True)
    except (TypeError, ValueError) as e:
        check(f"summarize({label}) is JSON-serializable", False, str(e))
    check(f"summarize({label}): no 'events' key",
          "events" not in summary)


# ─── Schema v2 constants ─────────────────────────────────────────────────────

section("v0.13.3 — Schema v2 constants & buckets")

check("SCHEMA_VERSION == 2", _schema.SCHEMA_VERSION == 2)
check("IED_METHODS defined",
      hasattr(_schema, "IED_METHODS")
      and "ensemble_heuristic" in _schema.IED_METHODS)
check("IED_RATE_BUCKETS defined",
      hasattr(_schema, "IED_RATE_BUCKETS")
      and len(_schema.IED_RATE_BUCKETS) == 6)
check("IED_AGE_FLAGS defined",
      hasattr(_schema, "IED_AGE_FLAGS")
      and "drift_warning" in _schema.IED_AGE_FLAGS)
check("IED_AGREEMENT_BUCKETS defined",
      hasattr(_schema, "IED_AGREEMENT_BUCKETS"))
check("IED_ROLANDIC_BUCKETS defined",
      hasattr(_schema, "IED_ROLANDIC_BUCKETS"))


# bucket_ied_rate
check("bucket_ied_rate(0.0) == '0'", bucket_ied_rate(0.0) == "0")
check("bucket_ied_rate(0.5) == '<1'", bucket_ied_rate(0.5) == "<1")
check("bucket_ied_rate(3.5) == '1-5'", bucket_ied_rate(3.5) == "1-5")
check("bucket_ied_rate(10.0) == '5-15'", bucket_ied_rate(10.0) == "5-15")
check("bucket_ied_rate(20.0) == '15-50'", bucket_ied_rate(20.0) == "15-50")
check("bucket_ied_rate(100.0) == '>50'", bucket_ied_rate(100.0) == ">50")
check("bucket_ied_rate(None) is None", bucket_ied_rate(None) is None)
check("bucket_ied_rate(-1) is None", bucket_ied_rate(-1) is None)

# bucket_ied_agreement
check("bucket_ied_agreement(30) == '<50'", bucket_ied_agreement(30) == "<50")
check("bucket_ied_agreement(60) == '50-75'", bucket_ied_agreement(60) == "50-75")
check("bucket_ied_agreement(80) == '75-90'", bucket_ied_agreement(80) == "75-90")
check("bucket_ied_agreement(95) == '>90'", bucket_ied_agreement(95) == ">90")
check("bucket_ied_agreement(None) is None", bucket_ied_agreement(None) is None)

# bucket_ied_rolandic
check("bucket_ied_rolandic(0) == '0'", bucket_ied_rolandic(0) == "0")
check("bucket_ied_rolandic(5) == 'small'", bucket_ied_rolandic(5) == "small")
check("bucket_ied_rolandic(20) == 'medium'", bucket_ied_rolandic(20) == "medium")
check("bucket_ied_rolandic(60) == 'large'", bucket_ied_rolandic(60) == "large")


# ─── Builder + validator end-to-end ──────────────────────────────────────────

section("v0.13.3 — build_submission with IED fields")


def _good_consent() -> Consent:
    return Consent(version=CURRENT_CONSENT_VERSION, given=True,
                   given_at_month="2026-05")


def _good_input() -> SubmissionInput:
    return SubmissionInput(
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


findings_v2 = {
    "quality": {"grade": "B"},
    "ied_ml": {
        "available": True,
        "method": "ensemble_heuristic",
        "rate_per_minute": 3.5,
        "agreement_with_morphology_pct": 85.0,
        "age_appropriateness_flag": "drift_warning",
        "n_likely_rolandic_benign": 4,
    },
}

sub_v2 = build_submission(
    findings=findings_v2,
    user_input=_good_input(),
    consent=_good_consent(),
    tool_version="0.13.3",
)
check("v2 submission: ied_method present",
      sub_v2["findings"].get("ied_method") == "ensemble_heuristic")
check("v2 submission: ied_rate_bucket = '1-5'",
      sub_v2["findings"].get("ied_rate_bucket") == "1-5",
      detail=f"got {sub_v2['findings'].get('ied_rate_bucket')!r}")
check("v2 submission: ied_age_flag = 'drift_warning'",
      sub_v2["findings"].get("ied_age_flag") == "drift_warning")
check("v2 submission: ied_agreement_bucket = '75-90'",
      sub_v2["findings"].get("ied_agreement_bucket") == "75-90",
      detail=f"got {sub_v2['findings'].get('ied_agreement_bucket')!r}")
check("v2 submission: ied_n_rolandic_benign_bucket = 'small'",
      sub_v2["findings"].get("ied_n_rolandic_benign_bucket") == "small")

# Validator accepts
ok, errs = validate_submission(sub_v2)
check("validator accepts v2 with IED fields", ok, "; ".join(errs))

# v1 target: IED fields NOT in output
sub_v1 = build_submission(
    findings=findings_v2,
    user_input=_good_input(),
    consent=_good_consent(),
    tool_version="0.13.3",
    schema_version_target=1,
)
check("v1 target: no IED fields",
      "ied_method" not in sub_v1["findings"]
      and "ied_rate_bucket" not in sub_v1["findings"])


# ─── PHI _SKIP_PATHS for IED ─────────────────────────────────────────────────

section("v0.13.3 — PHI _SKIP_PATHS for IED")

check("ied_method in _SKIP_PATHS",
      "$.findings.ied_method" in _SKIP_PATHS)
check("ied_rate_bucket in _SKIP_PATHS",
      "$.findings.ied_rate_bucket" in _SKIP_PATHS)
check("ied_age_flag in _SKIP_PATHS",
      "$.findings.ied_age_flag" in _SKIP_PATHS)
check("ied_agreement_bucket in _SKIP_PATHS",
      "$.findings.ied_agreement_bucket" in _SKIP_PATHS)
check("ied_n_rolandic_benign_bucket in _SKIP_PATHS",
      "$.findings.ied_n_rolandic_benign_bucket" in _SKIP_PATHS)

# Validator must still reject malformed values (defense-in-depth)
mal = json.loads(json.dumps(sub_v2))  # deep copy
mal["findings"]["ied_method"] = "John Smith DOB 1980-05-15"
ok_mal, errs_mal = validate_submission(mal)
check("validator rejects free-text in ied_method",
      not ok_mal, "; ".join(errs_mal))


# ─── Citation ────────────────────────────────────────────────────────────────

section("v0.13.3 — Citation")

from src.clinical.citations import CITATIONS, methods_attribution
check("jing_spikenet in CITATIONS", "jing_spikenet" in CITATIONS)
check("jing_spikenet PMID == 32049322",
      CITATIONS["jing_spikenet"].pubmed_id == "32049322")
check("methods_attribution ied_ml = jing_spikenet",
      methods_attribution().get("ied_ml") == "jing_spikenet")


# ─── Final ───────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  PASS: {n_pass}")
print(f"  FAIL: {n_fail}")
print(f"{'='*60}")
if n_fail > 0:
    print("\nFailed:")
    for name in failed:
        print(f"  - {name}")
    sys.exit(1)
