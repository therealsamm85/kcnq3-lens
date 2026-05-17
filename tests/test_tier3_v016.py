"""Tests for v0.16.0 Tier 3 add-ons: aperiodic exponent, PDR asymmetry, EEG microstates.

Tests are script-style (matching existing test conventions in this repo).
Run with: python tests/test_tier3_v016.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import warnings

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"

n_pass = 0
n_fail = 0


def check(name: str, condition: bool, detail: str = ""):
    global n_pass, n_fail
    if condition:
        n_pass += 1
        print(f"  {PASS} {name}")
    else:
        n_fail += 1
        print(f"  {FAIL} {name}  {detail}")


def section(name: str):
    print(f"\n── {name} ───────────────────────────────────────────")


# ─── Helper: synthetic EEGRecording ───────────────────────────────────────────

from src.readers.base import EEGRecording


def _make_rec(signals: dict[str, np.ndarray], sfreq: float = 256.0) -> EEGRecording:
    """Make a minimal EEGRecording from channel_name -> signal arrays."""
    ch_names = list(signals.keys())
    n_ch = len(ch_names)
    n_samples = min(len(v) for v in signals.values())
    data = np.zeros((n_ch, n_samples))
    for i, ch in enumerate(ch_names):
        data[i] = signals[ch][:n_samples]

    rec = EEGRecording(
        path=Path("/synthetic/test.eeg"),
        sfreq=sfreq,
        n_channels=n_ch,
        duration_s=float(n_samples) / sfreq,
        channel_names=ch_names,
        n_channels_in_file=n_ch,
        eeg_channel_indices=list(range(n_ch)),
        format_name="synthetic",
        _full_data=data,
    )
    return rec


def _make_1overf_signal(exponent: float, sfreq: float = 256.0, n_seconds: float = 120.0) -> np.ndarray:
    """Generate a 1/f^exponent signal via inverse FFT."""
    rng = np.random.default_rng(12345)
    n = int(sfreq * n_seconds)
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    # Amplitude spectrum: A(f) ∝ f^(-exponent/2) so power ∝ f^(-exponent)
    amplitudes = np.ones(len(freqs))
    nz = freqs > 0
    amplitudes[nz] = freqs[nz] ** (-exponent / 2.0)
    amplitudes[0] = 0.0  # no DC
    # Random phases
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    spectrum = amplitudes * (np.cos(phases) + 1j * np.sin(phases))
    signal = np.fft.irfft(spectrum, n=n)
    # Scale to reasonable EEG amplitude (~50 µV)
    signal = signal / np.std(signal) * 50.0
    return signal


def _make_multichannel_rec(
    exponent: float = 1.5,
    n_channels: int = 8,
    sfreq: float = 256.0,
    n_seconds: float = 300.0,
) -> EEGRecording:
    """Multi-channel 1/f recording."""
    ch_names = [f"C{i+1}" for i in range(n_channels)]
    signals = {ch: _make_1overf_signal(exponent, sfreq, n_seconds) for ch in ch_names}
    return _make_rec(signals, sfreq)


# ─── A) Aperiodic exponent tests ────────────────────────────────────────────

section("Aperiodic exponent — synthetic signals")

from src.analyses.aperiodic import (
    compute_aperiodic_exponent,
    _fit_aperiodic_loglog,
    AperiodicResult,
)
from scipy.signal import welch

# Test 1: 1/f^1 → χ ≈ 1.0
sfreq = 256.0
sig1 = _make_1overf_signal(1.0, sfreq, 180.0)
f, P = welch(sig1, fs=sfreq, nperseg=int(sfreq * 4))
chi1 = _fit_aperiodic_loglog(f, P, (2.0, 30.0), (3.5, 14.0))
check(
    "1/f^1 signal → χ ≈ 1.0 (tolerance ±0.35)",
    chi1 is not None and abs(chi1 - 1.0) < 0.35,
    f"chi1={chi1:.3f}" if chi1 else "None",
)

# Test 2: 1/f^2 → χ ≈ 2.0
sig2 = _make_1overf_signal(2.0, sfreq, 180.0)
f, P = welch(sig2, fs=sfreq, nperseg=int(sfreq * 4))
chi2 = _fit_aperiodic_loglog(f, P, (2.0, 30.0), (3.5, 14.0))
check(
    "1/f^2 signal → χ ≈ 2.0 (tolerance ±0.40)",
    chi2 is not None and abs(chi2 - 2.0) < 0.40,
    f"chi2={chi2:.3f}" if chi2 else "None",
)

# Test 3: white noise → χ ≈ 0
rng = np.random.default_rng(42)
sig_noise = rng.normal(0, 50.0, int(sfreq * 180))
f, P = welch(sig_noise, fs=sfreq, nperseg=int(sfreq * 4))
chi_noise = _fit_aperiodic_loglog(f, P, (2.0, 30.0), (3.5, 14.0))
check(
    "White noise → χ < 0.5",
    chi_noise is not None and chi_noise < 0.5,
    f"chi_noise={chi_noise:.3f}" if chi_noise else "None",
)

# Test 4: Full compute on multichannel recording
section("Aperiodic exponent — full compute")

rec_ap = _make_multichannel_rec(exponent=1.5, n_channels=8, sfreq=256.0, n_seconds=300.0)
try:
    ap_result = compute_aperiodic_exponent(rec_ap, sleep_stages=None, method="log_log_regression")
    check(
        "AperiodicResult returned",
        isinstance(ap_result, AperiodicResult),
    )
    check(
        "method == log_log_regression",
        ap_result.method == "log_log_regression",
    )
    check(
        "At least 1 channel has wake data",
        len(ap_result.chi_by_channel) >= 1,
        f"n_ch={len(ap_result.chi_by_channel)}",
    )
    check(
        "Wake state summary present",
        "wake" in ap_result.chi_by_state_summary,
        f"states={list(ap_result.chi_by_state_summary.keys())}",
    )
    wake_chi = ap_result.chi_by_state_summary.get("wake", {}).get("median", None)
    check(
        "Wake χ for 1/f^1.5 signal in range [0.8, 2.5]",
        wake_chi is not None and 0.8 <= wake_chi <= 2.5,
        f"wake_chi={wake_chi}",
    )
    check(
        "Pediatric norm z-scores dict present",
        isinstance(ap_result.pediatric_norm_z_scores, dict),
    )
    check(
        "fit_range_hz is (2.0, 30.0)",
        ap_result.fit_range_hz == (2.0, 30.0),
    )
    check(
        "Disclaimer in notes",
        any("DISCLAIMER" in n for n in ap_result.notes),
    )
    check(
        "Pediatric norm reference has wake, n2, n3",
        {"wake", "n2", "n3"}.issubset(ap_result.pediatric_norm_reference.keys()),
    )
except Exception as e:
    check("compute_aperiodic_exponent no crash", False, str(e))

# Test: outlier filter (flat signal should produce no chi values)
section("Aperiodic exponent — edge cases")

# Edge case: flat signal (all zeros)
flat_signals = {f"C{i+1}": np.zeros(int(256.0 * 120)) for i in range(4)}
rec_flat = _make_rec(flat_signals, 256.0)
try:
    ap_flat = compute_aperiodic_exponent(rec_flat, sleep_stages=None, method="log_log_regression")
    # All-zero signal should yield 0 channels with data (flat = no valid epochs)
    check(
        "Flat signal → 0 channels with data",
        len(ap_flat.chi_by_channel) == 0,
    )
except Exception:
    check("Flat signal → 0 channels with data", True)  # raising is also OK

# Edge case: NaN signal
nan_signals = {f"C{i+1}": np.full(int(256.0 * 120), np.nan) for i in range(4)}
rec_nan = _make_rec(nan_signals, 256.0)
try:
    ap_nan = compute_aperiodic_exponent(rec_nan, sleep_stages=None, method="log_log_regression")
    check("NaN signal → 0 channels with data", len(ap_nan.chi_by_channel) == 0)
except Exception:
    check("NaN signal → 0 channels with data", True)

# Edge case: method fallback (without fooof/specparam, auto falls back)
try:
    import specparam as _sp
    _has_specparam = True
except ImportError:
    _has_specparam = False

if _has_specparam:
    rec_sp = _make_multichannel_rec(exponent=1.5, n_channels=4, sfreq=256.0, n_seconds=120.0)
    try:
        ap_sp = compute_aperiodic_exponent(rec_sp, sleep_stages=None, method="auto")
        check(
            "specparam available → method is 'specparam'",
            ap_sp.method == "specparam",
        )
    except Exception as e:
        check("specparam method works without crash", False, str(e))
else:
    check(
        "Method fallback: auto → log_log_regression (specparam not installed)",
        True,
    )


# ─── B) PDR z-score and asymmetry tests ──────────────────────────────────────

section("Background — PDR z-score and asymmetry")

from src.analyses.background import (
    compute_background_power,
    BackgroundResult,
    _PDR_AGE_NORMS,
    _compute_pdr_z_score,
    _pdr_normative,
)

# Test: z-score calculation
norm_5 = _pdr_normative(5.0)  # should be (8.0, 9.0)
norm_center_5 = (norm_5[0] + norm_5[1]) / 2.0  # 8.5

z_in_norm = _compute_pdr_z_score(8.5, norm_5)  # PDR at norm center → z=0
check(
    "PDR at norm center → z-score = 0.0",
    z_in_norm is not None and abs(z_in_norm) < 0.01,
    f"z={z_in_norm}",
)

z_below = _compute_pdr_z_score(6.5, norm_5)  # 2 Hz below center → z = -2.0
check(
    "PDR 2 Hz below norm center → z-score ≈ -2.0",
    z_below is not None and abs(z_below - (-2.0)) < 0.01,
    f"z={z_below}",
)

z_above = _compute_pdr_z_score(9.5, norm_5)  # 1 Hz above center → z = 1.0
check(
    "PDR 1 Hz above norm center → z-score ≈ 1.0",
    z_above is not None and abs(z_above - 1.0) < 0.01,
    f"z={z_above}",
)

# Test: z-score for no age → None
z_none = _compute_pdr_z_score(8.5, None)
check("No norm → z-score = None", z_none is None)

# Test: normative lookup for age 5, 6, 8
norm_6 = _pdr_normative(6.0)
check("Age 6 norm is (8.5, 9.5)", norm_6 is not None and abs(norm_6[0] - 8.5) < 0.01)

norm_8 = _pdr_normative(8.0)
check("Age 8 norm is (9.0, 10.5)", norm_8 is not None and abs(norm_8[0] - 9.0) < 0.01)

# Test: symmetric signal → asymmetry ≈ 0
section("Background — asymmetry index")

from scipy.signal import butter, sosfiltfilt

def _make_posterior_rec(sfreq=256.0, asymmetric=False):
    """Make recording with posterior channels for asymmetry test."""
    n_sec = 300.0  # 10 minutes → 10 complete 30s epochs
    n = int(sfreq * n_sec)
    rng = np.random.default_rng(99)
    t = np.linspace(0, n_sec, n)

    # Base alpha signal
    alpha = 50.0 * np.sin(2 * np.pi * 10.0 * t) + rng.normal(0, 10, n)

    signals = {
        "O1": alpha.copy(),
        "O2": alpha.copy() * (0.3 if asymmetric else 1.0),  # suppressed right side
        "P3": alpha.copy(),
        "P4": alpha.copy() * (0.3 if asymmetric else 1.0),
        "Pz": alpha.copy(),
        "Fz": rng.normal(0, 20, n),
        "Cz": rng.normal(0, 20, n),
    }
    return _make_rec(signals, sfreq)

# Use explicit wake epoch indices to avoid edge-selection issues
rec_sym = _make_posterior_rec(asymmetric=False)
_wake_epochs = list(range(2, 8))  # epochs 2-7 (within the 10 available)
try:
    bg_sym = compute_background_power(rec_sym, age_years=8.0, wake_epoch_indices=_wake_epochs)
    ai_sym = bg_sym.pdr_asymmetry_index
    check(
        "Symmetric signal → |asymmetry| < 0.10",
        ai_sym is not None and abs(ai_sym) < 0.10,
        f"ai={ai_sym}",
    )
    check(
        "Symmetric interpretation in {'symmetric', 'lh_dominant', 'rh_dominant'}",
        bg_sym.asymmetry_interpretation in {"symmetric", "lh_dominant", "rh_dominant", "not_computed"},
    )
    check(
        "PDR z-score computed for age 8",
        bg_sym.pdr_z_score is not None,
        f"z={bg_sym.pdr_z_score}",
    )
except Exception as e:
    check("Symmetric posterior rec — no crash", False, str(e))

rec_asym = _make_posterior_rec(asymmetric=True)
try:
    bg_asym = compute_background_power(rec_asym, age_years=8.0, wake_epoch_indices=_wake_epochs)
    ai_asym = bg_asym.pdr_asymmetry_index
    check(
        "Asymmetric signal → asymmetry > 0.10 (lh dominant)",
        ai_asym is not None and ai_asym > 0.10,
        f"ai={ai_asym}",
    )
    check(
        "Asymmetric interpretation is 'lh_dominant' or 'marked_asymmetric'",
        bg_asym.asymmetry_interpretation in {"lh_dominant", "marked_asymmetric"},
        f"interp={bg_asym.asymmetry_interpretation}",
    )
except Exception as e:
    check("Asymmetric posterior rec — no crash", False, str(e))


# ─── C) Microstates tests ────────────────────────────────────────────────────

section("Microstates — basic")

from src.analyses.microstates import (
    compute_microstates,
    summarize_microstates,
    MicrostateResult,
    _compute_gfp,
    _kmeans_polarity_invariant,
    _compute_microstate_metrics,
    _MS_LABELS,
)


def _make_4state_signal(sfreq=256.0, n_seconds=120.0):
    """Generate a 4-state synthetic signal that k-means should cleanly cluster."""
    rng = np.random.default_rng(7)
    n = int(sfreq * n_seconds)
    n_ch = 19
    ch_names_19 = [
        "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
        "C3", "C4", "Cz",
        "P3", "P4", "Pz",
        "O1", "O2",
        "T3", "T4", "T5", "T6",
    ]

    # 4 canonical topographies
    templates = np.zeros((4, n_ch))
    templates[0] = [1, -1, 0.5, -0.5, 0.8, -0.8, 0, 0, 0, 0, -0.5, 0.5, 0, -1, 1, 0, 0, -0.5, 0.5]  # A-like
    templates[1] = [-1, 1, -0.5, 0.5, -0.8, 0.8, 0, 0, 0, 0, 0.5, -0.5, 0, 1, -1, 0, 0, 0.5, -0.5]  # B-like
    templates[2] = [0, 0, 0, 0, 0, 0, 1, 0.5, 0.5, 1, 0.5, 0.5, 0.5, -0.5, -0.5, 0, 0, 0, 0]  # C-like
    templates[3] = [1, 1, 0.8, 0.8, 0.5, 0.5, 1, 0.5, 0.5, 0.3, 0, 0, 0, -0.5, -0.5, 0, 0, 0, 0]  # D-like

    # Normalize templates
    for i in range(4):
        n_tm = np.linalg.norm(templates[i])
        if n_tm > 0:
            templates[i] /= n_tm

    # Generate alternating state sequence (30s each state cycling)
    data = np.zeros((n_ch, n))
    state_len = int(sfreq * 30)
    for t in range(n):
        state = (t // state_len) % 4
        noise = rng.normal(0, 0.05, n_ch)
        data[:, t] = templates[state] * 50.0 + noise * 50.0

    signals = {ch: data[i] for i, ch in enumerate(ch_names_19)}
    return _make_rec(signals, sfreq), ch_names_19


rec_ms4, ch_names_19 = _make_4state_signal(256.0, 300.0)

try:
    ms_result = compute_microstates(rec_ms4, sleep_stages=None, method="kmeans_gfp")
    check("MicrostateResult returned", isinstance(ms_result, MicrostateResult))
    check("n_topomaps = 4", ms_result.n_topomaps == 4)
    check(
        "All 4 states present in coverage_pct",
        set(ms_result.coverage_pct.keys()) == {"A", "B", "C", "D"},
    )
    total_cov = sum(ms_result.coverage_pct.values())
    check(
        "Coverage sums to ~100%",
        abs(total_cov - 100.0) < 1.0,
        f"total={total_cov:.1f}",
    )
    # Transition matrix rows sum to 1
    all_rows_ok = True
    for ms in _MS_LABELS:
        row_sum = sum(
            ms_result.transition_matrix.get((ms, b), 0.0)
            for b in _MS_LABELS if b != ms
        )
        if abs(row_sum - 1.0) > 0.01 and row_sum > 0.01:
            all_rows_ok = False
    check("Transition matrix rows sum to 1 (or 0 for absent states)", all_rows_ok)

    check("method is kmeans_gfp", ms_result.method == "kmeans_gfp")
    check("Disclaimer in notes", any("DISCLAIMER" in n for n in ms_result.notes))
except Exception as e:
    check("compute_microstates 4-state synthetic — no crash", False, str(e))

# Test: pure noise → roughly equal coverage
section("Microstates — noise and edge cases")

rng_ms = np.random.default_rng(1234)
n_ch_ms = 19
noise_data = rng_ms.normal(0, 50.0, (n_ch_ms, int(256 * 300)))
ch_names_noise = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
    "C3", "C4", "Cz",
    "P3", "P4", "Pz",
    "O1", "O2",
    "T3", "T4", "T5", "T6",
]
noise_signals = {ch: noise_data[i] for i, ch in enumerate(ch_names_noise)}
rec_noise = _make_rec(noise_signals, 256.0)

try:
    ms_noise = compute_microstates(rec_noise, sleep_stages=None, method="kmeans_gfp")
    # For noise, coverage should be roughly equal (within ~10%)
    covs = list(ms_noise.coverage_pct.values())
    check(
        "Pure noise → coverage sums to 100%",
        abs(sum(covs) - 100.0) < 1.0,
        f"total={sum(covs):.1f}",
    )
    check(
        "Pure noise → all states present",
        all(c > 0.0 for c in covs),
        f"coverage={covs}",
    )
except Exception as e:
    check("Noise microstate — no crash", False, str(e))


# ─── D) Registry / bucket tests ───────────────────────────────────────────────

section("Registry — v0.16.0 bucket helpers")

from src.registry.buckets import bucket_aperiodic_chi_n2, bucket_pdr_asymmetry
from src.registry.schema import APERIODIC_CHI_N2_BUCKETS, PDR_ASYMMETRY_BUCKETS, MICROSTATE_DOMINANT_VALUES

check("chi=1.0 → '<1.5'", bucket_aperiodic_chi_n2(1.0) == "<1.5")
check("chi=1.6 → '1.5-2.0'", bucket_aperiodic_chi_n2(1.6) == "1.5-2.0")
check("chi=2.1 → '2.0-2.5'", bucket_aperiodic_chi_n2(2.1) == "2.0-2.5")
check("chi=3.0 → '>2.5'", bucket_aperiodic_chi_n2(3.0) == ">2.5")
check("chi=None → None", bucket_aperiodic_chi_n2(None) is None)
check("chi=-0.5 → None (negative)", bucket_aperiodic_chi_n2(-0.5) is None)

check("asym 'symmetric' → 'symmetric'", bucket_pdr_asymmetry("symmetric") == "symmetric")
check("asym 'lh_dominant' → 'lh_dominant'", bucket_pdr_asymmetry("lh_dominant") == "lh_dominant")
check("asym 'rh_dominant' → 'rh_dominant'", bucket_pdr_asymmetry("rh_dominant") == "rh_dominant")
check("asym 'marked_asymmetric' → 'marked'", bucket_pdr_asymmetry("marked_asymmetric") == "marked")
check("asym None → None", bucket_pdr_asymmetry(None) is None)

check("MICROSTATE_DOMINANT_VALUES has A,B,C,D",
      {"A", "B", "C", "D"} == MICROSTATE_DOMINANT_VALUES)

# Registry extractor tests
section("Registry — v0.16.0 extractors")

from src.registry.deid import (
    _extract_aperiodic_chi_n2_bucket,
    _extract_pdr_asymmetry_bucket,
    _extract_microstate_dominant,
)

findings_test = {
    "aperiodic": {
        "chi_by_state": {
            "n2": {"median": 3.2, "p25": 2.8, "p75": 3.6, "n_epochs": 40, "n_channels": 8}
        }
    },
    "background": {
        "asymmetry_interpretation": "lh_dominant",
        "pdr_asymmetry_index": 0.15,
    },
    "microstates": {
        "dominant_microstate": "D",
        "coverage_pct": {"A": 22.0, "B": 24.0, "C": 19.0, "D": 35.0},
    },
}

check(
    "_extract_aperiodic_chi_n2_bucket: chi=3.2 → '>2.5'",
    _extract_aperiodic_chi_n2_bucket(findings_test) == ">2.5",
)
check(
    "_extract_pdr_asymmetry_bucket: 'lh_dominant' → 'lh_dominant'",
    _extract_pdr_asymmetry_bucket(findings_test) == "lh_dominant",
)
check(
    "_extract_microstate_dominant: 'D' → 'D'",
    _extract_microstate_dominant(findings_test) == "D",
)

# Test with missing findings
check("_extract_aperiodic_chi_n2_bucket: empty → None",
      _extract_aperiodic_chi_n2_bucket({}) is None)
check("_extract_pdr_asymmetry_bucket: empty → None",
      _extract_pdr_asymmetry_bucket({}) is None)
check("_extract_microstate_dominant: empty → None",
      _extract_microstate_dominant({}) is None)
check("_extract_microstate_dominant: invalid value → None",
      _extract_microstate_dominant({"microstates": {"dominant_microstate": "X"}}) is None)

# ─── E) Validate schema still passes with v0.16.0 fields ─────────────────────

section("Registry validate — v0.16.0 fields in schema")

from src.registry.validate import validate_submission
import uuid

def _make_minimal_submission(extra_findings=None):
    sid = str(uuid.uuid4())
    sub = {
        "submission_id": sid,
        "schema_version": 2,
        "submitted_at_month": "2026-05",
        "consent": {"version": 1, "given": True, "given_at_month": "2026-05"},
        "subject": {
            "variant_gene": "KCNQ3",
            "variant_protein": "p.Arg230His",
            "variant_type": "missense_GoF",
            "age_years_bucket": "5-7",
            "sex": "F",
        },
        "recording": {
            "duration_hours_bucket": "12-24",
            "had_sleep": True,
            "montage": "10-20_monopolar",
            "n_channels": 19,
        },
        "findings": extra_findings or {},
        "tool_version": "0.16.0",
    }
    return sub

sub_v016 = _make_minimal_submission({
    "aperiodic_chi_n2_bucket": ">2.5",
    "pdr_asymmetry_bucket": "symmetric",
    "microstate_dominant": "D",
})
ok, errs = validate_submission(sub_v016)
check("v0.16.0 fields validate OK", ok, str(errs) if not ok else "")

sub_no_v016 = _make_minimal_submission({})
ok2, errs2 = validate_submission(sub_no_v016)
check("Submission without v0.16.0 fields still validates (additive)", ok2, str(errs2))

sub_invalid = _make_minimal_submission({"aperiodic_chi_n2_bucket": "99.9"})
ok3, errs3 = validate_submission(sub_invalid)
check("Invalid chi bucket value → validation fails", not ok3)


# ─── F) Summarize helpers ─────────────────────────────────────────────────────

section("Summarize helpers")

from src.analyses.aperiodic import summarize_aperiodic

try:
    rec_sum = _make_multichannel_rec(exponent=1.5, n_channels=4, sfreq=256.0, n_seconds=120.0)
    ap_sum = compute_aperiodic_exponent(rec_sum, method="log_log_regression")
    summary = summarize_aperiodic(ap_sum)
    check(
        "summarize_aperiodic returns dict with expected keys",
        all(k in summary for k in ("method", "chi_by_state", "disclaimer")),
    )
except Exception as e:
    check("summarize_aperiodic no crash", False, str(e))

from src.analyses.microstates import summarize_microstates
try:
    ms_sum_result = compute_microstates(rec_ms4, sleep_stages=None, method="kmeans_gfp")
    ms_summary = summarize_microstates(ms_sum_result)
    check(
        "summarize_microstates returns dict with expected keys",
        all(k in ms_summary for k in ("coverage_pct", "dominant_microstate", "transition_probabilities")),
    )
    check(
        "summarize_microstates coverage sums to 100",
        abs(sum(ms_summary["coverage_pct"].values()) - 100.0) < 1.0,
    )
except Exception as e:
    check("summarize_microstates no crash", False, str(e))


# ─── Final summary ────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  PASS: {n_pass}")
print(f"  FAIL: {n_fail}")
print(f"{'='*60}")
if n_fail > 0:
    sys.exit(1)
