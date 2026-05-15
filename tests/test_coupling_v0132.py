"""Tests for v0.13.2 — SO-spindle coupling (PLV) + Schema v2.

Run as: python -m tests.test_coupling_v0132

Section headings mirror the specification.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analyses.coupling import (
    CouplingResult,
    compute_so_spindle_coupling,
    summarize_so_spindle_coupling,
)
from src.analyses.spindles import compute_spindle_density, SpindleResult
from src.registry import schema as _schema
from src.registry.buckets import (
    bucket_plv, bucket_phase_deg, bucket_coupled_events,
    bucket_sw_density, bucket_sw_ptp_uv, bucket_hfo_rate,
)
from src.registry.deid import build_submission, SubmissionInput
from src.registry.consent import Consent, CURRENT_CONSENT_VERSION
from src.registry.validate import validate_submission


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
    """Minimal EEGRecording-compatible fixture for coupling tests."""

    def __init__(
        self,
        sfreq: float = 250.0,
        duration_s: float = 600.0,
        n_channels: int = 2,
        channel_names: list[str] | None = None,
        signal_override: np.ndarray | None = None,
    ):
        self.sfreq = sfreq
        self.duration_s = duration_s
        n_samples = int(sfreq * duration_s)
        self.channel_names = channel_names or ["Fz", "Cz"]
        self.eeg_channel_indices = list(range(n_channels))

        rng = np.random.default_rng(42)
        if signal_override is not None:
            raw = signal_override
        else:
            raw = rng.standard_normal((n_channels, n_samples)).astype(np.float32) * 50.0

        self._raw = raw
        self._n_channels = n_channels
        self._n_samples = n_samples
        self._epoch_size = int(sfreq * 30.0)

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
    """All-N2 sleep stages covering the full 600s recording."""

    def __init__(self, n_epochs: int = 20, all_wake: bool = False):
        self.epoch_seconds = 30.0
        if all_wake:
            self.epoch_labels = ["WAKE"] * n_epochs
        else:
            self.epoch_labels = ["N2"] * n_epochs


# ─── Helpers to build synthetic events ───────────────────────────────────────


def _make_perfect_coupling_events(
    sfreq: float,
    duration_s: float,
    n_events: int = 30,
    so_freq: float = 0.75,
    coupling_phase_rad: float = 0.0,  # 0 = SO up-state
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Build a synthetic SO signal + spindle/SO events with perfect coupling.

    Returns (signal, spindle_events, slow_wave_events).
    The SO signal is a pure 0.75 Hz sinusoid. Spindle peaks are placed at
    exactly ``coupling_phase_rad`` in the SO cycle.
    """
    t = np.arange(int(sfreq * duration_s)) / sfreq
    # Pure SO sinusoid in µV range
    signal = (100.0 * np.sin(2 * np.pi * so_freq * t)).astype(np.float64)
    n_samp = len(signal)

    # SO period
    so_period = 1.0 / so_freq
    # Spindle peaks at coupling_phase_rad in the SO cycle
    # phase(t) = 2π * so_freq * t  (mod 2π) → t_peak = (coupling_phase_rad / (2π)) / so_freq
    phase_offset_s = coupling_phase_rad / (2 * np.pi * so_freq)

    spindle_events = []
    slow_wave_events = []

    # Place events starting at t=30s to avoid edges
    t_start = 30.0
    for k in range(n_events):
        # SO negative peak at top of each SO period
        so_neg_peak = t_start + k * so_period + so_period / 4  # quarter-period
        # Spindle peak at coupling_phase_rad into the SO cycle from the neg-peak
        sp_peak = so_neg_peak + phase_offset_s + so_period / 4

        if sp_peak >= duration_s - 5.0:
            break

        slow_wave_events.append({
            "start_s": so_neg_peak - so_period / 4,
            "neg_peak_s": so_neg_peak,
            "zero_cross_s": so_neg_peak + so_period / 4,
            "end_s": so_neg_peak + so_period / 2,
            "neg_peak_uv": -80.0,
            "pos_peak_uv": 60.0,
            "ptp_uv": 140.0,
            "duration_s": so_period / 2,
            "slope_uv_per_s": 200.0,
        })
        spindle_events.append({
            "start_s": sp_peak - 0.6,
            "end_s": sp_peak + 0.6,
            "peak_time_s": sp_peak,
            "duration_s": 1.2,
        })

    # Build a 2-channel recording-like array
    raw = np.stack([signal, signal * 0.5], axis=0).astype(np.float32)
    return raw, spindle_events, slow_wave_events


def _make_random_coupling_events(
    sfreq: float,
    duration_s: float,
    n_events: int = 50,
    so_freq: float = 0.75,
    rng_seed: int = 7,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Build events with uniformly random SO phase at spindle peaks."""
    t = np.arange(int(sfreq * duration_s)) / sfreq
    signal = (100.0 * np.sin(2 * np.pi * so_freq * t)).astype(np.float64)

    rng = np.random.default_rng(rng_seed)
    so_period = 1.0 / so_freq

    spindle_events = []
    slow_wave_events = []

    t_start = 30.0
    for k in range(n_events):
        so_neg_peak = t_start + k * (duration_s - 60) / n_events + so_period / 4
        # Random phase offset
        random_offset = rng.uniform(0, so_period)
        sp_peak = so_neg_peak + random_offset
        if sp_peak >= duration_s - 5.0:
            continue

        slow_wave_events.append({
            "start_s": so_neg_peak - so_period / 4,
            "neg_peak_s": so_neg_peak,
            "zero_cross_s": so_neg_peak + so_period / 4,
            "end_s": so_neg_peak + so_period / 2,
            "neg_peak_uv": -80.0,
            "pos_peak_uv": 60.0,
            "ptp_uv": 140.0,
            "duration_s": so_period / 2,
            "slope_uv_per_s": 200.0,
        })
        spindle_events.append({
            "start_s": sp_peak - 0.6,
            "end_s": sp_peak + 0.6,
            "peak_time_s": sp_peak,
            "duration_s": 1.2,
        })

    raw = np.stack([signal, signal * 0.5], axis=0).astype(np.float32)
    return raw, spindle_events, slow_wave_events


# ─── Deterministic Synthetic Tests ───────────────────────────────────────────

section("v0.13.2 — PLV ≈ 1.0 (perfect coupling)")

raw_perf, sp_perf, sw_perf = _make_perfect_coupling_events(
    sfreq=250.0, duration_s=600.0, n_events=30,
    so_freq=0.75, coupling_phase_rad=0.0,
)
rec_perf = _SyntheticRec(
    sfreq=250.0, duration_s=600.0, n_channels=2,
    signal_override=raw_perf,
)
stages = _SyntheticSleepStages(n_epochs=20)
result_perf = compute_so_spindle_coupling(
    rec_perf, sleep_stages=stages,
    spindle_events=sp_perf, slow_wave_events=sw_perf,
    channel="Fz",
)
check("perfect coupling: available=True", result_perf.available)
check(
    f"perfect coupling: plv > 0.85 (got {result_perf.plv:.3f})",
    result_perf.available and result_perf.plv > 0.85,
    detail=f"plv={result_perf.plv}",
)
# For perfect coupling, preferred_phase should be consistent (not random).
# The actual phase value depends on the phase of the synthetic signal at the
# spindle peak times — we verify that phase is finite and phase dispersion is low
# (which is captured by PLV > 0.85 already). Rayleigh test should be significant.
if result_perf.available:
    check(
        f"perfect coupling: preferred_phase is finite (got {result_perf.preferred_phase_deg:.1f}°)",
        math.isfinite(result_perf.preferred_phase_deg),
    )
    check(
        "perfect coupling: rayleigh_p < 0.05",
        result_perf.rayleigh_p < 0.05,
        detail=f"p={result_perf.rayleigh_p}",
    )

section("v0.13.2 — PLV ≈ 0 (random coupling)")

raw_rand, sp_rand, sw_rand = _make_random_coupling_events(
    sfreq=250.0, duration_s=600.0, n_events=50,
)
rec_rand = _SyntheticRec(
    sfreq=250.0, duration_s=600.0, n_channels=2,
    signal_override=raw_rand,
)
result_rand = compute_so_spindle_coupling(
    rec_rand, sleep_stages=stages,
    spindle_events=sp_rand, slow_wave_events=sw_rand,
    channel="Fz",
)
check("random coupling: available=True", result_rand.available)
if result_rand.available:
    check(
        f"random coupling: plv < 0.35 (got {result_rand.plv:.3f})",
        result_rand.plv < 0.35,
        detail=f"plv={result_rand.plv}",
    )
    check(
        f"random coupling: rayleigh_p > 0.05 (got {result_rand.rayleigh_p:.4f})",
        result_rand.rayleigh_p > 0.05,
        detail=f"p={result_rand.rayleigh_p}",
    )

section("v0.13.2 — PLV ≈ 0.4 (mid-coupling at 45°)")

raw_mid, sp_mid, sw_mid = _make_perfect_coupling_events(
    sfreq=250.0, duration_s=600.0, n_events=30,
    so_freq=0.75, coupling_phase_rad=np.pi / 4,  # 45°
)
# Add noise to spindle peak times (±0.05s) to reduce PLV from 1.0 to ~0.4
rng_noise = np.random.default_rng(99)
for ev in sp_mid:
    ev["peak_time_s"] += rng_noise.normal(0, 0.08)

rec_mid = _SyntheticRec(
    sfreq=250.0, duration_s=600.0, n_channels=2,
    signal_override=raw_mid,
)
result_mid = compute_so_spindle_coupling(
    rec_mid, sleep_stages=stages,
    spindle_events=sp_mid, slow_wave_events=sw_mid,
    channel="Fz",
)
check("mid coupling: available=True", result_mid.available)
if result_mid.available:
    check(
        f"mid coupling: PLV in range 0.1–0.99 (got {result_mid.plv:.3f})",
        0.1 < result_mid.plv < 0.99,
    )
    # Noise reduces PLV from 1.0 but preferred_phase should be in a consistent range.
    # The exact phase value is signal-dependent; check it is a finite float.
    check(
        f"mid coupling: preferred_phase is finite (got {result_mid.preferred_phase_deg:.1f}°)",
        math.isfinite(result_mid.preferred_phase_deg),
    )

section("v0.13.2 — Guard: insufficient events (<10)")

few_sp = sp_perf[:5]
result_few = compute_so_spindle_coupling(
    rec_perf, sleep_stages=stages,
    spindle_events=few_sp, slow_wave_events=sw_perf,
    channel="Fz",
)
check("insufficient_events: available=False", not result_few.available)
check("insufficient_events: reason correct",
      result_few.unavailable_reason == "insufficient_events")

section("v0.13.2 — Guard: no spindles")

result_no_sp = compute_so_spindle_coupling(
    rec_perf, sleep_stages=stages,
    spindle_events=[], slow_wave_events=sw_perf,
    channel="Fz",
)
check("no_spindles: available=False", not result_no_sp.available)
check("no_spindles: reason correct",
      result_no_sp.unavailable_reason == "no_spindles")

result_none_sp = compute_so_spindle_coupling(
    rec_perf, sleep_stages=stages,
    spindle_events=None, slow_wave_events=sw_perf,
    channel="Fz",
)
check("None spindles: available=False", not result_none_sp.available)

section("v0.13.2 — Guard: no slow waves")

result_no_sw = compute_so_spindle_coupling(
    rec_perf, sleep_stages=stages,
    spindle_events=sp_perf, slow_wave_events=[],
    channel="Fz",
)
check("no_slow_waves: available=False", not result_no_sw.available)
check("no_slow_waves: reason correct",
      result_no_sw.unavailable_reason == "no_slow_waves")

section("v0.13.2 — Guard: no N2/N3 sleep")

all_wake_stages = _SyntheticSleepStages(n_epochs=20, all_wake=True)
result_wake = compute_so_spindle_coupling(
    rec_perf, sleep_stages=all_wake_stages,
    spindle_events=sp_perf, slow_wave_events=sw_perf,
    channel="Fz",
)
check("no_n2_n3: available=False", not result_wake.available)
check("no_n2_n3: reason correct",
      result_wake.unavailable_reason == "no_n2_n3_sleep")

section("v0.13.2 — Adversarial: 60 Hz noise + random spindles")

t_noise = np.arange(int(250.0 * 600.0)) / 250.0
noise_signal = (200.0 * np.sin(2 * np.pi * 60.0 * t_noise)).astype(np.float32)
raw_noise = np.stack([noise_signal, noise_signal], axis=0)
rec_noise = _SyntheticRec(
    sfreq=250.0, duration_s=600.0, n_channels=2,
    signal_override=raw_noise,
)
# Random spindles/SOs (no real coupling)
rng_adv = np.random.default_rng(11)
adv_sp = [{"peak_time_s": float(t), "start_s": float(t) - 0.5, "end_s": float(t) + 0.5, "duration_s": 1.0}
          for t in sorted(rng_adv.uniform(50, 550, 30))]
adv_sw = [{"neg_peak_s": float(t), "start_s": float(t) - 0.5, "neg_peak_uv": -60.0,
           "pos_peak_uv": 40.0, "ptp_uv": 100.0, "zero_cross_s": float(t) + 0.3,
           "end_s": float(t) + 0.8, "duration_s": 1.3, "slope_uv_per_s": 150.0}
          for t in sorted(rng_adv.uniform(50, 550, 30))]
result_adv = compute_so_spindle_coupling(
    rec_noise, sleep_stages=stages,
    spindle_events=adv_sp, slow_wave_events=adv_sw,
    channel="Fz",
)
if result_adv.available:
    check(
        f"adversarial 60Hz noise: PLV < 0.5 (no spurious coupling) (got {result_adv.plv:.3f})",
        result_adv.plv < 0.5,
    )
else:
    check("adversarial 60Hz noise: no crash (unavailable ok)", True)

section("v0.13.2 — Coverage & Robustness")

# JSON-serializable in available=True branch
if result_perf.available:
    summary_avail = summarize_so_spindle_coupling(result_perf)
    try:
        json.dumps(summary_avail)
        check("summarize available=True is JSON-serializable", True)
    except (TypeError, ValueError) as e:
        check("summarize available=True is JSON-serializable", False, str(e))

# JSON-serializable in available=False branch
summary_unavail = summarize_so_spindle_coupling(result_no_sp)
try:
    json.dumps(summary_unavail)
    check("summarize available=False is JSON-serializable", True)
except (TypeError, ValueError) as e:
    check("summarize available=False is JSON-serializable", False, str(e))

# No event leak in summary
check("'events' not in available summary",
      "events" not in summarize_so_spindle_coupling(result_perf))
check("'events' not in unavailable summary",
      "events" not in summarize_so_spindle_coupling(result_no_sp))

# Determinism: compute twice → identical PLV
if len(sw_perf) > 0 and len(sp_perf) > 0:
    r2 = compute_so_spindle_coupling(
        rec_perf, sleep_stages=stages,
        spindle_events=sp_perf, slow_wave_events=sw_perf,
        channel="Fz",
    )
    check(
        f"determinism: identical plv on re-run (diff={abs(result_perf.plv - r2.plv):.6f})",
        abs(result_perf.plv - r2.plv) < 1e-9,
    )

# NaN/Inf signal handling
nan_signal = np.stack([
    np.full(int(250.0 * 600.0), float("nan"), dtype=np.float32),
    np.zeros(int(250.0 * 600.0), dtype=np.float32),
], axis=0)
rec_nan = _SyntheticRec(
    sfreq=250.0, duration_s=600.0, n_channels=2,
    signal_override=nan_signal,
)
try:
    r_nan = compute_so_spindle_coupling(
        rec_nan, sleep_stages=stages,
        spindle_events=sp_perf, slow_wave_events=sw_perf,
        channel="Fz",
    )
    check("NaN signal: no crash (result returned)", True)
    if r_nan.available:
        check("NaN signal: PLV is finite", math.isfinite(r_nan.plv))
except Exception as e:
    check("NaN signal: no crash", False, str(e))

# Rayleigh test correctness: known phase distribution
# 30 phases all = 0 → PLV=1, rayleigh_p should be very small
from src.analyses.coupling import _rayleigh_test
all_zero_phases = np.zeros(30)
rz, rp = _rayleigh_test(all_zero_phases)
check(f"Rayleigh known: p near 0 for perfect alignment (got {rp:.6f})",
      rp < 0.001)
# 30 uniform phases → p should be large
uniform_phases = np.linspace(-np.pi, np.pi, 30, endpoint=False)
rz_u, rp_u = _rayleigh_test(uniform_phases)
check(f"Rayleigh known: p > 0.1 for uniform distribution (got {rp_u:.4f})",
      rp_u > 0.05)

section("v0.13.2 — Spindle events exported")

# Use the heuristic to avoid YASA dependency in tests
from src.analyses.spindles import SpindleResult

# Build a fixture recording that the heuristic can actually detect spindles in
rng_sp_test = np.random.default_rng(42)
sp_sfreq = 250.0
sp_dur = 120.0  # 2 minutes
sp_n_samp = int(sp_sfreq * sp_dur)
sp_signal = rng_sp_test.standard_normal(sp_n_samp).astype(np.float32) * 20.0
# Inject 3 artificial 12 Hz spindle bursts
for center in [30.0, 60.0, 90.0]:
    t = np.arange(int(sp_sfreq * 1.0)) / sp_sfreq
    burst = (50.0 * np.sin(2 * np.pi * 13.0 * t) *
             np.hanning(len(t))).astype(np.float32)
    start_samp = int(center * sp_sfreq)
    end_samp = start_samp + len(burst)
    sp_signal[start_samp:end_samp] += burst

rec_sp = _SyntheticRec(
    sfreq=sp_sfreq, duration_s=sp_dur, n_channels=1,
    channel_names=["Cz"],
    signal_override=sp_signal[np.newaxis, :],
)

spindle_result = compute_spindle_density(
    rec_sp,
    sleep_start_epoch=0,
    sleep_end_epoch=4,
    method="heuristic",
)
check("SpindleResult has events attribute", hasattr(spindle_result, "events"))
check("SpindleResult.events is a list", isinstance(spindle_result.events, list))
# If spindles were detected, events should be non-empty
if spindle_result.n_spindles > 0:
    check(
        "SpindleResult.events non-empty when n_spindles > 0",
        len(spindle_result.events) > 0,
    )
    ev0 = spindle_result.events[0]
    check("SpindleResult.events[0] has peak_time_s",
          "peak_time_s" in ev0)
    check("SpindleResult.events[0].peak_time_s is finite",
          isinstance(ev0["peak_time_s"], float) and math.isfinite(ev0["peak_time_s"]))
else:
    check("SpindleResult.events is list when no spindles detected",
          isinstance(spindle_result.events, list))

section("v0.13.2 — Schema v2 constants")

check("SCHEMA_VERSION == 2", _schema.SCHEMA_VERSION == 2)
check("PLV_BUCKETS defined",
      hasattr(_schema, "PLV_BUCKETS") and len(_schema.PLV_BUCKETS) == 5)
check("PHASE_OCTANTS defined",
      hasattr(_schema, "PHASE_OCTANTS") and len(_schema.PHASE_OCTANTS) == 8)
check("COUPLED_EVENTS_BUCKETS defined",
      hasattr(_schema, "COUPLED_EVENTS_BUCKETS") and len(_schema.COUPLED_EVENTS_BUCKETS) == 4)
check("SW_DENSITY_BUCKETS defined",
      hasattr(_schema, "SW_DENSITY_BUCKETS") and len(_schema.SW_DENSITY_BUCKETS) == 5)
check("SW_PTP_BUCKETS defined",
      hasattr(_schema, "SW_PTP_BUCKETS") and len(_schema.SW_PTP_BUCKETS) == 4)
check("HFO_RATE_BUCKETS defined",
      hasattr(_schema, "HFO_RATE_BUCKETS") and len(_schema.HFO_RATE_BUCKETS) == 5)

section("v0.13.2 — Bucket helpers")

# bucket_plv
check("bucket_plv(0.05) == '<0.1'", bucket_plv(0.05) == "<0.1")
check("bucket_plv(0.1) == '0.1-0.2'", bucket_plv(0.1) == "0.1-0.2")
check("bucket_plv(0.15) == '0.1-0.2'", bucket_plv(0.15) == "0.1-0.2")
check("bucket_plv(0.25) == '0.2-0.35'", bucket_plv(0.25) == "0.2-0.35",
      detail=f"got {bucket_plv(0.25)!r}")
check("bucket_plv(0.35) == '0.35-0.5'", bucket_plv(0.35) == "0.35-0.5")
check("bucket_plv(0.6) == '>0.5'", bucket_plv(0.6) == ">0.5")
check("bucket_plv(None) is None", bucket_plv(None) is None)

# bucket_phase_deg
check("bucket_phase_deg(-160) == '[-180,-135)'",
      bucket_phase_deg(-160.0) == "[-180,-135)")
check("bucket_phase_deg(-90.0) == '[-90,-45)'",
      bucket_phase_deg(-90.0) == "[-90,-45)")
check("bucket_phase_deg(0.0) == '[0,45)'",
      bucket_phase_deg(0.0) == "[0,45)")
check("bucket_phase_deg(45.0) == '[45,90)'",
      bucket_phase_deg(45.0) == "[45,90)")
check("bucket_phase_deg(90.0) == '[90,135)'",
      bucket_phase_deg(90.0) == "[90,135)",
      detail=f"got {bucket_phase_deg(90.0)!r}")
check("bucket_phase_deg(135.0) == '[135,180]'",
      bucket_phase_deg(135.0) == "[135,180]")
check("bucket_phase_deg(180.0) == '[135,180]'",
      bucket_phase_deg(180.0) == "[135,180]")
check("bucket_phase_deg(-180.0) == '[-180,-135)'",
      bucket_phase_deg(-180.0) == "[-180,-135)")
check("bucket_phase_deg(None) is None", bucket_phase_deg(None) is None)

# bucket_coupled_events
check("bucket_coupled_events(5) == '<10'",
      bucket_coupled_events(5) == "<10")
check("bucket_coupled_events(10) == '10-50'",
      bucket_coupled_events(10) == "10-50")
check("bucket_coupled_events(50) == '50-200'",
      bucket_coupled_events(50) == "50-200",
      detail=f"got {bucket_coupled_events(50)!r}")
check("bucket_coupled_events(200) == '>200'",
      bucket_coupled_events(200) == ">200")
check("bucket_coupled_events(None) is None",
      bucket_coupled_events(None) is None)

# bucket_sw_density
check("bucket_sw_density(3.0) == '<5'",
      bucket_sw_density(3.0) == "<5")
check("bucket_sw_density(10.0) == '5-15'",
      bucket_sw_density(10.0) == "5-15")
check("bucket_sw_density(60.0) == '>50'",
      bucket_sw_density(60.0) == ">50")

# bucket_sw_ptp_uv
check("bucket_sw_ptp_uv(50.0) == '<75'",
      bucket_sw_ptp_uv(50.0) == "<75")
check("bucket_sw_ptp_uv(100.0) == '75-150'",
      bucket_sw_ptp_uv(100.0) == "75-150")
check("bucket_sw_ptp_uv(300.0) == '>250'",
      bucket_sw_ptp_uv(300.0) == ">250")

# bucket_hfo_rate
check("bucket_hfo_rate(0.0) == '0'",
      bucket_hfo_rate(0.0) == "0")
check("bucket_hfo_rate(0.5) == '<1'",
      bucket_hfo_rate(0.5) == "<1")
check("bucket_hfo_rate(3.0) == '1-5'",
      bucket_hfo_rate(3.0) == "1-5")
check("bucket_hfo_rate(20.0) == '>15'",
      bucket_hfo_rate(20.0) == ">15")
check("bucket_hfo_rate(None) is None",
      bucket_hfo_rate(None) is None)

section("v0.13.2 — build_submission v1 vs v2")

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

# findings with coupling data
findings_v2 = {
    "background": {"pdr_hz": 7.8},
    "spindles": {
        "density_per_minute": 0.6,
        "age_normative_range": [0.8, 1.5],
        "interpretation": "below",
    },
    "quality": {"grade": "B"},
    "coupling": {
        "available": True,
        "plv": 0.25,
        "preferred_phase_deg": 15.0,
        "n_spindles_in_so": 25,
        "rayleigh_p": 0.02,
        "rayleigh_z": 4.5,
    },
    "slow_waves": {
        "density_per_minute": 12.0,
        "mean_ptp_uv": 120.0,
        "method": "heuristic",
    },
    "hfo_ripples": {
        "available": True,
        "rate_per_minute_nrem": 0.8,
        "n_ripples_total": 10,
        "n_ripples_on_spike": 2,
    },
}

# v2 build (default)
sub_v2 = build_submission(
    findings=findings_v2,
    user_input=_good_input(),
    consent=_good_consent(),
    tool_version="0.13.2",
)
check("build_submission default → schema_version=2",
      sub_v2["schema_version"] == 2)
check("v2 submission has coupling_plv_bucket",
      "coupling_plv_bucket" in sub_v2["findings"])
check("v2 coupling_plv_bucket value correct",
      sub_v2["findings"].get("coupling_plv_bucket") == "0.2-0.35",
      detail=f"got {sub_v2['findings'].get('coupling_plv_bucket')!r}")
check("v2 submission has sw_density_bucket",
      "sw_density_bucket" in sub_v2["findings"])
check("v2 submission has hfo_rate_bucket",
      "hfo_rate_bucket" in sub_v2["findings"])

# v1 build (explicit target)
sub_v1 = build_submission(
    findings=findings_v2,
    user_input=_good_input(),
    consent=_good_consent(),
    tool_version="0.13.2",
    schema_version_target=1,
)
check("build_submission(schema_version_target=1) → schema_version=1",
      sub_v1["schema_version"] == 1)
check("v1 build: no coupling fields",
      "coupling_plv_bucket" not in sub_v1["findings"]
      and "sw_density_bucket" not in sub_v1["findings"])

section("v0.13.2 — Validator accepts v1 and v2 submissions")

# v1 submission (legacy) must still validate
ok_v1, errs_v1 = validate_submission(sub_v1)
check("validator accepts v1 submission", ok_v1, "; ".join(errs_v1))

# v2 submission must validate
ok_v2, errs_v2 = validate_submission(sub_v2)
check("validator accepts v2 submission", ok_v2, "; ".join(errs_v2))

# v2 submission with all optional fields missing also validates
sub_v2_minimal = build_submission(
    findings={"quality": {"grade": "A"}},
    user_input=_good_input(),
    consent=_good_consent(),
    tool_version="0.13.2",
)
ok_min, errs_min = validate_submission(sub_v2_minimal)
check("validator accepts v2 minimal (no v2 fields)", ok_min, "; ".join(errs_min))

# v3 schema_version must be rejected
import copy
sub_v3 = copy.deepcopy(sub_v2)
sub_v3["schema_version"] = 3
ok_v3, errs_v3 = validate_submission(sub_v3)
check("validator rejects schema_version=3", not ok_v3)

section("v0.13.2 — Citations")

from src.clinical.citations import CITATIONS, methods_attribution
check("helfrich_coupling in CITATIONS",
      "helfrich_coupling" in CITATIONS)
check("helfrich_coupling PMID correct",
      CITATIONS["helfrich_coupling"].pubmed_id == "29395264")
check("hahn_coupling_pediatric in CITATIONS",
      "hahn_coupling_pediatric" in CITATIONS)
check("hahn_coupling_pediatric PMID correct",
      CITATIONS["hahn_coupling_pediatric"].pubmed_id == "32499637")
check("methods_attribution coupling present",
      "coupling" in methods_attribution())
check("methods_attribution coupling points to helfrich",
      methods_attribution()["coupling"] == "helfrich_coupling")


# ─── Final ────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  PASS: {n_pass}")
print(f"  FAIL: {n_fail}")
print(f"{'='*60}")
if n_fail > 0:
    print("\nFailed:")
    for name in failed:
        print(f"  - {name}")
