"""Tests for A2 — entropy/complexity metrics (analyses/entropy.py).

Ground-truth strategy: a regular signal (sine) must score LOWER on every
irregularity/complexity estimator than broadband noise."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.readers.base import EEGRecording
from src.analyses.entropy import (
    compute_entropy, summarize_entropy,
    _perm_entropy, _spectral_entropy, _sample_entropy, _hjorth,
    _higuchi_fd, _lziv_complexity,
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


def _make_rec(signal_1d, sfreq=100.0, ch="Pz"):
    data = signal_1d.reshape(1, -1).astype(np.float32)
    rec = EEGRecording(
        path=Path("/tmp/e.eeg"), sfreq=sfreq, n_channels=1,
        duration_s=signal_1d.size / sfreq, channel_names=[ch],
        n_channels_in_file=1, eeg_channel_indices=[0], format_name="synthetic",
    )
    rec._full_data = data
    return rec


sf = 100.0
t = np.arange(60 * int(sf)) / sf
rng = np.random.RandomState(0)
sine = 30.0 * np.sin(2 * np.pi * 10 * t)
noise = 30.0 * rng.randn(t.size)
# Brownian/random-walk: smooth (low fractal dimension ≈1.5) — a valid Higuchi
# ground-truth partner for white noise (FD≈2). A pure sine is NOT (it violates
# the self-affine assumption, so its Higuchi FD is unstable / can exceed 2).
brown = np.cumsum(rng.randn(t.size)).astype(float)
brown = 30.0 * brown / np.std(brown)


# ── individual estimators on known inputs ──────────────────────────────────
print("\n── A2: estimators on known signals ────────────────────────────────")

mono = np.arange(500.0)
check("perm_entropy of a monotonic ramp ≈ 0 (one ordinal pattern)",
      _perm_entropy(mono) < 0.01, f"got {_perm_entropy(mono)}")
check("perm_entropy of noise is high (≈1)", _perm_entropy(noise) > 0.9,
      f"got {_perm_entropy(noise)}")
check("spectral_entropy(sine) < spectral_entropy(noise)",
      _spectral_entropy(sine, sf) < _spectral_entropy(noise, sf))
check("sample_entropy(sine) < sample_entropy(noise)",
      _sample_entropy(sine[:1500]) < _sample_entropy(noise[:1500]))
check("higuchi_fd(brown) < higuchi_fd(noise) (smooth < rough)",
      _higuchi_fd(brown) < _higuchi_fd(noise), f"{_higuchi_fd(brown):.2f} vs {_higuchi_fd(noise):.2f}")
check("higuchi_fd(noise) is near 2", 1.6 < _higuchi_fd(noise) <= 2.05,
      f"got {_higuchi_fd(noise)}")
check("higuchi_fd(brown) is in the fractal range [1,2]",
      1.0 <= _higuchi_fd(brown) <= 2.0, f"got {_higuchi_fd(brown)}")
check("lziv(constant) < lziv(noise)",
      _lziv_complexity(np.ones(500)) < _lziv_complexity(noise))
mob_s, _ = _hjorth(sine)
mob_n, _ = _hjorth(noise)
check("hjorth_mobility(sine) < hjorth_mobility(noise)", mob_s < mob_n,
      f"{mob_s:.3f} vs {mob_n:.3f}")


# ── full pipeline: sine vs noise recording ─────────────────────────────────
print("\n── A2: compute_entropy pipeline ───────────────────────────────────")

res_sine = compute_entropy(_make_rec(sine))
res_noise = compute_entropy(_make_rec(noise))
res_brown = compute_entropy(_make_rec(brown))

keys = {"perm_entropy", "spectral_entropy", "sample_entropy", "hjorth_mobility",
        "hjorth_complexity", "higuchi_fd", "lziv_complexity"}
check("all 7 metrics present", set(res_sine.metrics) == keys, f"got {set(res_sine.metrics)}")
check("metrics finite for a normal signal",
      all(np.isfinite(v) for v in res_noise.metrics.values()))
check("noise more irregular: perm_entropy",
      res_noise.metrics["perm_entropy"] > res_sine.metrics["perm_entropy"])
check("noise more irregular: sample_entropy",
      res_noise.metrics["sample_entropy"] > res_sine.metrics["sample_entropy"])
check("noise more complex than brown: higuchi_fd",
      res_noise.metrics["higuchi_fd"] > res_brown.metrics["higuchi_fd"])
check("epochs were used", res_sine.n_epochs_used >= 1, f"got {res_sine.n_epochs_used}")
check("backend is numpy", res_sine.backend == "numpy")


# ── degenerate handling ────────────────────────────────────────────────────
print("\n── A2: degenerate handling ────────────────────────────────────────")

dead = compute_entropy(_make_rec(np.zeros(60 * int(sf))))
check("flat channel → no usable epochs, no crash", dead.n_epochs_used == 0)
short = compute_entropy(_make_rec(sine[:100]))  # < 1 epoch (30s)
check("sub-epoch recording handled", short.n_epochs_used == 0
      and any("one epoch" in n for n in short.notes))

# summarize converts NaN → None and is JSON-friendly.
import json
summ = summarize_entropy(dead)
check("summarize NaN → None + JSON-serializable",
      isinstance(json.dumps(summ), str)
      and all(v is None or isinstance(v, (int, float))
              for v in summ["metrics"].values()))

# fallback channel surfaced when the requested name (and all candidates) absent.
res_fb = compute_entropy(_make_rec(noise, ch="T3"), target_channel="Pz")
check("absent requested channel → fallback used + flagged",
      res_fb.channel == "T3" and res_fb.is_fallback_channel is True,
      f"got {res_fb.channel} fb={res_fb.is_fallback_channel}")


print(f"\n{'='*60}\n  PASS: {n_pass}\n  FAIL: {n_fail}\n{'='*60}")
if n_fail:
    sys.exit(1)
