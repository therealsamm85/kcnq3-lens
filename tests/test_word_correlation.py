"""Tests for the word-tracker correlation (Wave 11). Synthetic ground-truth
covering the n-gating honesty policy, plus a real round-trip through the
SQLite storage + diary API."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.longitudinal.storage import StoredEntry, save_entry, load_all_entries
from src.longitudinal.diary import DiaryEntry, append_entry, load_all_entries as load_diary
from src.longitudinal.trends import METRICS
from src.longitudinal.word_correlation import (
    compute_word_correlation, summarize_word_correlation,
    render_word_correlation_md, MIN_PAIRS_FOR_RHO, MIN_PAIRS_FOR_SIGNIFICANCE,
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


def _entry(date: str, **metric_values) -> StoredEntry:
    findings: dict = {}
    for metric, value in metric_values.items():
        path = _LABEL_TO_PATH[metric]
        cur = findings
        for k in path[:-1]:
            cur = cur.setdefault(k, {})
        cur[path[-1]] = value
    return StoredEntry(recording_date=date, label=date, findings=findings)


def _by_metric(result):
    return {mc.metric: mc for mc in result.metric_correlations}


# ── perfect anticorrelation: spikes down as words up ───────────────────────
print("\n── Wave 11: word correlation — expected-direction match ───────────")

dates = ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01", "2025-05-01"]
spikes = [20.0, 16.0, 12.0, 8.0, 4.0]   # falling
words = [10, 20, 35, 55, 80]            # rising → perfect rank anticorrelation
entries = [_entry(d, spike_rate_per_min=s) for d, s in zip(dates, spikes)]
diary = [DiaryEntry(date=d, word_count=w) for d, w in zip(dates, words)]

res = compute_word_correlation(entries, diary, metrics=["spike_rate_per_min"])
mc = _by_metric(res)["spike_rate_per_min"]
check("5 pairs formed (same-day matches)", mc.n_pairs == 5, f"got {mc.n_pairs}")
check("rho ≈ -1 (perfect anticorrelation)", mc.spearman_rho == -1.0, f"got {mc.spearman_rho}")
check("expected sign negative for spikes", mc.expected_sign == -1)
check("observed sign matches expected", mc.matches_expected is True)
check("p-value omitted at n=5 (< significance floor)", mc.p_value is None)
check("not flagged conclusive at n=5", mc.statistically_conclusive is False)
check("interpretation calls it a picture not proof",
      "picture" in mc.interpretation or "omitted" in mc.interpretation)
check("n_word_observations counted", res.n_word_observations == 5)


# ── positive correlation + maturation confound (PDR) ───────────────────────
print("\n── Wave 11: word correlation — PDR positive + confound flag ───────")

pdr = [4.0, 4.3, 4.6, 5.0, 5.4]   # rising
entries_pdr = [_entry(d, pdr_hz=p) for d, p in zip(dates, pdr)]
res_pdr = compute_word_correlation(entries_pdr, diary, metrics=["pdr_hz"])
mp = _by_metric(res_pdr)["pdr_hz"]
check("PDR rho ≈ +1 (rises with words)", mp.spearman_rho == 1.0, f"got {mp.spearman_rho}")
check("PDR expected sign positive", mp.expected_sign == +1)
check("PDR matches expectation", mp.matches_expected is True)
check("PDR flagged maturation-confounded", mp.maturation_confounded is True)
check("PDR interpretation mentions maturation", "maturation" in mp.interpretation)


# ── n-gating: too few pairs → no coefficient ───────────────────────────────
print("\n── Wave 11: word correlation — n-gating ───────────────────────────")

few = compute_word_correlation(entries[:3], diary[:3], metrics=["spike_rate_per_min"])
mf = _by_metric(few)["spike_rate_per_min"]
check(f"{MIN_PAIRS_FOR_RHO-1} pairs → rho is None", mf.spearman_rho is None and mf.n_pairs == 3)
check("too-few interpretation explains the floor", "too few" in mf.interpretation.lower())

# ≥ significance floor → p-value shown + conclusiveness evaluated.
big_dates = [f"2025-{m:02d}-01" for m in range(1, 9)]            # 8 months
big_spikes = [20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0]      # falling
big_words = [12, 10, 30, 40, 55, 70, 90, 110]                    # rising w/ one early swap
big_entries = [_entry(d, spike_rate_per_min=s) for d, s in zip(big_dates, big_spikes)]
big_diary = [DiaryEntry(date=d, word_count=w) for d, w in zip(big_dates, big_words)]
res_big = compute_word_correlation(big_entries, big_diary, metrics=["spike_rate_per_min"])
mb = _by_metric(res_big)["spike_rate_per_min"]
check(f"n={mb.n_pairs} ≥ {MIN_PAIRS_FOR_SIGNIFICANCE} → p-value shown", mb.p_value is not None,
      f"p={mb.p_value}")
check("strong negative corr at n=8 → conclusive", mb.statistically_conclusive is True,
      f"rho={mb.spearman_rho} p={mb.p_value}")


# ── degenerate inputs ──────────────────────────────────────────────────────
print("\n── Wave 11: word correlation — degenerate inputs ──────────────────")

# Zero variance in word counts.
flat_diary = [DiaryEntry(date=d, word_count=50) for d in dates]
res_flat = compute_word_correlation(entries, flat_diary, metrics=["spike_rate_per_min"])
check("zero word variance → rho None + explained",
      _by_metric(res_flat)["spike_rate_per_min"].spearman_rho is None
      and "variance" in _by_metric(res_flat)["spike_rate_per_min"].interpretation)

# Out-of-window word entry → no pair.
far = compute_word_correlation(
    [_entry("2025-01-01", spike_rate_per_min=10.0)],
    [DiaryEntry(date="2025-06-01", word_count=40)],  # ~151 days away > 45
    metrics=["spike_rate_per_min"], max_pair_gap_days=45,
)
check("word entry outside window is not paired",
      _by_metric(far)["spike_rate_per_min"].n_pairs == 0)

# No vocabulary at all → explanatory note, empty correlations.
none = compute_word_correlation(entries, [], metrics=["spike_rate_per_min"])
check("no word data → note + no correlations",
      none.n_word_observations == 0 and len(none.metric_correlations) == 0
      and any("No vocabulary" in n for n in none.notes))


# ── audit regressions (Wave 12) ────────────────────────────────────────────
print("\n── Wave 11: audit fixes — exact-p, non-finite, confound ───────────")

# CRITICAL: false significance from scipy's asymptotic p. The exact reviewer
# trigger (metric ranks 0..7, words 0,1,2,4,7,6,3,5) has rho=0.714 with
# asymptotic p=0.0465 (<0.05) but EXACT permutation p=0.0576 (NOT significant).
fs_dates = [f"2025-{m:02d}-01" for m in range(1, 9)]
fs_metric = [float(i) for i in range(8)]
fs_words = [0, 1, 2, 4, 7, 6, 3, 5]
fs_entries = [_entry(d, pdr_hz=v) for d, v in zip(fs_dates, fs_metric)]
fs_diary = [DiaryEntry(date=d, word_count=w) for d, w in zip(fs_dates, fs_words)]
fs = _by_metric(compute_word_correlation(fs_entries, fs_diary, metrics=["pdr_hz"]))["pdr_hz"]
check("exact-p reproduces rho≈0.714", fs.spearman_rho == 0.714, f"got {fs.spearman_rho}")
check("exact permutation p ≈ 0.0576 (not the asymptotic 0.0465)",
      fs.p_value is not None and abs(fs.p_value - 0.0576) < 0.002, f"got {fs.p_value}")
check("borderline corr NOT flagged conclusive (was false-True with asymptotic p)",
      fs.statistically_conclusive is False)
check("interpretation states not significant",
      "not statistically significant" in fs.interpretation)

# non-finite metric value (inf) is filtered out, not silently correlated.
inf_dates = ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01", "2025-05-01"]
inf_spikes = [5.0, 4.0, float("inf"), 2.0, 1.0]
inf_words = [10, 20, 30, 40, 50]
inf_entries = [_entry(d, spike_rate_per_min=s) for d, s in zip(inf_dates, inf_spikes)]
inf_diary = [DiaryEntry(date=d, word_count=w) for d, w in zip(inf_dates, inf_words)]
inf_mc = _by_metric(compute_word_correlation(inf_entries, inf_diary,
                                             metrics=["spike_rate_per_min"]))["spike_rate_per_min"]
check("inf biomarker dropped → only 4 finite pairs", inf_mc.n_pairs == 4, f"got {inf_mc.n_pairs}")
check("no inf value survives into the pairs",
      all(p.metric_value != float("inf") for p in inf_mc.pairs))

# delta_alpha_ratio now flagged maturation-confounded (was missing).
dar_entries = [_entry(d, delta_alpha_ratio=v) for d, v in zip(dates, [5.0, 4.0, 3.0, 2.0, 1.0])]
dar = _by_metric(compute_word_correlation(dar_entries, diary,
                                          metrics=["delta_alpha_ratio"]))["delta_alpha_ratio"]
check("delta_alpha_ratio flagged maturation-confounded", dar.maturation_confounded is True)
check("delta_alpha_ratio interpretation carries the maturation caveat",
      "maturation" in dar.interpretation)


# ── serialization + render ─────────────────────────────────────────────────
print("\n── Wave 11: serialization + render ────────────────────────────────")

check("summary is JSON-serializable",
      isinstance(json.dumps(summarize_word_correlation(res)), str))
md = render_word_correlation_md(res)
check("markdown is a non-empty string", isinstance(md, str) and len(md) > 50)
check("markdown carries the hypothesis-generating caveat",
      "causation" in md or "hypothesis" in md.lower() or "proven" in md)


# ── real round-trip through SQLite storage + diary API ─────────────────────
print("\n── Wave 11: real storage/diary round-trip ─────────────────────────")

tmp = Path(tempfile.mkdtemp())
reference_dates = ["2024-02-01", "2024-08-01", "2025-01-01", "2025-06-01", "2025-11-01"]
reference_spikes = [14.0, 15.0, 13.0, 14.0, 12.0]
reference_words = [5, 8, 12, 18, 30]
for d, s in zip(reference_dates, reference_spikes):
    save_entry(_entry(d, spike_rate_per_min=s, pdr_hz=4.0 + reference_dates.index(d) * 0.25),
               storage_dir=tmp)
for d, w in zip(reference_dates, reference_words):
    append_entry(DiaryEntry(date=d, word_count=w), path=tmp)

loaded = load_all_entries(storage_dir=tmp)
loaded_diary = load_diary(path=tmp)
check("5 recordings + 5 word obs reloaded",
      len(loaded) == 5 and sum(1 for e in loaded_diary if e.word_count is not None) == 5)
res_real = compute_word_correlation(loaded, loaded_diary, metrics=["spike_rate_per_min", "pdr_hz"])
spike_real = _by_metric(res_real)["spike_rate_per_min"]
check("real-API: 5 spike pairs formed", spike_real.n_pairs == 5, f"got {spike_real.n_pairs}")
check("real-API: p-value still omitted at n=5", spike_real.p_value is None)
check("real-API summary serializes",
      isinstance(json.dumps(summarize_word_correlation(res_real)), str))


print(f"\n{'='*60}\n  PASS: {n_pass}\n  FAIL: {n_fail}\n{'='*60}")
if n_fail:
    sys.exit(1)
