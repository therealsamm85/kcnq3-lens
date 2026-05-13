# Handoff — KCNQ3-Lens v0.3 build + audit — 2026-05-13 17:25

## Goal

Build v0.3 features of KCNQ3-Lens (PDF reports, time-of-night spike-burden plot, topographic scalp maps, auto-sleep-onset detection, quality-control flags) **and** audit the four remaining analysis modules (topography, background, bursts, morphology) for over-counting / under-counting issues comparable to the YASA-vs-heuristic spindle finding from v0.2-pre.

## Phase status

- [x] v0.1 — initial repo + Nihon Kohden reader + 5 analyses + Streamlit app
- [x] v0.2 — multi-AI providers (Claude/GPT/Gemini), pre/post comparison, German UI
- [x] v0.2.5 — README positioning (scope, comparison table, what-it-is-NOT), YASA integration as default spindle backend (heuristic now fallback). Committed: NO — uncommitted on disk.
- [~] v0.3 — IN PROGRESS:
  - [x] PDF reports (doctor + parent versions) — DONE, tested on Liyana's data
  - [ ] Time-of-night spike-burden chart — not started
  - [ ] Topographic scalp maps (MNE topomap) — not started
  - [ ] Auto-sleep-onset detection — not started
  - [ ] QC flags (bad channels, artifact epochs) — not started
  - [ ] Formal stadium-specific SWI — not started
- [x] v0.3-audit — **COMPLETED**. Findings at `data/_session/current/audit_findings.md`.
  Two real bugs found:
  - **Morphology**: same structural bug as spindle heuristic — global MAD over
    full concatenated trace causes noise-peak detection in quiet intervals.
    85.8 events/min is 6-17× published literature rates. CRITICAL.
  - **Bursts**: `n_channels_involved` always returns 18/18 because 500 µV
    threshold is below baseline signal amplitudes. Burst counts have ±3×
    uncertainty across threshold multipliers. HIGH.
  - Background PDR (4 Hz) and Topography (Fp1/F4) are validated as REAL findings.

## Completed in this session (Sonnet 4.7, ~25 turns)

- Built `kcnq3-lens` repo at `/Users/weddad/Claude/kcnq3-lens` (~3200 LOC)
- Nihon Kohden EEG-1200A reader (`src/readers/nihon_kohden.py`) — genuinely novel
- 5 analyses: topography, spindles, background, bursts, morphology
- Streamlit frontend with single-recording + pre/post-comparison modes
- Multi-AI: Anthropic / OpenAI / Gemini providers with router abstraction
- i18n: English + German, lightweight Translator class, ~120 strings
- README.md + README.de.md with full scope/comparison/acknowledgments sections
- ROADMAP.md
- **YASA integration** — replaced heuristic spindle backend with validated YASA detector after side-by-side test on Liyana's EEG showed heuristic over-counted ~150× (483 vs 3 spindles in 6h)
- `tests/test_end_to_end.py` + `tests/compare_yasa.py` + `tests/yasa_sensitivity.py`

## Files touched

```
kcnq3-lens/
├── README.md                              (extended with scope/comparison/ack)
├── README.de.md                           (mirrored to German)
├── ROADMAP.md                             (new)
├── DISCLAIMER.md
├── LICENSE                                (MIT)
├── requirements.txt                       (added yasa)
├── app.py                                 (mode + language selector, ~430 LOC)
├── src/
│   ├── readers/{base,nihon_kohden,edf,auto_detect}.py
│   ├── analyses/{topography,spindles,background,bursts,morphology}.py
│   │   └── spindles.py                   (REFACTORED: YASA default + heuristic fallback)
│   ├── ai/{base,prompt,router}.py
│   │   ├── providers/{anthropic,openai,gemini}_provider.py
│   ├── i18n/{__init__,translations}.py    (new)
│   ├── comparison/compare.py              (new)
│   └── runner.py                          (new)
└── tests/{test_end_to_end,compare_yasa,yasa_sensitivity}.py
```

## Verification status

- End-to-end smoke test PASSING on Liyana's pre-treatment EEG
- Streamlit boot PASSING (port 8504)
- Both YASA and heuristic spindle paths verified
- All 3 AI providers register (only Anthropic key tested in this session)
- App.py syntax compiles cleanly

## Errors encountered & fixed

- YASA 0.7 API change: `spindles_detect` returns `SpindlesResults` not `DataFrame` — use `.summary()`. Fixed in `tests/compare_yasa.py:117`.
- NK ADC scaling: int16 values range ±32k, YASA expects µV. Solved by auto-rescaling to target_amplitude_uv (default 20µV std). Fixed in `src/analyses/spindles.py:_detect_with_yasa()`.

## Open risks / unknowns — POTENTIAL SURPRISES IN OTHER ANALYSES

**This is the most important section.** The YASA finding raised the hypothesis that other analyses may also over- or under-count. Audit targets:

1. **`topography.py` (kurtosis)** — kurtosis is a known robust spike marker but tends to also pick up muscle and movement artifact. Compare against **mne-features** or **eelbrain** spike detection. Liyana's "Fp1 = 8.58 / F4 = 8.57" topography max may include eye-blink artifact (Fp electrodes are particularly prone).

2. **`background.py` (PDR)** — our 4 Hz "PDR" on the night recording is suspiciously low. Likely contaminated by drowsy/peri-sleep states classified as "wake". Should validate against alert-wake EEGs (the four daytime recordings at /Volumes/INTENSO/NKT/EEG2100/). If daytime PDR is also ~4 Hz, finding stands. If it's 7-8 Hz, our wake-window detection is broken.

3. **`bursts.py` (sustained rhythmic events)** — we found 104 bursts ≥10s. **Cross-check** against MNE's `find_events` or detect_artifacts. The verified 19s burst at 00:20 had Pz amplitude 10× quiet baseline and 18/19 channels involved — that one is real. But 104 of them? Some are likely amplitude-elevated NREM transients (K-complexes, vertex waves) not subclinical events.

4. **`morphology.py` (simple/sharp/complex %)** — 56% complex was on Pz with 71630 detections. The "complex" classifier (≥200ms FWHM on broadband) may be confusing slow waves with riding spikes for actual spike-wave complexes. Validate by visual inspection of 20 random "complex" detections.

## Naming conventions in force

- All imports use absolute paths from `src.*`
- All modules use `from __future__ import annotations`
- Public analysis fns return dataclasses; `summarize_*()` returns dict for AI/UI
- All files inherit MIT license + DISCLAIMER reference
- German strings under `src/i18n/translations.py` with `_AGE_NORMS`-style underscore prefix for module-private dicts
- Test scripts under `tests/` named `test_*.py` (smoke), `compare_*.py` (cross-method), `*_sensitivity.py` (parameter sweep)

## Exact next step

The next session must do TWO things, in this order:

### 1. Commit current state (uncommitted on disk)

Everything below is ready to commit:
- README + DE README with scope/comparison/acknowledgments
- YASA integration as default spindle backend
- PDF reports module (`src/reports/pdf.py`, both doctor and parent versions, wired into app.py)
- ROADMAP.md, DISCLAIMER.md, this session/ directory

The commit message in the handoff's previous version is good — use it.

### 2. Fix the two bugs the audit surfaced — CRITICAL before v0.3 release

**Bug A (Morphology, CRITICAL):**
- Location: `src/analyses/morphology.py`, line ~60-110
- Bug: `find_peaks` is called against `mad_multiplier * mad` where `mad` is
  computed once over the full concatenated 6h trace. Large CSWS bursts inflate
  this global MAD, but the threshold still catches noise peaks in quiet
  inter-burst intervals.
- Fix: Replace global MAD with **per-epoch (30s) MAD**, OR add a hard floor
  requiring peak amplitude ≥ 3× the local (epoch-level) RMS. Pattern: compute
  threshold inside `iter_epochs` loop, not before.
- Validation: after fix, re-run `tests/test_end_to_end.py` — events/min should
  drop from 85.8 to roughly 5–15/min based on literature ranges.
- Cross-check: MNE has no built-in spike detector; consider `mne-features`
  `compute_spect_slope` as a tangential validator, or visual inspection of 10
  random detected events as we discussed.

**Bug B (Bursts, HIGH):**
- Location: `src/analyses/bursts.py`, line ~109-117
- Bug: `n_channels_involved` counts channels where `np.ptp(multi[:, i:j]) > 500`.
  The 500 µV (= 500 ADC unit) threshold is far below baseline amplitude on
  this NK recording (signal exceeds 500 µV during ordinary background).
  Result: every burst trivially shows 18-19/19 channel involvement.
- Fix: Replace fixed 500 with a per-channel adaptive threshold. Suggested:
  compute each channel's median peak-to-peak amplitude over a quiet baseline
  window, then count channels exceeding 3× that as "involved." Or simpler:
  count channels exceeding `5× MAD` for that specific channel.
- Validation: real 19s burst should still show ~18/19 channels; a small/noise
  burst should show 1-3 channels. The current implementation can't make that
  distinction.

**Bug C (Background PDR, LOW):**
- Not a code bug — the 4 Hz PDR is real (confirmed across 24h overnight +
  4 daytime EEGs). But: add a "clinical confirmation recommended" caveat to
  the PDF reports and AI interpretation prompt when `interpretation ==
  "severely_slow"`.

**Bug D (Topography, COSMETIC):**
- Not a code bug — Fp1 vs F4 ordering is sampling noise at this margin.
- Optional cosmetic fix: in `summarize_topography`, add a `_FRONTAL_POLE` flag
  on Fp1 and Fp2 entries, displayed in PDF/UI as "(artifact-prone)".

### Recommended next-session prompt

> "Read `data/_session/current/handoff.md` and `audit_findings.md`. Fix the
> two CRITICAL/HIGH bugs documented in the 'Exact next step' section
> (Morphology global-MAD + Bursts n_channels_involved). After each fix, run
> `python -m tests.test_end_to_end` and report the before/after numbers.
> Do not start any new v0.3 features (time-of-night chart, topo plots,
> auto-sleep, QC flags) until both bugs are fixed and verified."

## Recommended model for next phase

- **Audit subagent**: `sonnet` (mechanical comparison, no novel algorithm design)
- **v0.3 main-thread work**: `sonnet` (PDF templating + UI wiring is mechanical)
- **Escalate to `opus`**: only if the audit surfaces a real algorithmic problem requiring redesign

## Compaction suggestion

**spawn_fresh_subagent** — context is ~80% full. The two next workstreams (audit + v0.3 build) are independent; running them in the same thread risks running out of context mid-build.

Recommended split:
1. Spawn audit subagent with the prompt above (background)
2. Continue v0.3 PDF reports in current thread until context warning, then checkpoint again
3. Merge audit findings into v0.3 in a fresh post-compaction session
