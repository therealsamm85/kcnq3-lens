# Changelog

All notable changes to KCNQ3-Lens. Format inspired by [Keep a Changelog](https://keepachangelog.com/).

This project is in early development. The 0.x line is for the rare-epilepsy community to use and validate; a 1.0 release will come after sustained real-world use by multiple families and clinicians.

---

## [0.19.0] — 2026-06-14

Twelve new capabilities from a verified survey of mature open-source EEG tools
(what they offer that this tool lacked), each built with an explicit
build-vs-borrow decision, then hardened by a 42-agent adversarial code review.

### Added — 12 features (build-vs-borrow noted)

- **Annotated EDF+ export** (borrow edfio) — hand the neurologist an EDF+ with
  the tool's spike/SWI/HFO marks, openable in free EDFbrowser.
- **Entropy / complexity** (build) — sample/permutation/spectral entropy, Hjorth,
  Higuchi FD, Lempel-Ziv: background-disorganization markers.
- **Graph-theory network metrics** (build) — clustering, efficiency, small-world
  σ on the existing wPLI matrices.
- **Raw-trace viewer** (borrow mne/matplotlib) — stacked-trace windows + desktop
  Qt browser, so a human can eyeball the waveform behind a number.
- **Spike-triggered averaging → peak topography** (build) — focal/regional/
  bilateral field readout from the detected IEDs.
- **Two-stage HFO classification** (build) — artifact rejection + spike-coupled
  (spkHFO) flag on the existing HFO events.
- **Ictal screener** (build, flag-for-review) — sensitivity-first rhythmic-
  evolution detector for electrographic seizures; confidence capped at moderate.
- **ICA + ICLabel** (borrow mne-icalabel + fallback) — component-based ocular/
  muscle/cardiac removal.
- **ASR** (borrow asrpy + fallback) — transient burst correction.
- **Age-normative qEEG z-scores** (build engine; norms pluggable) — ships
  UNVERIFIED placeholder norms behind a do-not-use-clinically banner.
- **SCORE/IFCN-structured report** (build) — re-words findings into the familiar
  clinical section layout.
- **Micromed .TRC reader** (borrow python-neo) + Natus note.

### Fixed — 42-agent adversarial code review (30 confirmed defects)

A batched per-module review (one reviewer + independent refute-by-default
verification each) found 30 defects, all reproduced against the real code and
fixed with regression tests. Dominant class: **non-finite (NaN/inf) inputs
leaking into clinical values** (5 critical) — now guarded at every module
boundary. Also: EDF integer-rate write, spike focal-vs-bilateral ordering,
Micromed channel-name normalization, ICA data-truncation + honest W-ICA note,
duplicate-channel-name aliasing, and several honesty/doc corrections.

### Wired

- All 6 per-recording analyses + the SCORE report run inside `run_all_analyses`
  (error-resilient); the 4 transforms stay opt-in tools.
- Doctor PDF gains a "Novel quantitative metrics (v0.19)" section; the app's
  advanced tab surfaces them + the SCORE report, and single-recording mode adds
  on-demand ICA/ASR/EDF+ tools.

Verified: full suite 1325+ checks, 0 fail (12 new suites + regressions + a
runner-wiring smoke test).

---

## [0.18.21–0.18.23] — 2026-06-14

Two family-facing longitudinal features the family asked for ("an EEG reader
nobody has seen"), each built as a wave with synthetic + real-DB verification,
then hardened by a 10-agent adversarial audit (4 review lenses → independent
refute-by-default verification of every flagged defect).

### Added

- **Treatment-response dashboard** (`longitudinal/treatment_response.py`).
  Anchors each stored EEG biomarker to the medication-change events in the
  development diary and reports the before→after change per biomarker with a
  clinical direction (improved / worsened / no-clear-change). Every comparison
  is one recording before vs one after, so it carries explicit maturation /
  sleep-state / measurement caveats and never claims causation. Reads stored
  findings only — no EEG re-read.
- **Vocabulary ↔ EEG correlation** (`longitudinal/word_correlation.py`).
  Spearman rank correlation of diary word counts against each biomarker
  (nearest-in-time pairing), with the expected clinical direction stated. Strict
  honesty gating for a handful-of-recordings reality: no coefficient below 4
  pairs, **exact permutation p** (never the anti-conservative asymptotic
  approximation) shown only at ≥8 pairs, maturation-confounded biomarkers
  flagged. Hypothesis-generating, explicitly not a significance test.
- Shared single-source-of-truth helpers: `metric_polarity.py` (which direction
  is clinically "better") and `time_align.py` (date parse, before/after split,
  nearest-in-time match). Both surfaced in the Streamlit longitudinal view.

### Fixed — from the adversarial audit (6 confirmed defects)

- **CRITICAL** word-correlation reported false significance at n=8: scipy's
  default p is the asymptotic t-approximation (anti-conservative at small n), so
  rho=0.71 was flagged significant (p=0.047) when the exact permutation p is
  0.058. Now uses an exact permutation p (full enumeration for n≤8).
- **CRITICAL** a NaN biomarker rendered as a false "worsened" in the
  treatment-response dashboard (and emitted invalid bare-`NaN` JSON). Non-finite
  findings are now dropped at the series boundary → `not_evaluable`.
- **HIGH** `delta_alpha_ratio` is maturation-confounded (same axis as PDR) but
  was missing from the confound set, so its caveat was suppressed in both
  features. Added.
- **MEDIUM** an `inf` biomarker produced an ordinary-looking correlation; now
  filtered. **LOW** `nearest_within` tie-break now honours "earlier date" on
  unsorted input.

Verification: `tests/test_treatment_response.py` 51/0 and
`tests/test_word_correlation.py` 35/0 (synthetic ground-truth, the audit
regressions, and real round-trips through the SQLite storage/diary API). Full
suite green (1142 checks, 0 fail).

---

## [0.18.4–0.18.15] — 2026-06-13

A reader-correctness audit followed by a preprocessing/QC feature build
(Tier 1–3 from a review of mature open-source EEG tools). Every wave was
verified on real recordings + synthetic ground truth before commit.

### Fixed — silent reader bugs (0.18.4–0.18.5)

- **µV calibration in the long-form NK reader.** The reverse-engineered
  fallback path returned raw int16 ADC counts, ~10× mis-scaled vs the MNE
  path. Applied the NK EEG-1200A fixed gain (0.09766 µV/count). The 24h
  recording now reads at physiological amplitude.
- **Dead/flat channel mapping.** Analyses fell back only on *absent*
  channels, not present-but-dead ones, so a detector could run silently on a
  0.4 µV unplugged electrode (it once reported 0 slow waves). Added
  `EEGRecording.is_channel_live()` / `resolve_live_channel()` and wired them
  into slow-waves, spindles, morphology, and the PDR posterior average.
- **Case-sensitive EEG-channel detection.** Upper-case montages ("FP1",
  "FZ", "CZ") silently lost the midline channels the sleep detectors need.
  Now case-insensitive across all reader paths.

### Added — preprocessing & QC layer (Tier 1, 0.18.6–0.18.8)

- **PREP-style channel QC** (`quality.py`): µV-correct thresholds (the old
  ADC-scale thresholds were dead on µV data) plus relative "noisy" and
  correlation-based "uncorrelated" flags that catch a junk reference channel.
- **Lazy common-average re-referencing + interpolation**
  (`src/preprocessing/reference.py`): per-epoch CAR over good channels +
  neighbour interpolation of bad channels; opt-in; works on 24h recordings.
- **Ocular/blink detection + epoch masking** (`src/preprocessing/ocular.py`):
  found that the early "frontal topographic drift" was largely eye-blink.

### Added — Tier 2 / Tier 3

- **Per-channel epoch rejection** (`src/preprocessing/artifact.py`):
  Autoreject-inspired data-driven thresholds (0.18.10).
- **Minimal privacy-preserving BIDS-EEG export** (`src/reports/bids.py`):
  de-identified by construction; optional EDF signal export (0.18.11).
- **Debiased wPLI connectivity** (`src/analyses/connectivity.py`): robust to
  volume conduction; quantifies the thalamocortical decoupling (0.18.12).
- **Broadband sharpness-gated spike detector** (`src/analyses/spike_sharp.py`):
  additive second estimate; on real data ~28% of the 10–30 Hz morphology
  "spikes" do not pass the sharpness gate (rhythmic contamination) (0.18.13).
- **Event annotation export for human review** (`src/reports/annotations.py`):
  BIDS events.tsv round-trip so a clinician can confirm/reject candidates
  (0.18.14).
- **Longitudinal spike-burden biomarker tracker**
  (`src/longitudinal/biomarker.py`): same channel + threshold across
  timepoints — an objective treatment-response measure (0.18.9).

### Wiring (0.18.15)

- ocular / connectivity / sharp-spikes integrated into the runner and the
  doctor PDF ("Advanced quantitative metrics" section). The opt-in transforms
  (re-reference, epoch rejection, BIDS, biomarker, annotations) are tools the
  user invokes, not auto-run.

### Tests

- 5 new suites (preprocessing, biomarker, connectivity, spike_sharp,
  annotations) — 65 checks; full project 1051 checks, 0 fail.
- `edfio` and `pypdf` added as explicitly optional dependencies.

---

## [0.13.3] — 2026-05-15

### Added — automated IED detection (Tier 2)

- `src/analyses/ied_ml.py`: ensemble heuristic (morphology score + template correlation + amplitude threshold). Opt-in SpikeNet wrapper present as a stub — requires locally downloaded model weights, not distributed with the tool.
- Schema v2 fields for IED method flag (`ied_method_bucket`) integrated into registry submission builder.
- Named constants `HFO_PCT_ON_SPIKE_BUCKETS` + `SW_METHODS` in registry for round-trip stability.

**Note:** The IED detector is rule-based, not ML. The "SpikeNet" path is a stub pending a separately trained model. False positive / false negative rates are unknown — treat output as a screening flag only.

---

## [0.13.2] — 2026-05-15

### Added — SO-spindle coupling + registry schema v2

- `src/analyses/coupling.py`: PLV-based SO-spindle coupling (coupling angle and strength per night). Descriptive — no pediatric normative ranges currently incorporated.
- Registry schema v2: adds `hfo_rate_bucket`, `coupling_strength_bucket`, `coupling_angle_bucket`, `sw_density_bucket` fields. Schema version bumped; existing v1 submissions remain valid.

---

## [0.13.1] — 2026-05-15

### Added — HFO ripple detection

- `src/analyses/hfo_ripples.py`: Staba-style energy detector for 80–250 Hz ripples. Requires ≥500 Hz sampling rate — most clinical EEGs (200–250 Hz) will return zero detections by design.
- Outputs: HFO rate per minute, per-channel density, co-occurrence with IEDs.

**Note:** HFO detection is a research field with no consensus algorithm or validated pediatric norms. Output is a research metric, not a clinical measurement.

---

## [0.13.0] — 2026-05-15

### Added — slow-wave detection (first Tier 2 component)

- `src/analyses/slow_waves.py`: slow-oscillation (SO) density, mean amplitude, and duration in NREM3 epochs. Amplitude-and-duration heuristic; no pediatric normative database for comparison.
- Tier 1 e2e integration test (`tests/test_tier1_e2e.py`) covering the live pipeline across both repos.

---

## [0.12.0–0.12.4] — 2026-05-15

### Added — federated registry pipeline

- v0.12.0: SQLite local storage for longitudinal tracking of EEG metrics across recordings.
- v0.12.1: Registry schema v1 + de-identification submission builder. Allowlist-by-construction architecture — fields not in the schema cannot appear in output. PHI regex sweep.
- v0.12.2: PHI scanner fix + registry repo cross-link (`therealsamm85/kcnq3-registry`).
- v0.12.3: Contribute mode — pre-filled GitHub PR flow with one-JSON-line submission. No backend; audit trail via git history.
- v0.12.4: Aggregates download + peer-comparison UI. k-anonymized cohort percentiles displayed in-app. Closes the full registry loop.

---

## [0.11.0–0.11.1] — 2026-05-15

### Added

- Mac + Windows + Linux standalone installers (PyInstaller). No Python needed — double-click to run.
- Public sample EEG data (CHB-MIT, PhysioNet) bundled for first-time users.

### Fixed — v0.11.1

- **Corrected four hallucinated citations and values.** Most critically: spindle norm reference was Wamsley 2012, which is an adult schizophrenia paper containing no pediatric norms. Replaced with McClain 2016 (n=8, ages 2–5) and Kwon 2023 (n=567, ages 0–18). Revised norms are roughly 3× lower at age 5 than previously claimed. Interpretation labels for all pre-v0.11.1 recordings were therefore overly optimistic ("in range" when density was actually "below").

---

## [0.9.0–0.10.1] — 2026-05-14

### Added

- v0.9: Quick Start mode — parent-friendly 4-step guided UI.
- v0.9.1: Fix Streamlit widget-key/session-state conflict.
- v0.9.2: Streamlit AppTest runtime tests.
- v0.10: Live EEG-trace viewer with event overlays.
- v0.10.1: Copy-paste AI prompt for families without API keys.

---

## [0.8.0–0.8.2] — 2026-05-14

### Added

- **Longitudinal tracker**: store findings to SQLite across recordings; plot metric trends over time. Answers "is treatment working?" with a graph.
- CI integration for the test suite.
- v0.8.1: Corrupt-input handling and degenerate edge-case hardening.
- v0.8.2: Fix YASA SleepStaging integration + cycle counter.

---

## [0.7.0] — 2026-05-14

### Added — clinical credibility

- **Reference citations** (`src/clinical/citations.py`): 8 indexed entries (Tassinari, Wamsley, Lacourse, Vallat, Hagne, Niedermeyer, Binnie, Gramfort) with full citation + PubMed ID + URL + supporting-claim note. Rendered at the end of doctor PDF's methods section.
- **Negative findings** (`src/clinical/negative_findings.py`): plain-language statements of what was checked and not present (no CSWS, no generalized SW, no background slowing, no strong activation, no sustained bursts, channel-quality OK, spindles in normal range). Doctor PDF section.
- **Bootstrap confidence intervals** (`src/utils/bootstrap.py`): `bootstrap_count_ci(per_epoch_values, aggregate, n_bootstrap)` + `format_ci()`. Resampling is per-epoch.
- **ILAE / ACNS terminology** (`src/clinical/terminology.py`): `acns_pattern_for_burst(freq_hz)` → ACNS 2021 names. `ilae_descriptor_for_synchrony(pattern_id)` → ILAE 2017 descriptors.
- **Anonymization helper** (`src/clinical/anonymize.py`): strips patient identifiers from EDF / NK headers. Auto-detect by extension. Creates `_anonymized` copy, never modifies original.

### Test suite

- 81/81 tests pass (was 61).
- 20 new tests covering citations, negative findings, terminology, bootstrap, EDF + NK + auto anonymization paths.

### Version

0.6.0 → 0.7.0.

---

## [0.6.0] — 2026-05-14

### Added — clinical report restructure (Impression first)

- **Doctor PDF now opens with an Impression section** — one-paragraph clinician-readable summary built rule-based from findings (`src/clinical/impression.py`). Hedged phrasing ("consistent with", "raises concern for") — never diagnoses. Followed by a Recommendations / questions-to-discuss section pulled from matched patterns.
- **Recording metadata** (`src/clinical/metadata.py`): RecordingMetadata dataclass captures patient label, age, sex, variant, recording date / time-of-day / indication, current medications, last medication change date, days since last seizure, technologist notes. Surfaces in PDF header and feeds the Impression generator.
- **Sidebar form** in Streamlit for entering metadata at analysis time.

### Added — sleep architecture metrics

- **`src/analyses/sleep_architecture.py`**: builds on v0.5 sleep stages to produce REM latency, WASO (Wake After Sleep Onset), sleep fragmentation index (transitions per hour), count and mean duration of complete NREM cycles, first-cycle N3 minutes, total sleep time, sleep onset / final awakening times.
- Integrated into the doctor PDF as a new section. Sleep architecture metrics answer the "sleep quality" questions independent of spike findings.

### Test suite

- 61/61 tests pass (was 52).
- 9 new v0.6 tests: RecordingMetadata roundtrip, empty-findings impression, empty-findings recommendations, sleep_architecture all-wake handling, sleep_architecture REM-latency / cycle / first-cycle-N3 computation.

### Version

0.5.0 → 0.6.0.

---

## [0.5.0] — 2026-05-14

### Added (the five gaps a real pediatric neurologist asked for)

- **Formal Spike-Wave Index (SWI) per sleep stage** (`src/analyses/swi.py`): Tassinari definition — % of stage time covered by continuous SW bursts (≥1 spike/s sustained ≥3s). CSWS / ESES criterion check (N3 SWI ≥ 85%) with automatic red-banner alert when met.
- **Wake vs sleep spike-rate split with activation factor** (`src/analyses/state_split.py`): separate rates per state plus the single number every epileptologist uses. Labels: none / mild / moderate / strong (≥10× = dramatic activation).
- **Bilateral synchrony / spread analysis** (`src/analyses/synchrony.py`): for each detected spike, checks ±50ms co-firing window and classifies into five patterns: focal, regional, bilateral synchronous, bilateral asynchronous, generalized. Distribution and dominant pattern reported.
- **Sleep stage classification** (`src/analyses/sleep_stages.py`): YASA SleepStaging wrapper with µV-scaling and minimal-channel configuration. Heuristic delta/alpha fallback when YASA unavailable. Returns per-30s labels, stage minutes, sleep efficiency, NREM cycle count.
- **Sample EEG traces in PDF**: new `plot_eeg_trace()` produces clinical-style multi-channel stacked plots. Doctor PDF accepts `sample_traces=list[(caption, png_bytes)]` and embeds them as 16×8 cm images.
- **Methods section in PDF**: complete algorithm + parameter + reference documentation. Software version + analysis timestamp embedded. References Tassinari, Lacourse, Wamsley, Hagne, Niedermeyer.

### Added (UI)

- New "Clinical" tab in the findings view (between Quality and Topography). Shows SWI per stage (5 columns), state split (4 metrics), synchrony distribution (bar chart), sleep architecture (bar chart + sleep efficiency / cycle count). EN + DE strings.

### Verified

- 37/37 edge-case tests pass. Suite extended from 29 to 37 covering the new modules.
- Graceful degradation: SWI on all-wake returns 0% (no crash). State split with zero wake minutes uses fallback definition (no div-by-zero). Synchrony with no events returns 'no_events' pattern.
- Per-analysis try/except in the runner: one failure cannot abort the others.

### Notes

YASA's `SleepStaging` model is trained on adult polysomnography. Pediatric output is flagged as `confidence='heuristic'`. The output is still useful for SWI calculation; it's not a substitute for human-scored pediatric polysomnography.

---

## [0.4.1] — 2026-05-13

### Fixed

- **Pattern matcher: false-positive matches on empty / normal findings.** Two bugs were caught by the new edge-case test suite:
  - With `findings = {}`, the BECTS pattern fired because `(pct_complex_spike_wave or 0) < 30` evaluated `True` for missing data. Same issue affected the SMA and CSWS criteria with `or 0` defaults.
  - Children with normal EEGs (high simple-spike %, low complex %, no bursts) legitimately satisfied BECTS' supporting criteria — but without any centro-temporal focus.
- The fix introduces a `required=True` flag on `PatternCriterion`. Each pattern now has at least one gating criterion that must be met for the pattern to appear at all; supporting criteria only refine confidence beyond that gate.

### Added

- `tests/test_edge_cases.py` — 29 tests across 8 categories: empty findings, unknown channels, degenerate comparisons, AI router error paths, i18n missing keys, PDF generation with sparse fields, pattern sanity, reader errors, short-recording sleep onset. All pass.
- i18n coverage check confirms zero gaps between EN and DE translations.

### Verified

- All four clinical patterns (KCNQ-spectrum, CSWS, BECTS, SMA-predominance) still match the reference patient's reference profile correctly after the gating fix.

---

## [0.4.0] — 2026-05-13

### Added

- **Proactive clinical insights** — the major v0.4 feature. A new `src/insights/` module produces deterministic, rule-based interpretation of findings:
  - **Anatomical mapping** (`anatomical.py`): each 10-20-system channel mapped to its underlying brain region and primary functional contribution. Six functional networks (speech-motor, language, executive, sensorimotor, salience, visual) with anatomy + function + clinical implications.
  - **Clinical pattern recognition** (`patterns.py`): four patterns — KCNQ-spectrum, CSWS/ESES, BECTS/Rolandic, SMA / speech-motor predominance — each scored with confidence (weak/moderate/strong) and explicit criteria.
  - **Cross-modal observations** (`narrative.py`): combinations of findings that imply more than each alone (e.g. low spindles + sustained bursts → memory consolidation impact).
- New Insights section in the Streamlit UI between findings tabs and AI interpretation. Expanders show each top network's clinical implications and each pattern's criteria + suggested questions for the doctor.
- EN + DE translations for all insights strings.

### Notes

All insights output is deterministic (no LLM). The optional AI interpretation feature is still available alongside; insights are the rule-based, auditable layer.

---

## [0.3.2] — 2026-05-13

### Added

- **Auto sleep-window detection** (`src/analyses/sleep_onset.py`): computes per-epoch delta/alpha ratio on central channels, finds the longest contiguous sleep-like run with short-gap bridging for spike-cluster interruptions. Sanity check + fallback to a conventional overnight window if detection fails.
- **Quality control flags** (`src/analyses/quality.py`): per-channel flat/saturated/extreme detection, per-epoch artifact identification, overall A/B/C/D grade with human-readable warnings.
- New "🔍 Auto-detect sleep window" button in the sidebar. New "Quality" tab in the findings view (placed first to establish trust context for everything else).

### Verified on the reference patient's pre-treatment EEG

- Grade A, 92.5% usable epochs, 16/19 good channels.
- Cz/Pz correctly flagged as "extreme amplitude" — this is real high-amplitude spike-wave activity, not artifact, and the doctor reading the report can interpret it correctly.

---

## [0.3.1] — 2026-05-13

### Added

- **Topographic scalp map** (`src/utils/plots.py`): MNE-based `plot_topomap()` using the standard 10-20 montage. Renders epileptiform-activity heatmap on a brain-shaped scalp view. Falls back to a bar chart when too few channels map to the standard layout.
- **Time-of-night spike burden chart**: bins spikes into 30-minute windows across the recording. Highlights the peak time (typically 1–3 hours after sleep onset for sleep-activated patterns like CSWS).
- Both added to the findings tabs (Topography tab gets the topomap; new "Time of night" tab gets the burden chart).

---

## [0.3.0] — 2026-05-13

### Fixed (CRITICAL / HIGH — both surfaced by the v0.3 analysis audit)

- **Morphology analysis was over-counting events by 6–17×** vs published literature rates. Root cause: `find_peaks` was called against a threshold derived from a single MAD computed over the entire 6-hour concatenated sleep window. Large CSWS bursts inflated the global MAD, but during quiet inter-burst intervals the threshold was still low enough to catch noise peaks. Fix: per-30-second-epoch local MAD with an additional requirement that each peak exceed 3× the epoch's RMS. **Before: 71,630 events at 108.3/min on the reference patient's reference recording. After: 12,881 events at 19.5/min — within clinically plausible literature range.**
- **Bursts analysis: `n_channels_involved` always returned 18/19** because the fixed 500-amplitude threshold was below baseline signal amplitude on NK EEG-1200A recordings. Fix: per-channel adaptive threshold based on each channel's median peak-to-peak amplitude over 2-second baseline tiles. **Before: every burst trivially showed full involvement. After: discriminative — focal bursts now correctly show 1–3 channels, generalized bursts show 15–19.**

Both fixes verified with the end-to-end smoke test; full pipeline still passes.

---

## [0.3.0-pre] — 2026-05-13

### Added

- **Multi-AI provider abstraction** with three concrete implementations: Anthropic Claude, OpenAI GPT, Google Gemini. Users select a provider in the sidebar and supply their own API key.
- **Pre/post-treatment comparison mode**: upload two EEG recordings, see directional deltas (improved / worsened / unchanged) per metric, plus an overall verdict. Includes a comparison-specific AI prompt that is more cautious about over-interpreting numerical changes.
- **German UI translation**: ~120 strings translated; English remains the source of truth with fallback for missing keys.
- **PDF reports**: doctor (technical, dense tables) and parent (plain language, with AI interpretation embedded if generated). Both via reportlab, no external binaries.
- **YASA integration as default spindle backend.** The heuristic envelope-percentile detector is now a fallback. Side-by-side comparison on the reference patient's EEG showed the heuristic over-counted spindles by ~150× (483 spindles vs 3 validated). True spindle generation in her recording is near-absent — a much more clinically meaningful finding.
- **README sections** establishing scope: "What this tool is for / NOT for", minimum recording requirements, comparison with seven other EEG tools (MNE, YASA, EDFbrowser, Persyst, Brainstorm/EEGLAB, Luna, NeuroKit2), and "Built on the shoulders of giants" acknowledgments.
- **Analysis audit pipeline**: `data/_session/current/audit_findings.md` documents the validation of every analysis against an alternative method.
- **Session state for resumption**: `handoff.md`, `plan.md`, `decisions.md` under `data/_session/current/`.

### Verified

- Topography and Background PDR analyses pass audit (real findings, not artifacts).
- Morphology and Bursts flagged as having bugs that would be fixed in 0.3.0.

---

## [0.1.0] — 2026-05-13

### Added

- **Nihon Kohden EEG-1200A binary reader** (`src/readers/nihon_kohden.py`) — the genuinely novel contribution. No other open-source tool reads this format correctly: MNE-Python and EDFbrowser only see the 1-second setup block, not the multi-hour overnight data block. Reverse-engineered from the file family of a real patient recording.
- **Five quantitative analyses**: topography (per-channel kurtosis), sleep spindles, background power + posterior dominant rhythm, sustained rhythmic bursts, spike-wave morphology.
- **Streamlit frontend** with file upload, plotted findings, JSON export.
- **Optional AI interpretation** via Anthropic Claude (multi-provider support came in 0.3.0-pre).
- **MIT license** and prominent **medical disclaimer** (DISCLAIMER.md).
- **Scope-defining README** clarifying what the tool is and is not for.

---

## Verification notes

| Version | Smoke test | Edge-case suite | Streamlit boot | New file?  |
|---------|------------|-----------------|----------------|------------|
| 0.7.0   | PASS       | 81/81 PASS      | PASS           | src/clinical/{citations,negative_findings,terminology,anonymize}.py, src/utils/bootstrap.py |
| 0.6.0   | PASS       | 61/61 PASS      | PASS           | src/clinical/{metadata,impression}.py, src/analyses/sleep_architecture.py |
| 0.5.1   | PASS       | 52/52 PASS      | PASS           | src/utils/sanitize.py |
| 0.5.0   | PASS       | 37/37 PASS      | PASS           | src/analyses/{sleep_stages,swi,state_split,synchrony}.py |
| 0.4.1   | PASS       | 29/29 PASS      | PASS           | tests/test_edge_cases.py |
| 0.4.0   | PASS       | n/a             | PASS           | src/insights/ |
| 0.3.2   | PASS       | n/a             | PASS           | src/analyses/{sleep_onset, quality}.py |
| 0.3.1   | PASS       | n/a             | PASS           | src/utils/plots.py, src/analyses/time_of_night.py |
| 0.3.0   | PASS (numbers now plausible) | n/a | PASS | — (bug fixes only) |
| 0.3.0-pre | PASS     | n/a             | PASS           | src/{ai,comparison,i18n,reports}/ |
| 0.1.0   | PASS       | n/a             | PASS           | full foundation |

---

## What's not yet validated

- The Nihon Kohden EEG-1200A reader has only been tested on one recording family. Other recordings in this format may have minor header variations.
- All clinical-pattern thresholds were tuned against a single reference recording (the reference patient's pre-treatment overnight EEG). They need refinement against more cases as the tool reaches more families.
- The auto sleep-onset detector is heuristic. It has a low-confidence flag and a sensible fallback, but it is not a substitute for proper polysomnography sleep staging.
- Tested on macOS only. Linux and Windows should work but have not been verified by maintainers.

---

## What's next

See [ROADMAP.md](ROADMAP.md) for Tier 3 candidates and open contributions.
