# Changelog

All notable changes to KCNQ3-Lens. Format inspired by [Keep a Changelog](https://keepachangelog.com/).

This project is in early development. The 0.x line is for the rare-epilepsy community to use and validate; a 1.0 release will come after sustained real-world use by multiple families and clinicians.

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

- All four clinical patterns (KCNQ-spectrum, CSWS, BECTS, SMA-predominance) still match Liyana's reference profile correctly after the gating fix.

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

### Verified on Liyana's pre-treatment EEG

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

- **Morphology analysis was over-counting events by 6–17×** vs published literature rates. Root cause: `find_peaks` was called against a threshold derived from a single MAD computed over the entire 6-hour concatenated sleep window. Large CSWS bursts inflated the global MAD, but during quiet inter-burst intervals the threshold was still low enough to catch noise peaks. Fix: per-30-second-epoch local MAD with an additional requirement that each peak exceed 3× the epoch's RMS. **Before: 71,630 events at 108.3/min on Liyana's reference recording. After: 12,881 events at 19.5/min — within clinically plausible literature range.**
- **Bursts analysis: `n_channels_involved` always returned 18/19** because the fixed 500-amplitude threshold was below baseline signal amplitude on NK EEG-1200A recordings. Fix: per-channel adaptive threshold based on each channel's median peak-to-peak amplitude over 2-second baseline tiles. **Before: every burst trivially showed full involvement. After: discriminative — focal bursts now correctly show 1–3 channels, generalized bursts show 15–19.**

Both fixes verified with the end-to-end smoke test; full pipeline still passes.

---

## [0.3.0-pre] — 2026-05-13

### Added

- **Multi-AI provider abstraction** with three concrete implementations: Anthropic Claude, OpenAI GPT, Google Gemini. Users select a provider in the sidebar and supply their own API key.
- **Pre/post-treatment comparison mode**: upload two EEG recordings, see directional deltas (improved / worsened / unchanged) per metric, plus an overall verdict. Includes a comparison-specific AI prompt that is more cautious about over-interpreting numerical changes.
- **German UI translation**: ~120 strings translated; English remains the source of truth with fallback for missing keys.
- **PDF reports**: doctor (technical, dense tables) and parent (plain language, with AI interpretation embedded if generated). Both via reportlab, no external binaries.
- **YASA integration as default spindle backend.** The heuristic envelope-percentile detector is now a fallback. Side-by-side comparison on Liyana's EEG showed the heuristic over-counted spindles by ~150× (483 spindles vs 3 validated). True spindle generation in her recording is near-absent — a much more clinically meaningful finding.
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
- All clinical-pattern thresholds were tuned against a single reference recording (Liyana's pre-treatment overnight EEG). They need refinement against more cases as the tool reaches more families.
- The auto sleep-onset detector is heuristic. It has a low-confidence flag and a sensible fallback, but it is not a substitute for proper polysomnography sleep staging.
- Tested on macOS only. Linux and Windows should work but have not been verified by maintainers.

---

## Looking ahead

Planned for 0.5.x:

- Longitudinal tracking across multiple recordings of the same child, including a symptom / developmental-milestone diary tied to the EEG timeline.
- Insights embedded directly in the PDF reports.
- Additional clinical patterns (Childhood Absence Epilepsy, Lennox-Gastaut, Doose).
- A research-grade CLI for batch processing with config hashes for reproducibility.
- Migration from the deprecated `google-generativeai` to `google-genai`.
