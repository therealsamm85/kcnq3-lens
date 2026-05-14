# Plan — KCNQ3-Lens v0.5: Clinical credibility

## Objective

Close the five most-cited gaps from a pediatric neurologist's perspective on the v0.4.1 release. Make the tool credible enough that a real epileptologist would defend its output in a clinical conference.

## Out of scope (deferred to v0.6+)

- Standardized clinical report restructure (Impression → Findings → Methods reorder)
- HFO detection at high sample rates
- Hyperventilation / photic stimulation response analysis
- Reactivity (eyes open / eyes closed) analysis
- Z-score / percentile visualizations

## Phases

### Phase 1 — Sleep stage classification
- Goal: per-30s-epoch labels (W / N1 / N2 / N3 / REM) covering the recording
- Owner: sonnet
- Allowed files: `src/analyses/sleep_stages.py` (new)
- Verification: returns same length as `rec.n_epochs`; on Liyana's recording, plausible split (≥40% NREM in sleep window)
- Exit: `compute_sleep_stages()` returns a `SleepStageResult` with per-epoch labels
- Implementation: wrap YASA's `SleepStaging.predict()` with NK EEG-1200A data scaled to µV

### Phase 2 — Formal SWI per sleep stage
- Goal: % of each NREM stage occupied by continuous spike-wave activity, with CSWS threshold (≥85% in slow-wave sleep)
- Owner: sonnet
- Allowed files: `src/analyses/swi.py` (new)
- Depends on: Phase 1 (sleep stages)
- Verification: SWI values 0–100, CSWS flag fires only when N3-SWI ≥ 85
- Exit: `compute_swi()` returns `SWIResult` with per-stage SWI + CSWS flag

### Phase 3 — Wake vs sleep spike rate split
- Goal: separate spike-rate calculation for wake vs NREM, plus activation factor (sleep / wake)
- Owner: sonnet
- Allowed files: `src/analyses/state_split.py` (new) — derives from existing morphology detector
- Depends on: Phase 1
- Verification: wake_rate < sleep_rate on Liyana's reference recording; activation factor > 2
- Exit: `compute_state_split()` returns rates per state + activation factor

### Phase 4 — Bilateral synchrony / spread analysis
- Goal: for each detected spike, classify spread pattern (focal / regional / bilateral synchronous / bilateral asynchronous)
- Owner: sonnet
- Allowed files: `src/analyses/synchrony.py` (new)
- Verification: returns distribution of spread categories; on Liyana's data, multi-regional spread should dominate
- Exit: `compute_synchrony()` returns `SynchronyResult` with category percentages

### Phase 5 — Sample EEG traces in PDF
- Goal: render multi-channel 10-second EEG plots for the most clinically important events
- Owner: sonnet
- Allowed files: `src/utils/plots.py` (extend), `src/reports/pdf.py` (extend)
- Implementation: matplotlib stack plot of 19 channels around event center; embed as image in PDF
- Verification: PDF generates with sample-trace images; image bytes > 5 KB each
- Exit: doctor PDF includes 3 sample traces (top burst, sample spindle, wake background)

### Phase 6 — Methods section in PDF
- Goal: explicit algorithm + parameter documentation in the PDF, with software version + analysis timestamp + file hash
- Owner: sonnet
- Allowed files: `src/reports/pdf.py` (extend), `src/__init__.py` (version)
- Verification: PDF contains a Methods section listing each analysis's algorithm and key parameters
- Exit: methods section visible in both doctor and parent PDFs

### Phase 7 — Hardening + tests
- Goal: regression-tested integration
- Owner: sonnet
- Allowed files: `tests/test_edge_cases.py` (extend), `tests/test_end_to_end.py` (extend)
- Verification: all tests pass; Streamlit boots clean
- Exit: commit v0.5

## Naming

- All new analyses follow the existing pattern: `compute_<name>()` returns `<Name>Result` dataclass + `summarize_<name>()` returns dict.
- New files in `src/analyses/` registered in `src/analyses/__init__.py`.
- Runner integrates new analyses preserving error-resilience (per-analysis try/except).

## Risk register

- **YASA sleep staging is trained on adult PSG** — may misclassify pediatric stages. Mitigation: still ship, mark `confidence='heuristic'`, document caveat.
- **Synchrony depends on accurate single-spike detection** — if morphology over-counts (as it did pre-v0.3), synchrony stats become meaningless. Mitigation: reuse the validated per-epoch local-MAD detector from morphology.py.
- **Sample traces double the PDF size** — image embedding adds ~30 KB per plot. Mitigation: PNG with 80 DPI, ≤3 traces total.
