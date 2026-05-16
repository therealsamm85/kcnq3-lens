# Roadmap

## Status — what has shipped

### v0.1–v0.7 (foundation + clinical depth)

- Nihon Kohden EEG-1200A binary reader (novel open-source implementation)
- Five core analyses: spike topography, sleep spindle density, background power + PDR, sustained burst detection, spike morphology
- Streamlit UI with file upload, per-analysis tabs, JSON export
- Multi-AI interpretation (Claude, GPT, Gemini — user's own key)
- Pre/post-treatment comparison mode
- German UI translation
- Topographic scalp maps, time-of-night spike burden chart
- Auto sleep-onset detection, recording quality control (A–D grade)
- Proactive clinical insights: anatomical mapping, pattern recognition, cross-modal observations
- SWI (Spike-Wave Index), wake/sleep state split, bilateral synchrony analysis
- YASA integration as default spindle backend (replaced heuristic over-counter)
- Sleep stage classification (YASA + delta/alpha heuristic fallback)
- Doctor and parent PDF reports (ReportLab)
- Bootstrap confidence intervals, ACNS/ILAE terminology
- Clinical citations module, negative findings panel
- Anonymization helper for EDF/NK headers
- v0.11.1: corrected Wamsley citation error — pediatric spindle norms updated to McClain 2016 + Kwon 2023 (previous values were ~3× too high)

### v0.12.0–v0.12.4 (federated registry)

- SQLite local storage for longitudinal tracking
- Registry schema v1 + de-identification submission builder (allowlist-by-construction, PHI regex sweep)
- PHI scanner fix + registry repo cross-link
- Contribute mode: pre-filled GitHub PR flow, no backend required
- Aggregates download + peer-comparison UI (k-anonymized cohort percentiles)

### v0.13.0–v0.13.3 (Tier 2 analyses)

- v0.13.0: Slow-wave detection (SO density, amplitude, duration — NREM3 marker)
- v0.13.1: HFO ripple detection (Staba-style energy detector, 80–250 Hz)
- v0.13.2: SO-spindle coupling (PLV-based coupling angle and strength) + registry schema v2 (HFO rate, coupling buckets, IED method fields)
- v0.13.3: Automated IED detection — ensemble heuristic (morphology + template + amplitude); opt-in SpikeNet wrapper (stub — requires local model weights)

---

## Tier 3 — NEXT (open, not started)

These are the most impactful next contributions. Open an issue before starting anything larger than a small fix.

### Clinical validation study

Work with a pediatric neurologist to compare tool outputs against expert-scored polysomnograms and clinical EEG reads on ≥10 recordings. Without this, all interpretation labels remain "tool convention" only. This is the highest-leverage thing anyone can do to improve the tool's real-world utility.

### BIDS-EEG export

Export findings and (optionally) the anonymized recording as a [BIDS-EEG](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html) dataset. Would allow families to share data in a format research labs can directly ingest, without custom parsing.

### Pediatric YASA tuning

YASA's SleepStaging model is trained on adult polysomnography. A pediatric-specific model (or a correction layer) would substantially improve sleep-stage accuracy for the 2–12-year-old range that most KCNQ3 families fall into. Likely requires a labeled pediatric polysomnography dataset (CHB-MIT is not staged; DREAMS or SEDF may be useful starting points).

### UI integration of Tier-2 findings

Slow waves, HFO ripples, coupling, and IED are computed but only exposed as JSON output and in the doctor PDF. A dedicated UI tab (or extension of the existing Clinical tab) showing these findings interactively — with interpretation caveats in-line — would make them accessible to families.

### Additional clinical patterns

Childhood Absence Epilepsy, Lennox-Gastaut, Doose syndrome pattern recognizers in `src/insights/patterns.py`. Requires clinical review of criteria before merging.

---

## What we deliberately will not build

- **FDA / CE certification.** This is a research and family-support tool, not a regulated medical device, and it will stay that way until a formal clinical-validation pathway is funded and staffed.
- **Real-time / streaming EEG.** Architecture is built for offline overnight recordings. Live EEG requires a fundamentally different stack.
- **Mobile app.** EEG files are large, processing is CPU-intensive, and the user base is small. A native mobile app would cost more than it would help.
- **Cloud hosting / shared accounts.** Local-only is a feature, not a gap. Medical data privacy is too easy to get wrong at scale.
- **Diagnostic claims.** The tool surfaces patterns; clinicians interpret. Hard rule — no output will ever say "this child has [condition]".
- **Treatment recommendations.** Same as above.

---

## How to contribute

1. Open an issue describing what you want to build.
2. For anything larger than a small fix — discuss before spending a weekend on it.
3. PRs must maintain the medical-safety guardrails. Anything that implies diagnosis or treatment will not be merged.
4. Test against at least one real EEG before submitting (synthetic data is fine for unit tests, but doesn't catch real-world edge cases).

---

## Contacts (for when the tool is ready to reach more families)

- **RIKEE** (rikee.org) — existing KCNQ patient registry; worth linking to the tool
- **Prof. Sarah Weckhuysen** (Antwerpen) — KCNQ research lead in Europe
- **Prof. Ed Cooper** (Baylor) — KCNQ research lead in the US
- **N=1 Collaborative** (Boston) — n-of-1 ASO research for rare epilepsies
- **CureKCNQ Family Foundation** — patient advocacy
- **EpiCARE** — European Reference Network for Rare and Complex Epilepsies
- KCNQ3 Facebook groups (~100–300 members each)
