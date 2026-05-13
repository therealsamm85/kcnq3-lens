# Roadmap

What we've built (v0.1), what's next (v0.2), and what's possible later. Contributions on any of these are welcome — open an issue first if it's larger than a small fix.

---

## v0.1 (shipped)

- Nihon Kohden EEG-1200A binary reader (novel)
- EDF / EDF+ / BDF / BrainVision / EEGLAB readers (via MNE)
- Five quantitative analyses:
  - Per-channel kurtosis topography
  - Sleep spindle density (Cz, age-normative)
  - Background power + posterior dominant rhythm
  - Sustained rhythmic burst detection
  - Spike-wave morphology classification
- Streamlit UI with file upload, per-analysis tabs, JSON export
- Multi-AI optional interpretation (Anthropic Claude, OpenAI GPT, Google Gemini)
- Privacy-first architecture: all processing local, only derived metrics sent to LLM
- MIT license, medical disclaimer

---

## v0.2 — next priorities

These are the features most likely to be high-impact for real families using the tool.

### Pre/post-treatment comparison ⭐

Upload two EEG recordings and see what changed between them. This is the single most valuable feature for any family tracking medication response — exactly the workflow we used to compare pre-Sultiam vs post-Sultiam in the reference patient's case.

- Side-by-side topography
- Spindle density change with significance test
- PDR shift
- Burst count delta
- Morphology distribution change
- LLM-generated "what changed and what likely matters" summary

### PDF report generation ⭐

A clean, professional report with plots, tables, and the AI interpretation that families can email to their doctor. Doctor-friendly version (technical detail) and parent-friendly version (plain language).

Use ReportLab or WeasyPrint; templates per audience.

### German UI translation

The first families likely to find this tool are German-speaking (Hamburg, Berlin, Munich pediatric neurology). A `README.de.md` plus a `i18n/` system in the app would lower the barrier. Volunteer translators welcome for other languages too.

### Time-of-night spike burden plot

The 30-min-bin chart we built to find the reference patient's first-NREM-cycle activation peak. Useful for: identifying when in the night spikes cluster, comparing across nights, showing the doctor the "shape" of the night.

### Auto-detect sleep onset / sleep window

Instead of manually entering sleep start/end seconds, use delta-band power, EMG (if a marker channel exists), and movement detection to estimate sleep onset within ~10 minutes. Reduces user error significantly.

---

## v0.3 — additional analyses

### High-frequency oscillation (HFO) detection

For recordings sampled at ≥500 Hz: detect ripples (80–250 Hz). HFO density is one of the best modern biomarkers for cognitive outcome in epilepsy. Limited applicability since most clinical EEGs are sampled at 200–250 Hz, but worth having for the families with research-grade recordings.

### Sleep stage estimation

A proper rule-based or model-based stager (N1/N2/N3/REM) per 30-second epoch. Lets stage-specific analyses (N2-only spindle density, N3-only SWI, REM-only morphology) be done correctly.

### Spike-Wave Index (SWI) — proper CSWS metric

Compute the formal SWI used in CSWS/ESES diagnosis: percentage of slow-wave sleep occupied by continuous spike-wave activity. Per stage, per cycle.

### Topographic scalp plots

Use MNE's `plot_topomap` to render brain-shaped scalp maps for kurtosis, spindle density, burst origin — much more intuitive than bar charts.

### Inter-hemispheric asymmetry

Left vs right comparison for all metrics. Important for catching focal patterns the standard reading might miss.

### Vertex sharp wave / K-complex detection

Additional sleep-architecture markers beyond spindles. Reduced vertex waves correlate with neurodevelopmental disorders.

---

## v0.4 — community & longitudinal

### Longitudinal tracking

Store findings to a local JSON file across recordings. Plot trends over months / years. The same child's spindle density, PDR, burst count plotted over time — answers the "is treatment working" question with quantitative evidence.

### Opt-in anonymous aggregate dataset

For families willing to share, contribute anonymized derived metrics (NOT raw EEG, NOT identifiers) to a community dataset. Over time this becomes a research resource that could meaningfully expand the KCNQ3 literature.

### Variant-specific comparison

If your child has KCNQ3 R230H and the community dataset has 30 other R230H recordings, show your child's metrics relative to that cohort. Same for KCNT1, SCN1A, STXBP1, GRIN2A, etc.

### Literature reference panel

For each variant entered, pull the most relevant recent papers from PubMed and link them in the report. Saves families from having to manually search for "KCNQ3 p.Arg230His" every visit.

---

## v0.5 — production polish

### Standalone installers

PyInstaller / py2app / py2exe builds so non-technical parents can double-click to install rather than learn `pip`.

### Anonymization helper

A one-click button to strip patient identifiers from EEG files before sharing (e.g. with a second-opinion doctor abroad). Important because hospital EEG exports often contain patient name, birth date, MRN in the header.

### Recording quality assessment

Flag bad channels, high-noise epochs, electrode drift. Some EEGs are simply too poor-quality to analyze meaningfully and the tool should say so honestly rather than producing misleading numbers.

### Export to MNE-Raw format

For clinicians who want to do their own analysis in MNE-Python or other tools, expose a "Save as EDF" or "Save as MNE Raw" option.

### Migrate Gemini provider to google-genai

The current `google-generativeai` SDK is deprecated. Switch to `google-genai` (different package, similar API).

---

## What we deliberately won't build (at least not initially)

- **Cloud hosting / shared accounts.** Medical data privacy is too easy to get wrong. Local-only is a feature, not a limitation.
- **Diagnostic claims.** This will never be a "the reference patient has CSWS / R230H disorder" output. It surfaces patterns; doctors interpret.
- **Treatment recommendations.** Not from the rule-based code, not from the AI layer. Hard rule.
- **Automatic medication adjustment guidance.** Same as above.
- **Patient data storage outside the user's device.** Even with opt-in, individual data stays local; only fully anonymized aggregate metrics may eventually be shared.

---

## How to contribute

1. Open an issue describing what you want to build or improve.
2. For larger changes (a new analysis, a new file format, a UX rework), let's discuss before you spend a weekend on it.
3. PRs welcome. Keep the medical-safety guardrails — anything that suggests diagnosis or treatment will not be merged.
4. Test against at least one real EEG before submitting (synthetic data is fine for unit tests but doesn't catch the messy real-world cases).

---

## Names to reach out to (parents and researchers, when this is ready)

- **RIKEE** (rikee.org) — the existing KCNQ patient registry. Worth letting them link to the tool.
- **Prof. Sarah Weckhuysen** (Antwerpen) — KCNQ research lead in Europe.
- **Prof. Ed Cooper** (Baylor) — KCNQ research lead in the US.
- **N=1 Collaborative** (Boston) — n-of-1 ASO research for rare epilepsies.
- **CureKCNQ Family Foundation** — patient advocacy.
- **EpiCARE** — European Reference Network for Rare and Complex Epilepsies.
- KCNQ3 Facebook groups (a few exist, ~100–300 members each).
