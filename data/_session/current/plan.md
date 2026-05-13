# Plan — KCNQ3-Lens

## Objective

Open-source quantitative EEG analysis tool for families and clinicians treating children with rare epilepsies (especially KCNQ3 spectrum). Provides Nihon-Kohden-EEG-1200A support no other open-source tool has, plus validated quantitative analyses with optional multi-LLM interpretation. Privacy-first (all local), multi-language (EN/DE), family-accessible.

## Out of scope

- Cloud hosting / shared server
- Replacing clinical EEG software (Persyst, NeuroWorks)
- Real-time streaming analysis
- Adult EEG (algorithms tuned for pediatric)
- Source localization (use MNE directly)
- General EEG viewing (use EDFbrowser)
- Diagnostic claims / medical advice

## Phases

### v0.1 — Foundation [DONE]
- Goal: working repo with NK reader + 5 analyses + Streamlit UI
- Files: `src/readers/`, `src/analyses/`, `app.py`, README, DISCLAIMER, LICENSE
- Exit criteria: end-to-end smoke test on Liyana's EEG passes

### v0.2 — Multi-AI + Comparison + i18n [DONE]
- Goal: provider abstraction, pre/post comparison, German UI
- Files: `src/ai/providers/`, `src/comparison/`, `src/i18n/`, `app.py`, `README.de.md`
- Exit criteria: all three providers register; comparison computes deltas; language switch works

### v0.2.5 — Positioning + YASA [DONE, uncommitted]
- Goal: README scope/comparison sections; replace heuristic spindle with YASA
- Files: README.md, README.de.md, `src/analyses/spindles.py`, requirements.txt
- Exit criteria: YASA is default backend; heuristic still works as fallback; smoke test passes

### v0.3 — Clinic-ready features ← **NEXT**
- Goal: PDF reports, time-of-night chart, topographic scalp maps, auto-sleep-onset, QC flags
- Owner model: sonnet (mechanical UI + reporting work)
- Allowed files: `src/reports/`, `src/analyses/qc.py`, `src/utils/plots.py`, `app.py`
- Forbidden files: existing analysis modules (audit them in parallel, don't modify)
- Verification: end-to-end test still passes; PDF generates on Liyana's findings; topographic plot renders
- Exit criteria: doctor sees a printable PDF; time-of-night chart shows the 00:00-01:30 spike-burden peak

### v0.3-audit — Validate remaining 4 analyses ← **PARALLEL TO v0.3**
- Goal: cross-check topography, background, bursts, morphology against alternative implementations
- Owner model: sonnet
- Allowed files: new files under `tests/audit_*.py` only — must NOT modify analyses yet
- Verification: each audit script runs and produces a comparison report
- Exit criteria: each analysis has a documented "validated within X% of [alternative]" or "needs revision because Y"

### v0.4 — Longitudinal tracker + symptom diary [PLANNED]
- Goal: families log multiple recordings + development notes over time
- Files: `src/storage/`, new Streamlit page

### v0.5 — Researcher CLI + batch mode [PLANNED]
- Goal: command-line batch processing with reproducibility (config hashes)
