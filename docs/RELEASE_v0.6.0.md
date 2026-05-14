# KCNQ3-Lens v0.6.0 — Clinical report restructure

**The report now reads the way clinicians actually read reports.**

v0.5 added the metrics a pediatric neurologist expects. v0.6 fixes the order they appear in: Impression first, then findings, then methods. Plus the recording-context fields that frame every interpretation (medications, recording date, indication), plus a proper sleep architecture report.

## What's new

### 1. Clinical Impression at the top of the doctor PDF

Every doctor PDF now opens with a one-paragraph **Impression** section — the section a busy clinician reads first. Rule-based (deterministic, auditable, no LLM), with hedged phrasing only ("consistent with", "compatible with", "raises concern for" — never "diagnoses").

Example for the reference KCNQ3 R230H profile:

> Background activity is severely slow (posterior dominant rhythm 4.0 Hz). Multi-regional epileptiform activity with maximum involvement of Pz, Cz, F4. Substantial NREM spike-wave activity (NREM SWI 18%, N3 SWI 32%). Sleep activation factor 7.2× (moderate; NREM rate 18.0/min vs wake rate 2.5/min). Sleep spindle density at Cz is markedly reduced (0.01/min vs age-typical 3–5/min). Dominant spread pattern is regional. Findings are consistent with: KCNQ-spectrum / multi-regional sleep-activated pattern, Speech-motor / SMA-region predominance. In the context of KCNQ3 p.Arg230His, the combination of multi-regional discharges, low spindle density, and slow background is characteristic of the spectrum disorder.

Below that: a structured **Recommendations / questions to discuss** section with the highest-priority follow-ups pulled from the matched patterns.

### 2. Recording metadata in the sidebar

New collapsed "📋 Recording metadata (optional)" expander captures the clinical context that changes interpretation:

- Patient label (anonymized — explicitly not PHI)
- Recording date, time of day, indication
- **Current medications** — the single most important context. The same EEG on Sultiam versus no medication is two different reports.
- Last medication change date
- Days since last seizure
- Technologist / clinical notes during recording

This metadata feeds into the PDF header, the Impression generator (variant-aware closing line when KCNQ is mentioned), and stays attached to the analysis throughout.

### 3. Sleep architecture report

Building on v0.5's sleep-stage classification, v0.6 computes the standard polysomnography architecture metrics:

| Metric | What it tells you |
|---|---|
| **REM latency** | Minutes from sleep onset to first REM. Normal pediatric: 90–180 min. |
| **WASO** (Wake After Sleep Onset) | Total wake minutes between sleep onset and final awakening. Healthy: < 30 min. |
| **Fragmentation index** | Stage transitions per hour. > 30/h = fragmented, non-restorative sleep. |
| **NREM cycles** | Count of complete NREM→REM cycles. Healthy pediatric: 4–6. |
| **Mean cycle duration** | Should average ~90 min in healthy sleep. |
| **First-cycle N3** | Most slow-wave sleep happens early. If this is low, memory consolidation is compromised. |
| **Total sleep time** | Minutes actually asleep within the recording. |

These metrics directly answer the questions a clinician would ask about sleep quality independent of the spike findings — and they make the spindle-density story make sense in context.

## What this looks like for a family

The doctor PDF now opens with one paragraph that summarizes the entire EEG in clinician-readable language. A parent can read that paragraph aloud to the doctor at the next appointment, and the doctor immediately knows what was found. The detailed tables follow for verification; the methods section at the back is there for the curious.

The parent PDF still leads with plain-language findings, but now also surfaces the SWI / activation factor / sleep architecture metrics in family-friendly form.

## Hardening

The v0.5.1 sanitization pass continues to guard every numeric output:
- Strict JSON serialization always succeeds (no NaN/Inf leaking)
- All numpy scalars are unboxed to native Python types
- Degenerate inputs (flat channels, all-zero signals) produce zeros instead of NaN

## Test coverage

| Version | Test suite |
|---|---|
| 0.6.0   | **61/61 PASS** (was 52 in v0.5.1) |
| 0.5.1   | 52/52 |
| 0.5.0   | 37/37 |

New v0.6 tests cover: RecordingMetadata roundtrip, build_impression on empty findings, build_recommendations on empty findings, sleep_architecture on all-wake (no REM), sleep_architecture with synthetic NREM-REM pattern (computes latency + cycles + first-cycle N3).

## What's deferred to v0.7+

From the doctor-perspective audit, these remain:

- **Reactivity analysis** (eyes open vs eyes closed background separately)
- **Standardized ILAE / ACNS terminology** in the report
- **Z-score plots** against age normatives
- **Confidence intervals** on burst counts and spike rates via bootstrap
- **"Negative findings" section** (what was checked and not found)
- **Reference citations** for every normative value
- **Anonymization helper** (strip identifiers from EEG file headers before sharing)

## Installation / upgrade

```bash
git pull
streamlit run app.py
```

No new dependencies.

## License

MIT. The DISCLAIMER.md still applies: KCNQ3-Lens is not a medical device.

---

**Full changelog**: see [CHANGELOG.md](../CHANGELOG.md)
**Commits in this release**: `60e7eae`
