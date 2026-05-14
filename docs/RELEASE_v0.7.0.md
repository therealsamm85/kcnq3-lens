# KCNQ3-Lens v0.7.0 — Clinical credibility

**Five small modules that close the gap between "research tool" and "clinically defensible report."**

v0.5–v0.6 brought the metrics and the report structure. v0.7 adds the things a critical reviewer would point out: cited normative values, what was checked and found absent, confidence intervals, the standard medical vocabulary, and an anonymization workflow for safe sharing.

## What's new

### 1. Reference citations for every normative value

The methods section of the doctor PDF now closes with a References list:

> • **Tassinari 1971**: CSWS / ESES criterion — SWI ≥ 85% during slow-wave sleep.
> • **Wamsley et al. 2012** (PMID 22431760): Sleep spindle density thresholds and N2-stage normative ranges.
> • **Lacourse et al. 2019** (PMID 30107208): YASA spindle-detection algorithm.
> • **Vallat & Walker 2021** (PMID 34648426): YASA SleepStaging model.
> • **Hagne 1972**: Pediatric posterior dominant rhythm developmental norms.
> • **Niedermeyer & Lopes da Silva 2005**: Standard reference for normative EEG.
> • **Binnie 2003** (PMID 14636778): Transient cognitive impairment from interictal discharges.
> • **Gramfort et al. 2013** (PMID 24431986): MNE-Python.

Every normative claim in the report is now auditable.

### 2. Negative findings — what was checked and found absent

Clinical EEG reports always include this section. KCNQ3-Lens now generates it automatically. Example output on a healthy reference EEG:

> • No CSWS / ESES pattern: N3 spike-wave index is 20% (criterion ≥ 85%).
> • No predominant generalized spike-wave pattern: only 5% of events are generalized, with 10% complex spike-wave morphology.
> • No background slowing: posterior dominant rhythm is age-appropriate at 9.0 Hz.
> • No strong sleep activation: activation factor 1.2× (threshold ≥ 10×).
> • No sustained rhythmic bursts ≥ 10 seconds detected.
> • No spindle reduction: spindle density 4.0/min (norm 3–5/min).

Absence of a pattern is clinical information, not noise.

### 3. Bootstrap confidence intervals

`src/utils/bootstrap.py` provides `bootstrap_count_ci()` and `format_ci()`. Any rate metric can now be reported as "19.5/min (95% CI 17.2–22.0)" — clinicians prefer reported uncertainty over false precision.

Per-epoch resampling preserves within-epoch spike-wave structure. 1000 bootstraps by default; configurable.

### 4. ILAE / ACNS standardized terminology

`src/clinical/terminology.py` maps internal vocabulary to the names clinicians actually use:
- "sustained rhythmic burst at 2.5 Hz" → "Rhythmic Delta Activity (RDA, 0.5–4 Hz)" (ACNS 2021)
- synchrony pattern "bilateral_synchronous" → "Generalized — homologous regions L and R fire simultaneously" (ILAE 2017)

A pediatric epileptologist reading the report instantly recognizes the vocabulary.

### 5. Anonymization helper

`src/clinical/anonymize.py` strips patient identifiers from EEG file headers before sharing:

- **EDF / EDF+**: blanks the patient identification (bytes 8–88) and recording identification (bytes 88–168) fields.
- **Nihon Kohden EEG-1200A**: zeros the documented patient-info region (offsets 0x30–0x80) while preserving the file signature so readers still work.
- **Auto-detect**: `anonymize_auto()` dispatches by extension. Unsupported formats return a clear warning instead of failing silently.

Never modifies the original; creates a `_anonymized` copy.

> ⚠ This is a privacy helper, not a forensic anonymizer. It strips fields the maintainers know about. Other vendor-specific extensions may embed identifiers elsewhere.

## Test coverage

| Version | Tests | Streamlit |
|---|---|---|
| 0.7.0   | **81/81 PASS** | PASS |
| 0.6.0   | 61/61 | PASS |
| 0.5.1   | 52/52 | PASS |

20 new tests cover: citations registry membership + lookup; negative findings on rich (healthy-looking) input and empty input; ACNS / ILAE terminology mappings + unknown-pattern fallback; bootstrap point estimate + bounds ordering + empty-input handling + format_ci rendering; EDF / NK / auto anonymization on synthetic files (identifier stripped, signature preserved, unsupported-format warning).

## Status of the original doctor-perspective list

After v0.7, **12 of 16** items from the audit are closed:

| ✅ Done | ❌ Deferred |
|---|---|
| 1. SWI per stage | 8. Reactivity (eyes open/closed) — needs event markers |
| 2. Wake vs sleep split | 11. Z-score plots — needs pediatric normative DB |
| 3. Synchrony / spread | 14. CI **integration** in bursts/morphology |
| 4. Sample EEG traces in PDF | |
| 5. Methods section | |
| 6. Impression-first PDF | |
| 7. Recording metadata | |
| 9. ILAE / ACNS terminology | |
| 10. Sleep architecture | |
| 12. Negative findings | |
| 13. Reference citations | |
| 15. Anonymization helper | |
| 16. Audit trail | |

## Installation / upgrade

```bash
git pull
streamlit run app.py
```

No new dependencies.

## License

MIT. KCNQ3-Lens is not a medical device. See DISCLAIMER.md.

---

**Full changelog**: [CHANGELOG.md](../CHANGELOG.md)
**Commits in this release**: `b7670d0`
