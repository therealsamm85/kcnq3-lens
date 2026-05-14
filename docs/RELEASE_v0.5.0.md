# KCNQ3-Lens v0.5.0 — Clinical-grade release

**The first release with the full set of metrics a pediatric neurologist actually expects from a quantitative EEG report.**

The v0.4.1 line proved the foundation worked. v0.5 closes the five gaps that were holding it back from clinical credibility — every one identified by walking through the tool from a real epileptologist's perspective: "if I got this report on my desk, what would I want that's missing?"

## What's new

### 1. Formal Spike-Wave Index per sleep stage

The single most-requested number for CSWS / ESES diagnosis. Tassinari's definition: percentage of each sleep stage occupied by continuous spike-wave activity (≥1 spike/s sustained ≥3 seconds). The CSWS criterion fires automatically when **N3 SWI ≥ 85%**.

The Clinical tab shows SWI broken down across all five stages (W / N1 / N2 / N3 / REM), with a CSWS banner that turns red when the criterion is met.

### 2. Wake vs sleep spike-rate split + activation factor

Separate spike rates for wake, NREM, and REM — plus a single number every epileptologist uses: **activation factor = NREM rate / wake rate**.

Reported with labels:
- `none` (< 1.5×) — no sleep activation
- `mild` (1.5–3×)
- `moderate` (3–10×) — significant sleep activation
- `strong` (≥ 10×) — dramatic activation, typical of CSWS

### 3. Bilateral synchrony / spread analysis

For each detected spike, the tool now checks which other channels co-fire within ±50 ms. Classifies into five patterns:

- **Focal** — 1–2 channels
- **Regional** — 3–5 channels on the same hemisphere
- **Bilateral synchronous** — homologous L+R pairs with balanced involvement
- **Bilateral asynchronous** — both sides involved without simultaneous pairs (suggests propagation)
- **Generalized** — ≥10 channels involved

The distribution is shown as a bar chart with a named dominant pattern. This is what tells you whether a recording is multifocal vs truly generalized.

### 4. Sample EEG traces in PDF

New `plot_eeg_trace()` produces clinical-style multi-channel stacked plots. The doctor PDF accepts these as embedded images with captions. Most clinicians believe what they see, not what an algorithm asserts.

### 5. Methods section in PDF

Every analysis is now documented with its algorithm, parameters, and citation. Software version + analysis timestamp embedded. Pediatric neurologists reviewing the report can verify the math, not just trust the numbers.

References cited: Tassinari (CSWS criterion), Lacourse (YASA spindle detection), Wamsley (pediatric spindle norms), Hagne (PDR), Niedermeyer (pediatric EEG textbook).

## Bonus: sleep architecture

Sleep stage classification (via YASA when available, heuristic fallback otherwise) unlocked:

- Per-stage minute breakdown (W / N1 / N2 / N3 / REM)
- Sleep efficiency percentage
- Estimated number of NREM cycles
- Method + confidence flag (`yasa` / `fallback_delta_alpha` / `fallback_no_channel`)

The pediatric-staging caveat is explicit: YASA's model is trained on adult polysomnography. The output for children should be interpreted as approximate.

## Hardening

- **37/37 edge-case tests pass.** Test suite extended from 29 to 37 covering the new v0.5 modules.
- **Graceful degradation everywhere**: if YASA fails, fall back. If sleep stages are all-wake, downstream SWI/state-split return zeros instead of crashing. If synchrony finds no events, returns `no_events` pattern with all-zero percentages. Per-analysis try/except in the runner: one failure can't abort the whole pipeline.

## Compatibility

All v0.5 modules are opt-in by virtue of being new fields in `findings`. The Streamlit Clinical tab and the PDF v0.5 sections only render when the data is present. v0.4 reports remain readable; v0.5 reports include the new metrics on top.

## Installation

No new system dependencies. YASA was already in `requirements.txt` since v0.3.0-pre.

```bash
git pull
pip install -r requirements.txt
streamlit run app.py
```

## What's deferred to v0.6

From the doctor-perspective audit, these remain for the next release:

- **Structured clinical report** (Background → Findings → Impression → Recommendations ordering)
- **Recording metadata form** (current medications, recording date, reason for recording — context that changes interpretation)
- **Reactivity analysis** (eyes open vs eyes closed background separately)
- **Standardized terminology** (ILAE classification, ACNS terms)
- **Confidence intervals** on burst counts and spike rates

## Acknowledgments

The clinical-criteria thresholds in this release (CSWS at 85%, activation factor labels, synchrony classification rules) follow standard pediatric epileptology literature. Where a single agreed threshold doesn't exist (e.g. activation factor labels), the chosen breakpoints reflect what teaching texts use and what was validated against the project's reference recording.

Built on [MNE-Python](https://mne.tools), [YASA](https://github.com/raphaelvallat/yasa), [SciPy](https://scipy.org), [Streamlit](https://streamlit.io), and the prior open-source EEG community.

## License

MIT. The DISCLAIMER.md still applies: KCNQ3-Lens is not a medical device. It surfaces patterns. The doctor interprets.

---

**Full changelog**: see [CHANGELOG.md](../CHANGELOG.md)
**Commits in this release**: `ac0ae3a`
