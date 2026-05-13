# KCNQ3-Lens v0.4.1 — Hardened release

**The first release intended for use by other families.** v0.4.1 closes out the v0.1 → v0.4 build cycle with a hardened pattern-matcher, a complete edge-case test suite, and confirmed bilingual (EN / DE) UI coverage.

## What this tool is

KCNQ3-Lens is an open-source quantitative EEG analysis tool for families of children with rare epilepsies — especially the KCNQ3 spectrum. It runs entirely on your local machine, supports the Nihon Kohden EEG-1200A format that no other open-source tool reads correctly, and produces a structured report you can take to your child's neurologist.

It is **not** a medical device and **does not diagnose** anything. It surfaces patterns. The doctor interprets.

## Headline features

- **Nihon Kohden EEG-1200A reader** (`.eeg` files from EEG-2100 / EEG-2200 systems) — novel
- **Seven quantitative analyses**: topography, sleep spindles (YASA-validated), background power, sustained rhythmic bursts, spike-wave morphology, time-of-night spike burden, quality control
- **Pre / post-treatment comparison mode** with directional deltas — useful for tracking medication response over time
- **Proactive clinical insights** (new in 0.4.0): anatomical mapping of affected brain regions, recognition of clinical patterns (KCNQ-spectrum, CSWS, BECTS, SMA-predominance), and cross-modal observations
- **Multi-AI optional interpretation**: choose Claude (Anthropic), GPT (OpenAI), or Gemini (Google) — bring your own API key
- **Doctor and parent PDF reports**
- **EN / DE UI** with zero translation gaps
- **Privacy by architecture**: raw EEG never leaves your machine; only derived numerical metrics are sent to the chosen AI provider

## What changed since v0.4.0

Two real false-positive bugs in the new pattern matcher, caught by the edge-case test suite that's also part of this release:

- The BECTS pattern fired on **empty findings** because criteria like `(pct_complex or 0) < 30` evaluated `True` on missing data.
- The BECTS pattern fired on **normal / healthy findings** at 67% moderate confidence — because a normal EEG legitimately satisfies its supporting criteria (high simple-spike %, low complex %) without any centro-temporal focus.

Fix: `PatternCriterion.required` gating. Each pattern now has at least one criterion that must be met for the pattern to appear at all. Without the gate, the pattern is silently dropped — not shown as "weak" or "moderate". All four patterns still correctly match the reference KCNQ3 R230H profile.

## Bugs caught across the v0.3 → v0.4.1 line

This project's track record on catching problems before they ship is, frankly, more useful than the feature list:

1. **YASA-vs-heuristic spindle audit (v0.3.0-pre)** — the heuristic spindle detector over-counted by ~150×. Replaced with the validated YASA backend.
2. **Morphology global-MAD over-counting (v0.3.0)** — spike event rates were 6–17× literature norms because a single MAD over the full sleep window caught noise peaks in quiet intervals. Per-epoch local MAD fixed it.
3. **Bursts `n_channels_involved` always returned 18/19 (v0.3.0)** — the fixed 500-amplitude threshold was below baseline signal amplitude on NK recordings. Adaptive per-channel baseline fixed it.
4. **Pattern matcher false positives (v0.4.1)** — described above.

Every one of these would have shipped quietly without the explicit audit step. The discipline of "validate every analysis against an alternative on real data" is what holds this together.

## Installation

```bash
git clone https://github.com/therealsamm85/kcnq3-lens.git
cd kcnq3-lens
python -m venv .venv
source .venv/bin/activate            # Linux/Mac
# .venv\Scripts\activate             # Windows
pip install -r requirements.txt
streamlit run app.py
```

The browser opens at `http://localhost:8501`. Drag and drop an EEG file in the supported formats (Nihon Kohden `.eeg`, EDF/EDF+, BDF, BrainVision `.vhdr`, EEGLAB `.set`).

## Supported recordings

| Requirement | Minimum | Recommended |
|---|---|---|
| Duration | 30 minutes | 6+ hours including sleep |
| Sampling rate | 100 Hz | 200–500 Hz |
| EEG channels | 6 (frontal + central + posterior) | 19 (full 10-20 system) |
| Child age | 2–18 years | 3–12 years |

See README for the full scope and "what this tool is NOT for" section.

## Known limitations

- The Nihon Kohden EEG-1200A reader has been tested on one recording family. Other recordings in this format may need verification.
- Clinical-pattern thresholds were tuned against a single reference recording. They will need refinement as the tool reaches more families.
- The auto sleep-onset detector is heuristic. It flags low confidence and falls back to a conventional overnight window if it fails, but it is not a substitute for proper polysomnography.
- Tested on macOS only. Linux and Windows should work but have not been verified by maintainers.

## How to help

- **Use it on your child's EEG and report what worked / didn't.** Issues, screenshots, and EEG files (anonymized) at https://github.com/therealsamm85/kcnq3-lens/issues
- **Validate the patterns against your own clinical experience** if you are a clinician. Each pattern is encoded as explicit criteria in `src/insights/patterns.py` — they can be refined as PRs.
- **Add a new Nihon Kohden reader variant** if your recording uses a different layout — the reader file lives at `src/readers/nihon_kohden.py` and the format is documented in the module docstring.
- **Translate the UI** to another language — adding a new entry to `src/i18n/translations.py` takes about 30 minutes per language.

## Acknowledgments

Built on the shoulders of [MNE-Python](https://mne.tools), [YASA](https://github.com/raphaelvallat/yasa), [SciPy](https://scipy.org), and [Streamlit](https://streamlit.io). LLM interpretation supported by [Anthropic](https://anthropic.com), [OpenAI](https://openai.com), and [Google](https://ai.google.dev).

Thanks to the [RIKEE](https://rikee.org) registry, the Cooper Lab at Baylor, the Weckhuysen Lab at Antwerp, and the N=1 Collaborative for keeping the KCNQ research community connected.

And to one specific child whose recordings forced every algorithm in this tool to be honest. You know who you are.

## License

MIT — free to use, modify, distribute, including for commercial purposes. The DISCLAIMER.md still applies: this is not a medical device.

---

**Full changelog**: see [CHANGELOG.md](../CHANGELOG.md)
**Commits in this release line**: `5067970`, `61e5862`, `3f59946`, `dfc62ce`, `79ebf4b`, `8c8d611`
