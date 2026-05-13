# Analysis audit — 2026-05-13

## Topography

- **Our result:** Fp1=8.58, F4=8.57 as top channels (full overnight, all epochs)
- **Alternative method:** Same kurtosis algorithm with Fp1/Fp2 excluded; plus epoch-level artifact classification (flag epochs where Fp1 amplitude > 3× F4 amplitude as eye-blink epochs)
- **Alternative result:** Without Fp1/Fp2: F4=7.43 is #1, Cz=6.67, T6=5.50. Zero epochs flagged as eye-blink artifacts in the first 200 epochs tested. Fp1 leads F4 by only 0.04 kurtosis points on the full overnight (7.47 vs 7.43, sampled).
- **Discrepancy:** ~1.0× — Fp1 barely edges F4, but the margin is within sampling noise
- **Verdict:** NEEDS_VALIDATION_FROM_USER
- **Recommendation:** Fp1's narrow lead over F4 is not artifact-driven (no eye-blink epochs detected during this overnight), but Fp1 ranking above F4 should be interpreted cautiously — the true focus is likely F4/right frontal. Consider excluding Fp1/Fp2 from the displayed top channels in the family-facing report, or annotating them as "artifact-prone." The algorithm itself is sound; the ranking instability at this margin is the concern.

---

## Background

- **Our result:** PDR = 4.0 Hz on overnight "wake" epochs (5–15% and 85–95% of recording); interpretation = severely_slow; delta=80%, alpha=3%, DAR=24
- **Alternative method:** Independent Welch PSD on same overnight at different time windows (start, mid-sleep, end); crosscheck on 4 daytime EEGs via MNE NihonKohden reader
- **Alternative result:** PDR = 4.0–4.4 Hz consistently across ALL time windows of the overnight (including first and last 30 min). All four daytime EEGs also show 4.0–4.9 Hz peak across O1/O2 with no window ever reaching 8 Hz. Note: daytime EEGs appear to have electrode/reference issues (amplitudes >>1000 µV on temporal channels), limiting their usefulness as wake ground truth.
- **Discrepancy:** ~1.0× — overnight and daytime agree on theta-dominant, 4–5 Hz spectrum
- **Verdict:** OK (result is likely real)
- **Recommendation:** The 4.0 Hz PDR is consistent across the entire 24h recording and across all available daytime EEGs, strongly suggesting this reflects genuine diffuse background slowing rather than a window-selection artifact. One caveat: the daytime EEGs have very high-amplitude artifacts on occipital channels (possibly electrode contact or reference issues), so they cannot fully confirm true alert-wake PDR. Flag this in the report as "consistent with severe background slowing; independent clinical confirmation recommended" rather than relying solely on the automated result. The window selection heuristic (5–15% / 85–95%) is reasonable for 24h recordings but could capture drowsy states; exposing a `wake_epoch_indices` parameter to the family would allow manual override.

---

## Bursts

- **Our result:** Reported as 462 bursts ≥3s, 104 ≥10s on full overnight; sleep window (epochs 1080–1800) gives 116 bursts ≥3s, 13 ≥10s, median 5.4s, max 22.7s; longest bursts show 18/18 channels involved
- **Alternative method:** Threshold sensitivity sweep (3×–10×MAD) on Pz; per-channel baseline amplitude check; direct inspection of the `n_channels_involved` metric at baseline
- **Alternative result:**
  - Threshold sensitivity: 3×MAD→160 bursts, 4×MAD→116, 6×MAD→76, 10×MAD→51 (≥3s, sleep window). A 3.1× range across the multiplier sweep.
  - `n_channels_involved` is BROKEN: in the 5–25 Hz bandpass, 18/19 EEG channels exceed 500 µV peak-to-peak during background epochs. The 500 µV criterion is always satisfied, so every burst trivially shows 18 channels involved. This metric provides zero discriminative information.
  - The core burst detection on Pz appears valid: MAD=39 µV, threshold=156 µV, signal reaches 26,870 µV during large events. The 19s burst at 00:20:40 is confirmed real (verified visually). But the count of 116 carries ±3× uncertainty from threshold choice.
- **Discrepancy:** n_channels_involved: completely unreliable (always 18/18). Burst count: 3× range across threshold multipliers.
- **Verdict:** NEEDS_FIX (two separate bugs)
- **Recommendation:** (1) Fix `n_channels_involved` — replace the fixed 500 µV p-p threshold with a per-channel adaptive threshold (e.g., median_p-p × 3 during non-burst baseline). (2) Add a confidence qualifier to burst counts ("low/medium/high" based on amplitude above threshold); consider reporting only bursts ≥10× MAD as "high-confidence" events rather than flat 4×MAD. Do NOT present 18/18 channel involvement as a validation metric until this is fixed.

---

## Morphology

- **Our result:** 17/39/44% simple/sharp/complex; classification = "mixed"; n_events = 30,880; 85.8 events/min
- **Alternative method:** Chunk-wise analysis (50-epoch windows) to test whether the global MAD is representative; direct inspection of detected peak amplitudes in windows with different noise floors
- **Alternative result:**
  - Event rate varies 4.6–128 events/min across 50-epoch chunks of the same sleep window. In a quiet chunk (epochs 1580–1630), 932 events were detected at 37/min despite the signal never exceeding 2× threshold — median detected peak amplitude was 84 µV vs. threshold 76 µV. These are noise peaks, not epileptiform spikes.
  - Root cause: the MAD is computed over the entire 6-hour concatenated trace. Large CSWS bursts inflate the global MAD, raising the threshold. But in quiet inter-burst intervals where local noise is ~50–60 µV, this inflated threshold (103 µV) still catches almost every peak in the 10–30 Hz band. Conversely, in a 50-epoch quiet sub-window, MAD=7 µV and threshold=42 µV catches everything above 42 µV.
  - The FWHM classification itself (simple/sharp/complex ratios) may be valid for true events but is meaningless when 50–90% of "detected" events are band-pass noise peaks.
- **Discrepancy:** Rate comparison: 85.8/min (ours) vs. ~5–15/min in neurologically similar children from published literature. Factor of ~6–17×. The morphology distribution (17/39/44%) cannot be trusted until the event count is fixed.
- **Verdict:** NEEDS_FIX (same structural problem as the spindle detector)
- **Recommendation:** The detection step must use a per-epoch or sliding-window MAD, not a global MAD over the full concatenated trace. Alternatively, add a hard floor: only count events where peak amplitude ≥ 3× the local (epoch-level) RMS. The FWHM classifier itself (lines 93–100) is sound and can be reused once the detection gating is fixed. Consider cross-validating against MNE's spike detector or a published automated spike detector (e.g., SpikeNet, EpiDetector) before publishing.

---

## Overall priority list

1. **Morphology (CRITICAL)** — 85.8 events/min is 6–17× literature rates; global-MAD over concatenated trace detects noise peaks in quiet intervals. Structurally identical to the spindle over-count bug. Fix the detection gate first; the FWHM classifier is fine.

2. **Bursts / n_channels_involved (HIGH)** — The `n_channels_involved` field always reports 18/18 due to a 500 µV fixed threshold that is orders of magnitude below baseline signal amplitudes in this recording. Burst counts themselves are directionally valid but carry ±3× uncertainty. The "18 channels involved" claim in family-facing output is misleading and must be fixed before publication.

3. **Background PDR (LOW / monitor)** — 4.0 Hz appears to be a real finding, consistent across the full 24h recording. The algorithm is correct. Flag to user: daytime EEGs have electrode/reference quality problems that prevent independent confirmation; clinical neurologist review of the PDR claim is required before v0.3.

4. **Topography (LOW / cosmetic)** — Fp1 narrowly outranks F4 by 0.04 kurtosis points, not driven by detectable eye-blink artifacts. No bug, but the margin is noise-level. Recommend suppressing Fp1/Fp2 from the top-channels display or annotating them as artifact-prone in the family report. Underlying algorithm is sound.
