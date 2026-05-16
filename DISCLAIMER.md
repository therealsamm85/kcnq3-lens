# Medical Disclaimer

**KCNQ3-Lens is NOT a medical device. It does NOT provide medical diagnosis, treatment, or advice.**

## What this tool is

KCNQ3-Lens is a research-grade quantitative analysis tool for EEG recordings. It surfaces patterns in EEG data that may be useful for families and clinicians to discuss together.

The tool is intended to:

- Help families understand what their child's EEG shows quantitatively
- Generate structured questions that families can bring to their treating neurologist
- Enable consistent measurement of EEG features across time and across recordings, so changes over months and years can be tracked

## What this tool is NOT

- **Not a medical device.** Not FDA-cleared, not CE-marked, not validated as clinical-grade software.
- **Not a diagnostic tool.** No output of this tool establishes, confirms, or rules out any medical condition.
- **Not a substitute for clinical EEG interpretation.** Only a qualified neurologist or epileptologist can interpret EEG data in the context of a patient's full clinical picture.
- **Not a basis for medication decisions.** No treatment should be started, stopped, or modified based on this tool's output. All therapeutic decisions belong to the treating physician.
- **Not validated against clinical gold standards.** The algorithms used are research-grade and have not been validated against expert-scored polysomnograms or formal clinical reads at scale.

## Per-analysis limitations

### Tier 1 (shipped v0.1–v0.7)

**Spike topography (kurtosis)**
Uses per-channel kurtosis as a proxy for epileptiform concentration. Kurtosis is sensitive to high-amplitude transients, including artifact. Topographic maps reflect signal statistics, not confirmed spike localization. A channel with high kurtosis may reflect muscle artifact or electrode noise rather than epileptiform activity.

**Sleep spindle density**
Interpretation labels ("below" / "in" / "above") use ±30% ranges around values from McClain 2016 (n=8 longitudinal, ages 2–5) and Kwon 2023. These are a **tool convention for intra-patient longitudinal tracking, not published clinical cutoffs.** The McClain cohort (n=8) is too small to support deterministic "abnormal" calls. Note: v0.11.1 corrected an error in earlier versions (≤v0.11.0) that cited Wamsley 2012 for pediatric norms — that paper contains no pediatric data. Revised norms are roughly 3× lower at age 5 than previously claimed.

**Background power + posterior dominant rhythm (PDR)**
PDR interpretation labels ("age_appropriate" / "mildly_slow" / "severely_slow") use a normative table based on Niedermeyer 2005 and Hagne 1968 textbook ranges. The lower bounds are conservative — tighter than some sources allow. The "severely_slow" label fires at >2 Hz below the lower bound; use this only as a flag for discussion with the clinician, not as a diagnostic statement.

**Sustained burst detection**
Detects rhythmic bursts ≥3 seconds in the 2–4 Hz range. Not equivalent to formal CSWS/ESES scoring. Sensitivity and specificity have not been validated against human-scored polysomnograms.

**Spike morphology classification**
Rule-based classification into simple spike / sharp wave / complex spike-wave. Morphology categories are based on amplitude, duration, and waveform shape heuristics — not reviewed by a neurologist for this tool. Classification accuracy varies with recording quality and electrode impedance.

**Sleep stages (YASA)**
YASA's SleepStaging model is trained on adult polysomnography. Pediatric output is flagged as `confidence='heuristic'`. Stage labels are useful for computing stage-specific metrics (SWI, spindle density) but are not a substitute for human-scored pediatric polysomnography.

**Spike-Wave Index (SWI)**
Implements the Tassinari definition (% of NREM stage time occupied by continuous SW bursts ≥1 spike/s sustained ≥3s). The CSWS/ESES criterion check (N3 SWI ≥ 85%) is a rule-based flag — diagnosis of CSWS/ESES requires clinical correlation and expert EEG review.

**Wake/sleep state split and activation factor**
Spike rates are computed separately per state using auto-detected sleep onset. Activation factor accuracy depends on sleep-window detection quality. The heuristic sleep-onset detector has a low-confidence flag and a sensible fallback but is not a substitute for polysomnography staging.

**Bilateral synchrony classification**
Classifies each detected spike into focal / regional / bilateral synchronous / bilateral asynchronous / generalized based on a ±50ms co-firing window. This is a heuristic temporal analysis — it does not perform source localization or independent component analysis.

### Tier 2 (shipped v0.13.x)

**Slow-wave detection**
Slow-oscillation (SO) density, amplitude, and duration reported per NREM3 epoch. Detection uses an amplitude-and-duration heuristic, not a validated ML detector. No pediatric normative database exists for comparison — output is descriptive only.

**HFO ripple detection (80–250 Hz)**
Uses a Staba-style energy detector. HFO detection is an active research field with no consensus algorithm or validated pediatric norms. HFO rate output is a **research metric** — it has not been validated against intracranial EEG or clinical outcome measures in this tool. Requires ≥500 Hz sampling rate; most clinical EEGs are 200–250 Hz and will show zero detections.

**SO-spindle coupling (PLV)**
Phase-locking value between slow oscillations and spindles. SO-spindle coupling is a descriptive measure of memory consolidation circuitry maturation. Coupling strength and preferred angle vary substantially with age and develop through adolescence — no age-normative ranges are currently incorporated. Treat output as descriptive only.

**IED detection (ensemble heuristic + optional SpikeNet)**
The default detector is a rule-based ensemble (morphology score + template correlation + amplitude threshold). It is **not a trained ML classifier** — the underlying SpikeNet model path is present in the code but requires locally downloaded model weights that are not distributed with the tool (stub). The heuristic ensemble has not been validated against expert-annotated spike catalogues. False positive and false negative rates are unknown.

## Privacy

- EEG files are processed **entirely on your local machine**. They are never uploaded to any server.
- If you choose to use the optional AI interpretation feature, only **derived numerical metrics** (e.g., "spindle density: 1.3 per minute") are sent to the AI provider API using your own key — never raw EEG data.
- You are responsible for the security of EEG files on your own device.

## Limitations (general)

- The algorithms make assumptions (sampling rate, channel naming, recording quality) that may not hold for all recordings.
- Age-normative reference ranges built into the tool come from published literature and may not represent all populations.
- The tool was developed with KCNQ3-spectrum patients in mind. Use for other conditions is at the user's discretion.
- The Nihon Kohden EEG-1200A reader is reverse-engineered from a single recording family. Other recordings in this format may not parse correctly.
- Tested primarily on macOS. Linux and Windows should work but have had limited maintainer testing.

## Use

By using this software, you acknowledge that:

1. You understand it is not a medical device.
2. You will not make medical decisions based solely on its output.
3. You will share any findings of interest with a qualified physician before acting on them.
4. The authors and contributors assume no liability for any use of this software.

If you are uncertain about anything this tool reports — bring the report to your child's doctor and ask. That is what the tool is built for.
